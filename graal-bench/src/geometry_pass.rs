//! Generic geometry (G-buffer) generation pass
use crate::{camera::Camera, mesh::Vertex3D, scene::Scene, shader::create_shader_module};
use glam::{Mat4, Vec4};
use graal::{ash::version::DeviceV1_0, vk, FragmentOutputInterface, VertexInputInterfaceExt};
use inline_spirv::include_spirv;
use std::ptr;
use egui::CursorIcon::Default;

static GEOMETRY_PASS_SHADER_VERT: &[u32] = include_spirv!("shaders/mesh_vis.vert", vert);
static GEOMETRY_PASS_SHADER_FRAG: &[u32] = include_spirv!("shaders/mesh_vis.frag", frag);

// --- Uniforms ------------------------------------------------------------------------------------

// --- Shader interfaces ---------------------------------------------------------------------------

#[derive(mlr::ShaderArguments)]
#[repr(C)]
struct SceneArguments {
    // uniform variables will be put in a single uniform buffer, at location 0
    u_view_matrix: Mat4,
    u_proj_matrix: Mat4,
    u_view_proj_matrix: Mat4,
    u_inverse_proj_matrix: Mat4,
}
// to create: `mlr::ArgumentBlock::new(SceneArguments { ... })`

#[derive(mlr::ShaderArguments)]
#[repr(C)]
struct MaterialArguments {
    u_color: Vec4,
}

// draw<Push: PushConstants>(..., push_constants: Option<Push>, ...)

// problem:
// 1. descriptor set will be created in the command callback, because we can't create a descriptor set before the resources are allocated
// 2. ShaderArguments can borrow data
// 3. BUT the command closure passed to `graal::Context` has a 'static bound
// => this doesn't work.
//
// Solution:
// A. ShaderArguments don't borrow stuff
// B. Make it so that command callbacks can borrow stuff
//      -> issue: they will be borrowed for *the whole frame*, which means that we won't be able to access an image mutably in subsequent passes
// C. Make MLR resources refcounted (bleh)
//
// ShaderArguments borrow stuff
// -> just before moving into pass callback, write VkDescriptor{Image,Buffer}Info in an auxiliary struct
// -> then just move this aux struct in the callback
//
// But, in the first place, how are we supposed to submit multiple draw calls into a single pass?
// -> we can't run a loop within the pass callback, since we can't *borrow* any of our resources in the pass callback
// -> the mlr::draw call won't happen within a pass callback
// -> must fuse consecutive draw calls without data dependencies into the same pass
//      -> must detect whether there's a data dependency between two consecutive draw calls
//      -> if there's one, end pass, submit batch
//      -> enum PendingDrawCall

#[derive(mlr::ShaderArguments)]
#[argument(dynamic_uniform_buffer)] // use a dynamic uniform buffer
#[repr(C)]
struct ObjectDataUniform<'a> {
    matrix: Mat4,
}

// to create: `mlr::ArgumentBlock::new(ObjectDataUniform { ... })`
// internally:
// - will create/reuse one single descriptor set for the type, and for the current frame, that points to the upload buffer
// - will upload the uniform data to an upload buffer
// - on bind, set the buffer offset

// draw<Push: PushConstants>(..., push_constants: Option<Push>, ...)
#[derive(mlr::PushConstants)]
struct ObjectData {
    matrix: Mat4,
    // texture index, instance index, etc.
}

#[derive(Copy, Clone, Debug, mlr::VertexInputInterface)]
struct MeshVertexInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: mlr::Buffer<Vertex3D>,
}

#[derive(mlr::FragmentOutputInterface, Copy, Clone, Debug)]
pub struct GBuffers<'a> {
    /// Color buffer
    #[attachment(color, load_op = "CLEAR", store_op = "STORE")]
    pub color: &'a mlr::Image,

    /// Normals
    #[attachment(color, load_op = "CLEAR", store_op = "STORE")]
    pub normal: &'a mlr::Image,

    /// Tangents
    #[attachment(color)]
    pub tangent: &'a mlr::Image,

    /// Depth: D32_SFLOAT
    #[attachment(
        depth,
        format = "D32_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "DEPTH_STENCIL_ATTACHMENT_OPTIMAL"
    )]
    pub depth: graal::ImageInfo,
}

// ShaderOutput
// fn add_color_target(attachment: usize, &mut image, load_op, store_op, blend_mode)
// fn add_color_target(attachment: usize, &mut image, load_op, store_op, blend_mode)

// -> internally, keeps track of which shader interfaces it was validated against
// -> re-creating it is not a big deal: cache it internally

// Problem: can't re-use a group of resources (like a FragmentOutputInterface)
// Solution: also derive PassResources on the group of resources?
//
// Problem: resources can be accessed in different ways depending on the pass
/*#[derive(PassResources)]
pub struct GeometryPassResources {
    #[access("ColorAttachmentReadWrite")]
    pub color: graal::ImageInfo,
    #[access("ColorAttachmentReadWrite")]
    pub normal: graal::ImageInfo,
    #[access("ColorAttachmentReadWrite")]
    pub tangent: graal::ImageInfo,
    #[access("DepthStencilAttachmentReadWrite")]
    pub depth: graal::ImageInfo,
}*/

/// A pass that renders mesh data to output buffers in various ways.
pub struct GeometryPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

// let's say I want to add a texture pass:
// - modify the Shader
// - modify the MaterialsInterface to take an image descriptor
// - register the use of the texture in the pass
// - create the image view
// - put the image view in the descriptor set

// With bindless:
// - modify the shader
// - register the use of the texture in the pass
// -

impl GeometryPass {
    //
    pub fn new(context: &mut graal::Context) -> GeometryPass {
        let vert = create_shader_module(context, GEOMETRY_PASS_SHADER_VERT);
        let frag = create_shader_module(context, GEOMETRY_PASS_SHADER_FRAG);

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert,
                p_name: b"main\0".as_ptr() as *const i8,
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag,
                p_name: b"main\0".as_ptr() as *const i8,
                ..Default::default()
            },
        ];

        let mut set_layouts = [
            context.get_or_create_descriptor_set_layout_for_interface::<GlobalsInterface>(),
            context.get_or_create_descriptor_set_layout_for_interface::<MaterialsInterface>(),
            context.get_or_create_descriptor_set_layout_for_interface::<PerObjectInterface>(),
        ];

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            context
                .vulkan_device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let render_pass = context.get_or_create_render_pass_from_interface::<GBuffers>();

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: 0,
            ..Default::default()
        };

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            patch_control_points: 0,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            p_viewports: ptr::null(),
            scissor_count: 1,
            p_scissors: ptr::null(),
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: vk::PolygonMode::FILL,
            cull_mode: vk::CullModeFlags::NONE,
            front_face: vk::FrontFace::CLOCKWISE,
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: 0,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: vk::TRUE,
            depth_write_enable: vk::TRUE,
            depth_compare_op: vk::CompareOp::LESS,
            depth_bounds_test_enable: vk::FALSE,
            stencil_test_enable: vk::FALSE,
            front: Default::default(),
            back: Default::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 1.0,
            ..Default::default()
        };

        let no_blend = vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            ..Default::default()
        };

        let color_blend_attachments = &[no_blend, no_blend, no_blend];

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            flags: Default::default(),
            logic_op_enable: 0,
            logic_op: Default::default(),
            attachment_count: color_blend_attachments.len() as u32,
            p_attachments: color_blend_attachments.as_ptr(),
            blend_constants: [0.0f32; 4],
            ..Default::default()
        };

        let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        let dynamic_state = vk::PipelineDynamicStateCreateInfo {
            flags: Default::default(),
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        let gpci = vk::GraphicsPipelineCreateInfo {
            flags: Default::default(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &MeshVertexInput::get_pipeline_vertex_input_state_create_info(),
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &tessellation_state,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state,
            layout: pipeline_layout,
            render_pass,
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            context
                .vulkan_device()
                .create_graphics_pipelines(vk::PipelineCache::null(), &[gpci], None)
                .unwrap()[0]
        };

        GeometryPass {
            pipeline,
            pipeline_layout,
            render_pass,
        }
    }

    /*fn allocate_buffers(&self, batch: &graal::Batch, size: (u32, u32)) -> GBuffers {
        let color =
            create_transient_gbuffer_color_image(batch, vk::Format::R16G16B16A16_SFLOAT, size);
        let normal = create_transient_gbuffer_color_image(batch, vk::Format::R16G16_SFLOAT, size);
        let tangent = create_transient_gbuffer_color_image(batch, vk::Format::R16G16_SFLOAT, size);

        let depth = batch.create_transient_image(
            "depth_buffer",
            &graal::ResourceMemoryInfo::DEVICE_LOCAL,
            &graal::ImageResourceCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
                    | vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::STORAGE,
                format: vk::Format::D32_SFLOAT,
                extent: vk::Extent3D {
                    width: size.0,
                    height: size.1,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                tiling: vk::ImageTiling::OPTIMAL,
            },
        );

        GBuffers {
            color,
            normal,
            tangent,
            depth,
        }
    }*/

    pub fn run<'a>(
        &self,
        frame: &graal::Frame<'a>,
        scene: &'a Scene,
        target_size: (u32, u32),
        camera: &Camera,
    ) -> GBuffers {
        // allocate the G-buffers of the frame
        let g: GBuffers = GBuffers::new(
            frame,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            target_size,
        );

        // setup the pass
        frame.add_graphics_pass("gbuffers", |pass| {
            /*// we don't really need to register those
            pass.register_buffer_access(
                global_uniforms.id,
                graal::AccessType::AnyShaderReadUniformBuffer,
            );
            pass.register_buffer_access(
                material_uniforms.id,
                graal::AccessType::AnyShaderReadUniformBuffer,
            );
            pass.register_buffer_access(
                per_object_uniforms.id,
                graal::AccessType::AnyShaderReadUniformBuffer,
            );

            //pass.register_buffer_access(vertex_buffer.id, graal::AccessType::VertexAttributeRead);
            pass.register_image_access(g.color.id, graal::AccessType::ColorAttachmentReadWrite);
            pass.register_image_access(g.normal.id, graal::AccessType::ColorAttachmentReadWrite);
            pass.register_image_access(g.tangent.id, graal::AccessType::ColorAttachmentReadWrite);
            pass.register_image_access(
                g.depth.id,
                graal::AccessType::DepthStencilAttachmentReadWrite,
            );*/

            pass.set_commands(move |context, cb| unsafe {

                let fragment_output = FragmentOutput::builder()
                    .add_color_attachment(
                        &g.color,
                        AttachmentLoadOp::Clear {
                            value: vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.05, 0.1, 0.15, 1.0],
                                },
                            },
                        },
                    )
                    .add_color_attachment(
                        &g.normal,
                        AttachmentLoadOp::Clear {
                            value: vk::ClearValue {
                                color: vk::ClearColorValue {
                                    float32: [0.05, 0.1, 0.15, 1.0],
                                },
                            },
                        },
                    )
                    .add_color_attachment(
                        &g.depth,
                        AttachmentLoadOp::Clear {
                            value: vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 1.0,
                                    stencil: 0,
                                },
                            },
                        },
                    )
                    .build();

                let globals = ArgumentBlock::new(Globals {
                    u_view_matrix: camera.view,
                    u_proj_matrix: camera.projection,
                    u_view_proj_matrix: camera.projection * camera.view,
                    u_inverse_proj_matrix: camera.projection.inverse(),
                });

                // loop over all objects in the scene
                for obj in scene.objects().values() {
                    let mesh = if let Some(m) = scene.mesh(obj.mesh) {
                        m
                    } else {
                        continue;
                    };

                    let material = ArgumentBlock::new(MaterialArguments {
                        u_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
                    });

                    let per_object = ArgumentBlock::new(ObjectData {
                        u_model_matrix: Mat4::IDENTITY,
                        u_model_it_matrix: Mat4::IDENTITY,
                    });

                    let vertex_input = VertexInput::new();

                    ctx.draw(
                        pipeline,
                        // arguments
                        &[&globals, &material, &per_object],
                        // framebuffer
                        fragment_output,
                        // vertex input (vertex and index buffers)
                        &[vertex_input],
                        // draw params
                        DrawParams { vertex_count: mesh.vertex_count, instance_count: 1, ..Default::default() }
                    );

                }

                context.vulkan_device().cmd_end_render_pass(cb);
            });
        });

        g
    }
}
