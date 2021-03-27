//! Generic geometry (G-buffer) generation pass
use crate::{camera::Camera, mesh::Vertex3D, shader::create_shader_module};
use glam::{Mat4, Vec4};
use graal::{ash::version::DeviceV1_0, vk, FragmentOutputInterface, VertexInputInterfaceExt};
use inline_spirv::include_spirv;
use std::ptr;
use crate::scene::Scene;

static GEOMETRY_PASS_SHADER_VERT: &[u32] = include_spirv!("shaders/mesh_vis.vert", vert);
static GEOMETRY_PASS_SHADER_FRAG: &[u32] = include_spirv!("shaders/mesh_vis.frag", frag);

// --- Uniforms ------------------------------------------------------------------------------------

/// Per-scene uniforms.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Globals {
    u_view_matrix: Mat4,
    u_proj_matrix: Mat4,
    u_view_proj_matrix: Mat4,
    u_inverse_proj_matrix: Mat4,
}

/// Per-material uniforms.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Material {
    u_color: Vec4,
}

/// Per-object uniforms
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct PerObject {
    u_model_matrix: Mat4,
    u_model_it_matrix: Mat4,
}

// --- Shader interfaces ---------------------------------------------------------------------------
#[derive(graal::DescriptorSetInterface)]
#[repr(C)]
struct GlobalsInterface {
    #[layout(binding = 0, uniform_buffer, stages(vertex, fragment))]
    globals: graal::BufferDescriptor<Globals>,
}

#[derive(graal::DescriptorSetInterface)]
#[repr(C)]
struct PerObjectInterface {
    #[layout(binding = 0, uniform_buffer, stages(vertex))]
    per_object: graal::BufferDescriptor<PerObject>,
}

#[derive(graal::DescriptorSetInterface)]
#[repr(C)]
struct MaterialsInterface {
    #[layout(binding = 0, uniform_buffer, stages(fragment))]
    material: graal::BufferDescriptor<Material>,
}

#[derive(Copy, Clone, Debug, graal::VertexInputInterface)]
struct MeshVertexInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: graal::VertexBufferView<Vertex3D>,
}

#[derive(graal::FragmentOutputInterface, Copy, Clone, Debug)]
pub struct GBuffers {
    /// Color buffer, R16G16B16A16_SFLOAT
    #[attachment(
        color,
        format = "R16G16B16A16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    pub color: graal::ImageInfo,

    /// Normals, RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    pub normal: graal::ImageInfo,

    /// Tangents: RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    pub tangent: graal::ImageInfo,

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

// Problem: can't re-use a group of resources (like a FragmentOutputInterface)
// Solution: also derive PassResources on the group of resources?
//
// Problem: resources can be accessed in different ways depending on the pass
//
// Bigger problem: must register access of ALL VERTEX BUFFERS in the scene
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
                .device()
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

        let color_blend_attachments = &[vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: Default::default(),
            dst_color_blend_factor: Default::default(),
            color_blend_op: Default::default(),
            src_alpha_blend_factor: Default::default(),
            dst_alpha_blend_factor: Default::default(),
            alpha_blend_op: Default::default(),
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
        }];

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
                .device()
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
        batch: &graal::Batch<'a>,
        scene: &'a Scene,
        target_size: (u32, u32),
        camera: &Camera,
    ) -> GBuffers
    {
        // allocate the G-buffers of the frame
        let g: GBuffers = GBuffers::new(
            batch,
            vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_SRC,
            target_size,
        );

        // setup uniforms & descriptors
        let global_uniforms = batch.upload(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &Globals {
                u_view_matrix: camera.view,
                u_proj_matrix: camera.projection,
                u_view_proj_matrix: camera.projection * camera.view,
                u_inverse_proj_matrix: camera.projection.inverse(),
            },
            None,
        );

        let material_uniforms = batch.upload(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &Material {
                u_color: Vec4::new(0.0, 1.0, 0.0, 1.0),
            },
            None,
        );

        let per_object_uniforms = batch.upload(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &PerObject {
                u_model_matrix: Mat4::IDENTITY,
                u_model_it_matrix: Mat4::IDENTITY,
            },
            None,
        );

        // setup the pass
        batch.add_graphics_pass("gbuffers", |pass| {
            // we don't really need to register those because
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
            );

            let pipeline = self.pipeline;
            let pipeline_layout = self.pipeline_layout;
            let render_pass = self.render_pass;

            pass.set_commands(move |context, cb| unsafe {
                let framebuffer = g.create_framebuffer(context, target_size);

                let clear_values = &[
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.05, 0.1, 0.15, 1.0],
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        color: vk::ClearColorValue {
                            float32: [0.0, 0.0, 0.0, 1.0],
                        },
                    },
                    vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue {
                            depth: 1.0,
                            stencil: 0,
                        },
                    },
                ];

                let render_pass_begin_info = vk::RenderPassBeginInfo {
                    render_pass,
                    framebuffer,
                    render_area: vk::Rect2D {
                        offset: vk::Offset2D { x: 0, y: 0 },
                        extent: vk::Extent2D {
                            width: target_size.0,
                            height: target_size.1,
                        },
                    },
                    clear_value_count: 4,
                    p_clear_values: clear_values.as_ptr(),
                    ..Default::default()
                };

                context.device().cmd_begin_render_pass(
                    cb,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );

                context.device().cmd_set_viewport(
                    cb,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: target_size.0 as f32,
                        height: target_size.1 as f32,
                        min_depth: 0.0,
                        max_depth: 1.0,
                    }],
                );
                context.device().cmd_set_scissor(
                    cb,
                    0,
                    &[vk::Rect2D {
                        offset: Default::default(),
                        extent: vk::Extent2D {
                            width: target_size.0,
                            height: target_size.1,
                        },
                    }],
                );

                context
                    .device()
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);

                let globals_set = context.create_descriptor_set(&GlobalsInterface {
                    globals: global_uniforms.into(),
                });

                // loop over all objects in the scene
                for obj in scene.objects().values() {

                    let mesh = if let Some(m) = scene.mesh(obj.mesh) { m } else { continue };

                    let materials_set = context.create_descriptor_set(&MaterialsInterface {
                        material: material_uniforms.into(),
                    });

                    let per_object_set = context.create_descriptor_set(&PerObjectInterface {
                        per_object: per_object_uniforms.into(),
                    });


                    context.device().cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[globals_set, materials_set, per_object_set],
                        &[],
                    );

                    context
                        .device()
                        .cmd_bind_vertex_buffers(cb, 0, &[mesh.vertex_buffer.handle], &[0]);

                    context.device().cmd_draw(cb, mesh.vertex_count as u32, 1, 0, 0);
                }

                context.device().cmd_end_render_pass(cb);
            });
        });

        g
    }
}
