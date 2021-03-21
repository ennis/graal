mod blit;
mod bounding_box;
mod camera;
mod load_image;
mod pipeline;

use inline_spirv::include_spirv;
use raw_window_handle::HasRawWindowHandle;
use std::{mem, path::Path, ptr};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use crate::load_image::{load_image, LoadedImageInfo};

use glam::*;
use graal::{ash::version::DeviceV1_0, vk, TypedBufferInfo, VertexInputInterfaceExt};

static BACKGROUND_SHADER_VERT: &[u32] = include_spirv!("shaders/background.vert", vert);
static BACKGROUND_SHADER_FRAG: &[u32] = include_spirv!("shaders/background.frag", frag);

static MESH_VIS_SHADER_VERT: &[u32] = include_spirv!("shaders/mesh_vis.vert", vert);
static MESH_VIS_SHADER_FRAG: &[u32] = include_spirv!("shaders/mesh_vis.frag", frag);

// --- Uniform structs -----------------------------------------------------------------------------
#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Globals {
    u_view_matrix: glam::Mat4,
    u_proj_matrix: glam::Mat4,
    u_view_proj_matrix: glam::Mat4,
    u_inverse_proj_matrix: glam::Mat4,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct Material {
    u_color: glam::Vec4,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct PerObject {
    u_model_matrix: Mat4,
    u_model_it_matrix: Mat4,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct BackgroundUniforms {
    u_resolution: [f32; 2],
    u_scroll_offset: [f32; 2],
    u_zoom: f32,
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
    #[layout(binding = 0, uniform_buffer, stages(vertex))]
    material: graal::BufferDescriptor<Material>,
}

#[derive(graal::DescriptorSetInterface)]
#[repr(C)]
struct BackgroundShaderInterface {
    #[layout(binding = 0, uniform_buffer, stages(fragment))]
    uniforms: graal::BufferDescriptor<BackgroundUniforms>,
}

// --- Vertex types --------------------------------------------------------------------------------
#[derive(Copy, Clone, Debug, graal::VertexData)]
#[repr(C)]
struct Vertex3D {
    position: Vec3,
    normal: Vec3,
    tangent: Vec3,
}

#[derive(Copy, Clone, Debug, graal::VertexData)]
#[repr(C)]
struct Vertex2D {
    position: [f32; 2],
    texcoords: [graal::Norm<u16>; 2],
}

impl Vertex2D {
    pub fn new(position: [f32; 2], texcoords: [f32; 2]) -> Vertex2D {
        Vertex2D {
            position,
            texcoords: [texcoords[0].into(), texcoords[1].into()],
        }
    }
}

#[derive(Copy, Clone, Debug, graal::VertexInputInterface)]
struct Vertex2DInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: graal::VertexBufferView<Vertex2D>,
}

#[derive(Copy, Clone, Debug, graal::VertexInputInterface)]
struct MeshVertexInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: graal::VertexBufferView<Vertex3D>,
}

fn create_color_attachment_description(
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
) -> vk::AttachmentDescription {
    vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::MAY_ALIAS,
        format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }
}

/// Creates a render pass with a single subpass, writing to a single color target with the specified
/// format.
fn create_single_color_target_render_pass(
    device: &graal::ash::Device,
    target_format: vk::Format,
) -> vk::RenderPass {
    let render_pass_attachments = &[vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::empty(),
        format: target_format,
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::DONT_CARE,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let color_attachments = &[vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    }];

    let subpasses = &[vk::SubpassDescription {
        flags: Default::default(),
        pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
        input_attachment_count: 0,
        p_input_attachments: ptr::null(),
        color_attachment_count: color_attachments.len() as u32,
        p_color_attachments: color_attachments.as_ptr(),
        p_resolve_attachments: ptr::null(),
        p_depth_stencil_attachment: ptr::null(),
        preserve_attachment_count: 0,
        p_preserve_attachments: ptr::null(),
    }];

    // render pass
    let render_pass_create_info = vk::RenderPassCreateInfo {
        flags: Default::default(),
        attachment_count: render_pass_attachments.len() as u32,
        p_attachments: render_pass_attachments.as_ptr(),
        subpass_count: subpasses.len() as u32,
        p_subpasses: subpasses.as_ptr(),
        dependency_count: 0,
        p_dependencies: ptr::null(),
        ..Default::default()
    };

    let render_pass = unsafe {
        device
            .create_render_pass(&render_pass_create_info, None)
            .unwrap()
    };

    render_pass
}

fn create_shader_module(context: &mut graal::Context, code: &[u32]) -> vk::ShaderModule {
    unsafe {
        context
            .device()
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    flags: Default::default(),
                    code_size: code.len() * 4,
                    p_code: code.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .expect("failed to create shader module")
    }
}

/// A pass that renders mesh data to output buffers in various ways.
struct MeshRenderPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

#[derive(graal::FragmentOutputInterface)]
struct GBuffers {
    /// Color buffer, R16G16B16A16_SFLOAT
    #[attachment(
        color,
        format = "R16G16B16A16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    color: graal::ImageInfo,

    /// Normals, RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    normal: graal::ImageInfo,

    /// Tangents: RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    tangent: graal::ImageInfo,

    /// Depth: D32_SFLOAT
    #[attachment(
        depth,
        format = "D32_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "DEPTH_STENCIL_ATTACHMENT_OPTIMAL"
    )]
    depth: graal::ImageInfo,

    #[framebuffer]
    framebuffer: vk::Framebuffer,
}

/*#[derive(PipelineInterface)]
#[allow_additional_descriptor_sets(3)]
#[allow_additional_push_constants]
#[vertex_shader(MESH_VIS_SHADER_VERT)]
#[fragment_shader(MESH_VIS_SHADER_VERT)]
struct MeshPipelineInterface
{
    #[vertex_input] vertex_input: MeshVertexInput,
    #[render_pass] render_pass: vk::RenderPass,
    #[fragment_output] attachments: GBuffers,
    #[descriptor_set(0)] globals: GlobalsInterface,
    #[descriptor_set(1)] material: MaterialsInterface,
    #[descriptor_set(2)] per_object: PerObjectInterface,
}*/

fn create_transient_gbuffer_color_image(
    batch: &graal::Batch,
    format: vk::Format,
    size: (u32, u32),
) -> graal::ImageInfo {
    batch.create_transient_image(
        "color_buffer",
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::ImageResourceCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            usage: vk::ImageUsageFlags::COLOR_ATTACHMENT
                | vk::ImageUsageFlags::SAMPLED
                | vk::ImageUsageFlags::STORAGE,
            format,
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
    )
}

impl MeshRenderPass {
    pub fn new(context: &mut graal::Context) -> MeshRenderPass {
        let vert = create_shader_module(context, MESH_VIS_SHADER_VERT);
        let frag = create_shader_module(context, MESH_VIS_SHADER_FRAG);

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

        let attachments = &[
            create_color_attachment_description(
                vk::Format::R16G16B16A16_SFLOAT,
                vk::AttachmentLoadOp::CLEAR,
            ),
            create_color_attachment_description(
                vk::Format::R16G16_SFLOAT,
                vk::AttachmentLoadOp::CLEAR,
            ),
            create_color_attachment_description(
                vk::Format::R16G16_SFLOAT,
                vk::AttachmentLoadOp::CLEAR,
            ),
            // depth
            vk::AttachmentDescription {
                flags: Default::default(),
                format: vk::Format::D32_SFLOAT,
                samples: vk::SampleCountFlags::TYPE_1,
                load_op: vk::AttachmentLoadOp::CLEAR,
                store_op: vk::AttachmentStoreOp::STORE,
                stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                initial_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
        ];

        let color_attachments = &[
            vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            vk::AttachmentReference {
                attachment: 2,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
        ];

        let subpasses = &[vk::SubpassDescription {
            flags: Default::default(),
            pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
            input_attachment_count: 0,
            p_input_attachments: ptr::null(),
            color_attachment_count: color_attachments.len() as u32,
            p_color_attachments: color_attachments.as_ptr(),
            p_resolve_attachments: ptr::null(),
            p_depth_stencil_attachment: ptr::null(),
            preserve_attachment_count: 0,
            p_preserve_attachments: ptr::null(),
        }];

        // render pass
        let render_pass_create_info = vk::RenderPassCreateInfo {
            flags: Default::default(),
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            subpass_count: subpasses.len() as u32,
            p_subpasses: subpasses.as_ptr(),
            dependency_count: 0,
            p_dependencies: ptr::null(),
            ..Default::default()
        };

        let render_pass = unsafe {
            device
                .create_render_pass(&render_pass_create_info, None)
                .unwrap()
        };

        MeshRenderPass {
            pipeline: Default::default(),
            pipeline_layout: Default::default(),
            render_pass: Default::default(),
        }
    }

    fn allocate_buffers(&self, batch: &graal::Batch, size: (u32, u32)) -> GBuffers {
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
    }

    pub fn run(
        &self,
        batch: &graal::Batch,
        vertex_buffer: graal::TypedBufferInfo<Vertex3D>,
        vertex_count: usize,
        target_size: (u32, u32),
    ) -> GBuffers {
        // allocate the G-buffers of the frame
        let g = self.allocate_buffers(batch, target_size);

        // setup uniforms & descriptors
        let globals = batch.upload(vk::BufferUsageFlags::UNIFORM_BUFFER, &Globals {});

        // setup the pass
        batch.add_graphics_pass("gbuffers", |pass| {
            pass.register_buffer_access(
                vertex_buffer.id,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::PipelineStageFlags::VERTEX_INPUT,
            );

            pass.add_image_usage(
                g.color.id,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            pass.add_image_usage(
                g.normal.id,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            pass.add_image_usage(
                g.tangent.id,
                vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            pass.add_image_usage(
                g.depth.id,
                vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            );

            let pipeline = self.pipeline;
            let pipeline_layout = self.pipeline_layout;

            pass.set_commands(move |context, cb| unsafe {
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
                    clear_value_count: 0,
                    p_clear_values: ptr::null(),
                    ..Default::default()
                };
                context.device().cmd_begin_render_pass(
                    cb,
                    &render_pass_begin_info,
                    vk::SubpassContents::INLINE,
                );
                context
                    .device()
                    .cmd_bind_vertex_buffers(cb, 0, &[vertex_buffer.handle], &[0]);
                context.device().cmd_bind_descriptor_sets(
                    cb,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline_layout,
                    0,
                    &[descriptor_set],
                    &[],
                );
                context.device().cmd_set_viewport(
                    cb,
                    0,
                    &[vk::Viewport {
                        x: 0.0,
                        y: 0.0,
                        width: 1024.0,
                        height: 768.0,
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
                            width: 1024,
                            height: 768,
                        },
                    }],
                );
                context
                    .device()
                    .cmd_bind_pipeline(cb, vk::PipelineBindPoint::GRAPHICS, pipeline);
                context.device().cmd_draw(cb, 6, 1, 0, 0);
                context.device().cmd_end_render_pass(cb);
            });
        });

        unimplemented!()
    }
}

struct BackgroundPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

impl BackgroundPass {
    fn new(context: &mut graal::Context) -> BackgroundPass {
        let vert = create_shader_module(context, BACKGROUND_SHADER_VERT);
        let frag = create_shader_module(context, BACKGROUND_SHADER_FRAG);

        let shader_stages = [
            vk::PipelineShaderStageCreateInfo {
                flags: Default::default(),
                stage: vk::ShaderStageFlags::VERTEX,
                module: vert,
                p_name: b"main\0".as_ptr() as *const i8,
                p_specialization_info: ptr::null(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                flags: Default::default(),
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: frag,
                p_name: b"main\0".as_ptr() as *const i8,
                p_specialization_info: ptr::null(),
                ..Default::default()
            },
        ];

        let mut set_layouts = Vec::new();
        let layout_handle = context
            .get_or_create_descriptor_set_layout_for_interface::<BackgroundShaderInterface>();
        set_layouts.push(layout_handle);

        let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
            flags: Default::default(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            context
                .device()
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .unwrap()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            flags: Default::default(),
            topology: vk::PrimitiveTopology::TRIANGLE_LIST,
            primitive_restart_enable: 0,
            ..Default::default()
        };

        let tessellation_state = vk::PipelineTessellationStateCreateInfo {
            flags: Default::default(),
            patch_control_points: 0,
            ..Default::default()
        };

        let viewport_state = vk::PipelineViewportStateCreateInfo {
            flags: Default::default(),
            viewport_count: 1,
            p_viewports: ptr::null(),
            scissor_count: 1,
            p_scissors: ptr::null(),
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            flags: Default::default(),
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
            flags: Default::default(),
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: 0,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: vk::FALSE,
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            flags: Default::default(),
            depth_test_enable: vk::FALSE,
            depth_write_enable: vk::FALSE,
            depth_compare_op: Default::default(),
            depth_bounds_test_enable: 0,
            stencil_test_enable: 0,
            front: Default::default(),
            back: Default::default(),
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
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

        let render_pass =
            create_single_color_target_render_pass(context.device(), vk::Format::B8G8R8A8_SRGB);

        let gpci = vk::GraphicsPipelineCreateInfo {
            flags: Default::default(),
            stage_count: shader_stages.len() as u32,
            p_stages: shader_stages.as_ptr(),
            p_vertex_input_state: &Vertex2DInput::get_pipeline_vertex_input_state_create_info(),
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

        BackgroundPass {
            pipeline,
            pipeline_layout,
            render_pass,
        }
    }

    pub fn run(
        &self,
        batch: &graal::Batch,
        target: graal::ImageInfo,
        target_format: vk::Format,
        target_size: (u32, u32),
    ) {
        let (left, top, right, bottom) = (-1.0, -1.0, 1.0, 1.0);

        let vertices = &[
            Vertex2D::new([left, top], [0.0, 0.0]),
            Vertex2D::new([right, top], [1.0, 0.0]),
            Vertex2D::new([left, bottom], [0.0, 1.0]),
            Vertex2D::new([left, bottom], [0.0, 1.0]),
            Vertex2D::new([right, top], [1.0, 0.0]),
            Vertex2D::new([right, bottom], [1.0, 1.0]),
        ];

        let vbo = batch.upload_slice(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
            Some("background vertices"),
        );
        let ubo = batch.upload(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &BackgroundUniforms {
                u_resolution: [target_size.0 as f32, target_size.1 as f32],
                u_scroll_offset: [0.0, 0.0],
                u_zoom: 1.0,
            },
            Some("background uniforms"),
        );

        let descriptor_set = unsafe {
            batch.create_descriptor_set(&BackgroundShaderInterface {
                uniforms: ubo.into(),
            })
        };

        let output_view = unsafe {
            batch.create_image_view(&vk::ImageViewCreateInfo {
                flags: vk::ImageViewCreateFlags::empty(),
                image: target.handle,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::B8G8R8A8_SRGB,
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            })
        };

        let framebuffer = unsafe {
            batch.create_framebuffer(
                target_size.0,
                target_size.1,
                1,
                self.render_pass,
                &[output_view],
            )
        };

        // what's left:
        // - safe transient image objects
        let render_pass = self.render_pass;
        let pipeline_layout = self.pipeline_layout;
        let pipeline = self.pipeline;

        batch.add_graphics_pass("background render", |pass| {
            // access uniforms buffer as SHADER_READ, vertex and fragment stages

            //dbg!(ubo);
            pass.register_buffer_access(
                ubo.id,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::empty(),
            );

            // access quad VBO as vertex input
            pass.register_buffer_access(
                vbo.id,
                vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                vk::PipelineStageFlags::VERTEX_INPUT,
                vk::PipelineStageFlags::empty(),
            );

            // write to the target image as a color attachment
            pass.add_image_usage(
                target.id,
                vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            );

            pass.set_commands(move |context, cb| {
                unsafe {
                    let render_pass_begin_info = vk::RenderPassBeginInfo {
                        render_pass,
                        framebuffer,
                        render_area: vk::Rect2D {
                            offset: vk::Offset2D { x: 0, y: 0 },
                            extent: vk::Extent2D {
                                width: target_size.0, // FIXME: size
                                height: target_size.1,
                            },
                        },
                        clear_value_count: 0,
                        p_clear_values: ptr::null(),
                        ..Default::default()
                    };
                    context.device().cmd_begin_render_pass(
                        cb,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    context
                        .device()
                        .cmd_bind_vertex_buffers(cb, 0, &[vbo.handle], &[0]);
                    context.device().cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    context.device().cmd_set_viewport(
                        cb,
                        0,
                        &[vk::Viewport {
                            x: 0.0,
                            y: 0.0,
                            width: 1024.0,
                            height: 768.0,
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
                                width: 1024,
                                height: 768,
                            },
                        }],
                    );
                    context.device().cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                    context.device().cmd_draw(cb, 6, 1, 0, 0);
                    context.device().cmd_end_render_pass(cb);
                }
            });
        });
    }
}

fn create_transient_image(context: &mut graal::Context, name: &str) -> graal::ImageId {
    let graal::ImageInfo { id, .. } = context.create_image(
        name,
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::ImageResourceCreateInfo {
            image_type: graal::vk::ImageType::TYPE_2D,
            usage: graal::vk::ImageUsageFlags::COLOR_ATTACHMENT
                | graal::vk::ImageUsageFlags::SAMPLED
                | graal::vk::ImageUsageFlags::TRANSFER_DST,
            format: graal::vk::Format::R8G8B8A8_SRGB,
            extent: graal::vk::Extent3D {
                width: 1280,
                height: 720,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: 1,
            tiling: graal::vk::ImageTiling::OPTIMAL,
        },
        true,
    );
    id
}

struct MeshData {
    vertex_buffer: graal::TypedBufferInfo<Vertex3D>,
    vertex_count: usize,
}

fn load_mesh(batch: &graal::Batch, obj_file_path: &Path) -> MeshData {
    let obj = obj::Obj::load(obj_file_path).expect("failed to load obj file");

    let mut vertices = Vec::new();

    let g = obj.data.objects.first().unwrap().groups.first().unwrap();
    for f in g.polys.iter() {
        if f.0.len() != 3 {
            continue;
        }
        for &obj::IndexTuple(pi, ti, ni) in f.0.iter() {
            vertices.push(Vertex3D {
                position: obj.data.position[pi].into(),
                normal: ni.map(|ni| obj.data.normal[ni]).unwrap_or_default().into(),
                tangent: ni.map(|ni| obj.data.normal[ni]).unwrap_or_default().into(), // TODO
            });
        }
    }

    // create the staging vertex buffer
    let staging_vbo = batch.upload_slice(
        vk::BufferUsageFlags::TRANSFER_SRC,
        &vertices,
        Some("staging"),
    );

    // byte size of the vertex buffers
    let byte_size = (vertices.len() * mem::size_of::<Vertex3D>()) as u64;

    // create the device local vertex buffer
    let device_vbo = batch.context().create_buffer(
        obj_file_path.to_str().unwrap(),
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            byte_size,
            map_on_create: false,
        },
        false,
    );

    // upload
    batch.add_transfer_pass("upload mesh", false, |pass| {
        pass.register_buffer_access(
            staging_vbo.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.register_buffer_access(
            device_vbo.id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.set_commands(move |context, command_buffer| unsafe {
            context.device().cmd_copy_buffer(
                command_buffer,
                staging_vbo.handle,
                device_vbo.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: byte_size,
                }],
            );
        });
    });

    MeshData {
        vertex_buffer: TypedBufferInfo {
            id: vertex_buffer_id,
            handle: vertex_buffer_handle,
            mapped_ptr: ptr::null_mut(),
        },
        vertex_count: vertices.len(),
    }
}

fn test_pass(
    batch: &graal::Batch,
    name: &str,
    images: &[(
        graal::ImageId,
        graal::vk::AccessFlags,
        graal::vk::PipelineStageFlags,
        graal::vk::PipelineStageFlags,
        graal::vk::ImageLayout,
    )],
) {
    batch.add_graphics_pass(name, |pass| {
        for &(img, access_mask, input_stage, output_stage, layout) in images {
            pass.add_image_usage(img, access_mask, input_stage, output_stage, layout);
        }
    });
}

fn color_attachment_output(
    img: graal::ImageId,
) -> (
    graal::ImageId,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::ImageLayout,
) {
    (
        img,
        vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    )
}

fn sample_image(
    img: graal::ImageId,
) -> (
    graal::ImageId,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::ImageLayout,
) {
    (
        img,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
        Default::default(),
        vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    )
}

fn compute_read(
    img: graal::ImageId,
) -> (
    graal::ImageId,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::ImageLayout,
) {
    (
        img,
        vk::AccessFlags::SHADER_READ,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        Default::default(),
        vk::ImageLayout::GENERAL,
    )
}

fn compute_write(
    img: graal::ImageId,
) -> (
    graal::ImageId,
    vk::AccessFlags,
    vk::PipelineStageFlags,
    vk::PipelineStageFlags,
    vk::ImageLayout,
) {
    (
        img,
        vk::AccessFlags::SHADER_WRITE,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::PipelineStageFlags::COMPUTE_SHADER,
        vk::ImageLayout::GENERAL,
    )
}

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle());
    let device = graal::Device::new(surface);
    let mut context = graal::Context::new(device);
    let swapchain = unsafe { context.create_swapchain(surface, window.inner_size().into()) };

    // Initial batch to upload static resources (meshes and textures)
    let mut init_batch = context.start_batch();
    let mesh = load_mesh(&init_batch, "data/sphere.obj".as_ref());
    init_batch.finish();

    let bkg_pass = BackgroundPass::new(&mut context);
    let mesh_pass = MeshRenderPass::new(&mut context);

    let mut swapchain_size = window.inner_size().into();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { window_id, event } => match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    *control_flow = ControlFlow::Exit
                }
                WindowEvent::Resized(size) => unsafe {
                    swapchain_size = size.into();
                    context.resize_swapchain(swapchain, swapchain_size);
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let swapchain_image = unsafe { context.acquire_next_image(swapchain) };
                let mut batch = context.start_batch();

                // draw background
                bkg_pass.run(
                    &batch,
                    swapchain_image.image_info,
                    vk::Format::B8G8R8A8_SRGB,
                    swapchain_size,
                );

                // draw our mesh to G-buffers
                let gbuffers = mesh_pass.run(&batch, mesh.vertex_buffer, mesh.vertex_count);

                batch.finish();
            }
            _ => (),
        }
    });
}
