use crate::shader::create_shader_module;
use graal::vk;
use inline_spirv::include_spirv;
use std::ptr;

static BACKGROUND_SHADER_VERT: &[u32] = include_spirv!("shaders/background.vert", vert);
static BACKGROUND_SHADER_FRAG: &[u32] = include_spirv!("shaders/background.frag", frag);

// --- Vertex types --------------------------------------------------------------------------------
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

// --- Uniform structs -----------------------------------------------------------------------------

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
struct BackgroundShaderInterface {
    #[layout(binding = 0, uniform_buffer, stages(fragment))]
    uniforms: graal::BufferDescriptor<BackgroundUniforms>,
}

// --- Pass ----------------------------------------------------------------------------------------
pub struct BackgroundPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    render_pass: vk::RenderPass,
}

impl BackgroundPass {
    pub fn new(context: &mut graal::Context) -> BackgroundPass {
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
                .vulkan_device()
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

        let render_pass = create_single_color_target_render_pass(
            context.vulkan_device(),
            vk::Format::B8G8R8A8_SRGB,
        );

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
                .vulkan_device()
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
        frame: &graal::Frame,
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

        let vbo = frame.upload_slice(
            vk::BufferUsageFlags::VERTEX_BUFFER,
            vertices,
            Some("background vertices"),
        );
        let ubo = frame.upload(
            vk::BufferUsageFlags::UNIFORM_BUFFER,
            &BackgroundUniforms {
                u_resolution: [target_size.0 as f32, target_size.1 as f32],
                u_scroll_offset: [0.0, 0.0],
                u_zoom: 1.0,
            },
            Some("background uniforms"),
        );

        let render_pass = self.render_pass;
        let pipeline_layout = self.pipeline_layout;
        let pipeline = self.pipeline;

        frame.add_graphics_pass("background render", |pass| {
            // access uniforms buffer as SHADER_READ, vertex and fragment stages
            pass.register_buffer_access(ubo.id, graal::AccessType::AnyShaderReadUniformBuffer);
            // access quad VBO as vertex input
            pass.register_buffer_access(vbo.id, graal::AccessType::VertexAttributeRead);
            // write to the target image as a color attachment
            pass.register_image_access(target.id, graal::AccessType::ColorAttachmentWrite);

            pass.set_commands(move |context, cb| {
                unsafe {
                    let descriptor_set =
                        context.create_descriptor_set(&BackgroundShaderInterface {
                            uniforms: ubo.into(),
                        });

                    let output_view = context.create_image_view(&vk::ImageViewCreateInfo {
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
                    });

                    let framebuffer = context.create_framebuffer(
                        target_size.0,
                        target_size.1,
                        1,
                        render_pass,
                        &[output_view],
                    );

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
                    context.vulkan_device().cmd_begin_render_pass(
                        cb,
                        &render_pass_begin_info,
                        vk::SubpassContents::INLINE,
                    );
                    context
                        .vulkan_device()
                        .cmd_bind_vertex_buffers(cb, 0, &[vbo.handle], &[0]);
                    context.vulkan_device().cmd_bind_descriptor_sets(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline_layout,
                        0,
                        &[descriptor_set],
                        &[],
                    );
                    context.vulkan_device().cmd_set_viewport(
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
                    context.vulkan_device().cmd_set_scissor(
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
                    context.vulkan_device().cmd_bind_pipeline(
                        cb,
                        vk::PipelineBindPoint::GRAPHICS,
                        pipeline,
                    );
                    context.vulkan_device().cmd_draw(cb, 6, 1, 0, 0);
                    context.vulkan_device().cmd_end_render_pass(cb);
                }
            });
        });
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
