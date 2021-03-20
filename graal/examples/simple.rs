use ash::{
    version::DeviceV1_0,
    vk::{BufferUsageFlags, Rect2D, SampleCountFlags},
};
use graal::{
    ash::version::DeviceV1_1, extract_descriptor_set_layouts_from_shader_stages, vk,
    BufferDescriptor, BufferResourceCreateInfo, DescriptorSetInterface, ImageId, ImageInfo,
    ImageResourceCreateInfo, Norm, PipelineShaderStage, ResourceId, ResourceMemoryInfo,
    VertexBufferView, VertexData, VertexInputInterface,
};
use inline_spirv::include_spirv;
use raw_window_handle::HasRawWindowHandle;
use std::{mem, path::Path, ptr};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

static BACKGROUND_SHADER_VERT: &[u32] = include_spirv!("shaders/background.vert", vert);
static BACKGROUND_SHADER_FRAG: &[u32] = include_spirv!("shaders/background.frag", frag);

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct BackgroundUniforms {
    u_resolution: [f32; 2],
    u_scroll_offset: [f32; 2],
    u_zoom: f32,
}

#[derive(Copy, Clone, Debug)]
#[repr(C)]
struct BlurUniforms {}

#[derive(graal::DescriptorSetInterface)]
#[repr(C)]
struct BackgroundShaderInterface<'a> {
    #[layout(binding = 0, uniform_buffer, stages(fragment))]
    uniforms: BufferDescriptor<'a, BackgroundUniforms>,
}

#[derive(Copy, Clone, Debug, VertexData)]
#[repr(C)]
struct Vertex {
    position: [f32; 2],
    texcoords: [Norm<u16>; 2],
}

impl Vertex {
    pub fn new(position: [f32; 2], texcoords: [f32; 2]) -> Vertex {
        Vertex {
            position,
            texcoords: [texcoords[0].into(), texcoords[1].into()],
        }
    }
}

#[derive(Copy, Clone, Debug, VertexInputInterface)]
struct VertexInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: VertexBufferView<Vertex>,
}

/// Creates a render pass with a single subpass, writing to a single color target with the specified
/// format.
fn create_single_color_target_render_pass(
    device: &ash::Device,
    target_format: vk::Format,
) -> vk::RenderPass {
    let render_pass_attachments = &[vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::MAY_ALIAS,
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

struct BlurPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    layout_handle: vk::DescriptorSetLayout,
    update_template: vk::DescriptorUpdateTemplate,
    render_pass: vk::RenderPass,
}

struct BackgroundPass {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    layout_handle: vk::DescriptorSetLayout,
    render_pass: vk::RenderPass,
}

impl BackgroundPass {
    fn new(context: &mut graal::Context) -> BackgroundPass {
        let vert = unsafe {
            context
                .device()
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        flags: Default::default(),
                        code_size: BACKGROUND_SHADER_VERT.len() * 4,
                        p_code: BACKGROUND_SHADER_VERT.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create shader module")
        };

        let frag = unsafe {
            context
                .device()
                .create_shader_module(
                    &vk::ShaderModuleCreateInfo {
                        flags: Default::default(),
                        code_size: BACKGROUND_SHADER_FRAG.len() * 4,
                        p_code: BACKGROUND_SHADER_FRAG.as_ptr(),
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create shader module")
        };

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

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            flags: Default::default(),
            vertex_binding_description_count: VertexInput::BINDINGS.len() as u32,
            p_vertex_binding_descriptions: VertexInput::BINDINGS.as_ptr(),
            vertex_attribute_description_count: VertexInput::ATTRIBUTES.len() as u32,
            p_vertex_attribute_descriptions: VertexInput::ATTRIBUTES.as_ptr(),
            ..Default::default()
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
            p_vertex_input_state: &vertex_input_state,
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
            layout_handle,
            pipeline_layout,
            render_pass,
        }
    }

    pub fn run(
        &self,
        batch: &graal::Batch,
        target: graal::ImageId,
        target_format: vk::Format,
        target_size: (u32, u32),
    ) {
        // get:
        // - descriptor set layout cache
        // - descriptor set layout allocator
        // - batch
        // - context
        // - device
        // - UBO ID
        // - mapped pointer to UBO
        // - buffer handle to UBO
        // - shader interface struct
        // - descriptor set
        // - update template
        // - get handle of output image
        // - create output image view
        // - create framebuffer
        //
        // Issues:
        // - indirection to allocate the descriptor set
        // - create_buffer_resource for the UBO is verbose
        // - unsafe cast of mapped pointer to UBO to update uniforms
        // - indirection: `batch.context().device()` everywhere
        // - indirection: `batch.context().buffer_handle/image_handle`
        // - framebuffers should be recreated
        //
        // Mistakes:
        // - forgot to add usage to a buffer
        // - transient buffers not used in the pass are immediately deleted?
        //      - it's not really wrong
        //
        // Simplify:
        // - typed resource IDs
        // - store handle along with IDs (128-bit resource pointer)
        // - allocation of the UBO and uniforms
        //       - batch convenience methods
        //       - typed UBOs
        // - batch APIs for:
        //       - transient image views
        //       - transient framebuffers
        //
        // Pass finish on drop
        //
        // Same sizes:
        // - framebuffer
        // - viewport
        // - render area
        // Same formats:
        // - framebuffer
        // - image view

        // who is responsible for the cleanup?
        // - we could call cleanup here, but since the allocator is possibly shared between
        // different users, we might have redundant calls.
        // - we could manually call cleanup in the main loop
        // - we could put descriptor allocators in the context and have the context automatically
        //   clean up things on batch submission
        // Creating a descriptor would simply be done like so:
        //    batch.allocate_descriptor_set::<BackgroundShaderInterface>();
        // or even:
        //    batch.allocate_descriptor_set(&BackgroundShaderInterface { ... });
        // No need to store the descriptor set layout id => stuff it in a static ONCE_CELL

        let target = batch.image_ref(target);

        let (left, top, right, bottom) = (-1.0, -1.0, 1.0, 1.0);

        let vertices = &[
            Vertex::new([left, top], [0.0, 0.0]),
            Vertex::new([right, top], [1.0, 0.0]),
            Vertex::new([left, bottom], [0.0, 1.0]),
            Vertex::new([left, bottom], [0.0, 1.0]),
            Vertex::new([right, top], [1.0, 0.0]),
            Vertex::new([right, bottom], [1.0, 1.0]),
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

        let descriptor_set = batch.create_descriptor_set(&BackgroundShaderInterface {
            uniforms: ubo.into(),
        });

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

        let framebuffer = batch.create_framebuffer(
            target_size.0,
            target_size.1,
            1,
            self.render_pass,
            &[output_view],
        );

        // what's left:
        // - safe transient image objects
        let render_pass = self.render_pass;
        let pipeline_layout = self.pipeline_layout;
        let pipeline = self.pipeline;
        let framebuffer = framebuffer.handle();

        // &'b1 mut Batch<'a>
        // add_graphics_pass: &'b2 Batch, for<'x> FnOnce(&'x mut PassBuilder<'a,'y>) +
        // closure borrows batch: handler: 'b3, 'b1: 'b3
        //
        //

        let vbo_handle = vbo.handle;

        batch.add_graphics_pass("background render", |pass| {
            // access uniforms buffer as SHADER_READ, vertex and fragment stages

            //dbg!(ubo);
            pass.register_buffer_access(ubo.id,
                vk::AccessFlags::SHADER_READ,
                vk::PipelineStageFlags::VERTEX_SHADER | vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::empty(),
            );

            // access quad VBO as vertex input
            pass.register_buffer_access(vbo.id,
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
                        .cmd_bind_vertex_buffers(cb, 0, &[vbo_handle], &[0]);
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

fn load_image(
    batch: &graal::Batch,
    path: &Path,
    usage: graal::vk::ImageUsageFlags,
    mipmaps: bool,
) -> (graal::ImageId, u32, u32) {
    use openimageio::{ImageInput, TypeDesc};

    let image_input = ImageInput::open(path).expect("could not open image file");
    let spec = image_input.spec();

    let nchannels = spec.num_channels();
    let format_typedesc = spec.format();
    let width = spec.width();
    let height = spec.height();

    if nchannels > 4 {
        panic!("unsupported number of channels: {}", nchannels);
    }

    let (vk_format, bpp) = match (format_typedesc, nchannels) {
        (TypeDesc::U8, 1) => (vk::Format::R8_UNORM, 1usize),
        (TypeDesc::U8, 2) => (vk::Format::R8G8_UNORM, 2usize),
        (TypeDesc::U8, 3) => (vk::Format::R8G8B8A8_UNORM, 4usize), // RGB8 not very well supported
        (TypeDesc::U8, 4) => (vk::Format::R8G8B8A8_UNORM, 4usize),
        (TypeDesc::U16, 1) => (vk::Format::R16_UNORM, 2usize),
        (TypeDesc::U16, 2) => (vk::Format::R16G16_UNORM, 4usize),
        (TypeDesc::U16, 3) => (vk::Format::R16G16B16A16_UNORM, 8usize),
        (TypeDesc::U16, 4) => (vk::Format::R16G16B16A16_UNORM, 8usize),
        (TypeDesc::U32, 1) => (vk::Format::R32_UINT, 4usize),
        (TypeDesc::U32, 2) => (vk::Format::R32G32_UINT, 8usize),
        (TypeDesc::U32, 3) => (vk::Format::R32G32B32A32_UINT, 16usize),
        (TypeDesc::U32, 4) => (vk::Format::R32G32B32A32_UINT, 16usize),
        (TypeDesc::HALF, 1) => (vk::Format::R16_SFLOAT, 2usize),
        (TypeDesc::HALF, 2) => (vk::Format::R16G16_SFLOAT, 4usize),
        (TypeDesc::HALF, 3) => (vk::Format::R16G16B16A16_SFLOAT, 8usize),
        (TypeDesc::HALF, 4) => (vk::Format::R16G16B16A16_SFLOAT, 8usize),
        (TypeDesc::FLOAT, 1) => (vk::Format::R32_SFLOAT, 4usize),
        (TypeDesc::FLOAT, 2) => (vk::Format::R32G32_SFLOAT, 8usize),
        (TypeDesc::FLOAT, 3) => (vk::Format::R32G32B32A32_SFLOAT, 16usize),
        (TypeDesc::FLOAT, 4) => (vk::Format::R32G32B32A32_SFLOAT, 16usize),
        _ => panic!("unsupported image format"),
    };

    let mip_levels = graal::get_mip_level_count(width, height);

    // create the texture
    let ImageInfo {
        handle: image_handle,
        id: image_id,
    } = batch.context().create_image(
        path.to_str().unwrap(),
        &ResourceMemoryInfo::DEVICE_LOCAL,
        &ImageResourceCreateInfo {
            image_type: vk::ImageType::TYPE_2D,
            usage: usage | vk::ImageUsageFlags::TRANSFER_DST,
            format: vk_format,
            extent: vk::Extent3D {
                width,
                height,
                depth: 1,
            },
            mip_levels,
            array_layers: 1,
            samples: 1,
            tiling: Default::default(),
        },
        false,
    );

    let byte_size = width as u64 * height as u64 * bpp as u64;

    // create a staging buffer
    let mut staging_buffer = batch.alloc_upload_slice::<u8>(
        vk::BufferUsageFlags::TRANSFER_SRC,
        byte_size as usize,
        Some("staging"),
    );

    // read image data
    unsafe {
        image_input
            .read_unchecked(
                0,
                0,
                0..nchannels,
                format_typedesc,
                staging_buffer.as_mut_ptr(),
                bpp,
            )
            .expect("failed to read image");
    }

    let staging_buffer_handle = staging_buffer.handle;

    // build the upload pass
    batch.add_graphics_pass("image upload", |pass| {
        pass.add_image_usage(
            image_id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        pass.register_buffer_access(staging_buffer.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );

        pass.set_commands(move |context, command_buffer| unsafe {
            let device = context.device();

            let regions = &[vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: width,
                buffer_image_height: height,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
            }];

            device.cmd_copy_buffer_to_image(
                command_buffer,
                staging_buffer_handle,
                image_handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                regions,
            );
        });
    });

    (image_id, width, height)
}

fn create_transient_image(context: &mut graal::Context, name: &str) -> ImageId {
    let ImageInfo { id, .. } = context.create_image(
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
    vertex_buffer: graal::BufferId,
    vertex_count: usize,
}

fn load_mesh(batch: &graal::Batch, obj_file_path: &Path) -> MeshData {
    let obj = obj::Obj::load(obj_file_path).expect("failed to load obj file");

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct Vertex {
        pos: [f32; 3],
        norm: [f32; 3],
        tex: [f32; 2],
    }

    let mut vertices = Vec::new();

    let g = obj.data.objects.first().unwrap().groups.first().unwrap();
    for f in g.polys.iter() {
        if f.0.len() != 3 {
            continue;
        }
        for &obj::IndexTuple(pi, ti, ni) in f.0.iter() {
            vertices.push(Vertex {
                pos: obj.data.position[pi],
                norm: ni.map(|ni| obj.data.normal[ni]).unwrap_or_default(),
                tex: ti.map(|ti| obj.data.texture[ti]).unwrap_or_default(),
            });
        }
    }

    let byte_size = (vertices.len() * mem::size_of::<Vertex>()) as u64;

    let graal::BufferInfo {
        id: vertex_buffer_id,
        handle: vertex_buffer_handle,
        ..
    } = batch.context().create_buffer(
        obj_file_path.to_str().unwrap(),
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            byte_size,
            map_on_create: false,
        },
        false,
    );

    // staging
    let staging_buffer = batch.upload_slice(
        vk::BufferUsageFlags::TRANSFER_SRC,
        &vertices,
        Some("staging"),
    );

    let staging_buffer_handle = staging_buffer.handle;

    // upload
    let mut upload_pass = batch.add_transfer_pass("upload mesh", false, |pass| {
        pass.register_buffer_access(
            staging_buffer.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.register_buffer_access(
            vertex_buffer_id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.set_commands(move |context, command_buffer| {
            unsafe {
                context.device().cmd_copy_buffer(
                    command_buffer,
                    staging_buffer_handle,
                    vertex_buffer_handle,
                    &[vk::BufferCopy {
                        src_offset: 0,
                        dst_offset: 0,
                        size: byte_size,
                    }],
                );
            }
        });
    });

    MeshData {
        vertex_buffer: vertex_buffer_id,
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

// safe persistent resources?
// - current way: ResourceId, can be invalidated
// - easy way: Rc<Context>, image has backref to context
//
// Keep IDs, explicit "use" in batches to get a BufferRef or an ImageRef?

fn main() {
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle());
    let device = graal::Device::new(surface);
    let mut context = graal::Context::new(device);
    let swapchain = unsafe { context.create_swapchain(surface, window.inner_size().into()) };

    let mut init_batch = context.start_batch();
    let (file_image_id, file_image_width, file_image_height) = load_image(
        &init_batch,
        "../data/El4KUGDU0AAW64U.jpg".as_ref(),
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
        false,
    );
    let mesh = load_mesh(&init_batch, "../data/sphere.obj".as_ref());
    init_batch.finish();

    let bkgpp = BackgroundPass::new(&mut context);

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
                let img_a = create_transient_image(&mut context, "A");
                let img_b = create_transient_image(&mut context, "B");
                let img_c = create_transient_image(&mut context, "C");
                let img_d1 = create_transient_image(&mut context, "D1");
                let img_d2 = create_transient_image(&mut context, "D2");
                let img_e = create_transient_image(&mut context, "E");
                let img_f = create_transient_image(&mut context, "F");
                let img_g = create_transient_image(&mut context, "G");
                let img_h = create_transient_image(&mut context, "H");
                let img_i = create_transient_image(&mut context, "I");
                let img_j = create_transient_image(&mut context, "J");
                let img_k = create_transient_image(&mut context, "K");

                // each resource has a ref count
                // - incremented when it's in use by a batch
                // - incremented by the user
                // non-transient resources are deleted once refcount is zero
                // transient resources are deleted once the batch is finished, regardless of refcounts

                let swapchain_image = unsafe { context.acquire_next_image(swapchain) };
                let mut batch = context.start_batch();

                bkgpp.run(
                    &batch,
                    swapchain_image.image_id,
                    vk::Format::B8G8R8A8_SRGB,
                    swapchain_size,
                );

                test_pass(&batch, "P0", &[color_attachment_output(img_a)]);
                test_pass(&batch, "P1", &[color_attachment_output(img_b)]);
                test_pass(
                    &batch,
                    "P2",
                    &[
                        compute_read(img_a),
                        compute_read(img_b),
                        compute_write(img_d1),
                        compute_write(img_d2),
                    ],
                );
                test_pass(&batch, "P3", &[color_attachment_output(img_c)]);
                test_pass(
                    &batch,
                    "P4",
                    &[
                        compute_read(img_d2),
                        compute_read(img_c),
                        compute_write(img_e),
                    ],
                );
                test_pass(
                    &batch,
                    "P5",
                    &[compute_read(img_d1), compute_write(img_f)],
                );
                test_pass(
                    &batch,
                    "P6",
                    &[
                        compute_read(img_e),
                        compute_read(img_f),
                        compute_write(img_g),
                    ],
                );
                test_pass(
                    &batch,
                    "P7",
                    &[compute_read(img_g), compute_write(img_h)],
                );
                test_pass(
                    &batch,
                    "P8",
                    &[compute_read(img_h), compute_write(img_i)],
                );
                test_pass(
                    &batch,
                    "P9",
                    &[
                        compute_read(img_i),
                        compute_read(img_g),
                        compute_write(img_j),
                    ],
                );
                test_pass(
                    &batch,
                    "P10",
                    &[compute_read(img_j), compute_write(img_k)],
                );

                test_pass(
                    &batch,
                    "P11",
                    &[color_attachment_output(swapchain_image.image_id)],
                );

                // blit pass
                /*let mut blit_pass = batch.build_graphics_pass("blit to screen");
                blit_pass.add_image_usage(
                    file_image_id,
                    vk::AccessFlags::TRANSFER_READ,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::empty(),
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                );
                blit_pass.add_image_usage(
                    swapchain_image.image_id,
                    vk::AccessFlags::TRANSFER_WRITE,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::PipelineStageFlags::TRANSFER,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                );

                let blit_w = file_image_width.min(swapchain_size.0);
                let blit_h = file_image_height.min(swapchain_size.1);
                blit_pass.set_commands(|context, command_buffer| {
                    let dst_image_handle = context.image_handle(swapchain_image.image_id);
                    let src_image_handle = context.image_handle(file_image_id);

                    let regions = &[vk::ImageBlit {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: blit_w as i32,
                                y: blit_h as i32,
                                z: 1,
                            },
                        ],
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            mip_level: 0,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offsets: [
                            vk::Offset3D { x: 0, y: 0, z: 0 },
                            vk::Offset3D {
                                x: blit_w as i32,
                                y: blit_h as i32,
                                z: 1,
                            },
                        ],
                    }];

                    unsafe {
                        context.device().cmd_blit_image(
                            command_buffer,
                            src_image_handle,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            dst_image_handle,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            regions,
                            vk::Filter::NEAREST,
                        );
                    }
                });
                blit_pass.finish();*/

                batch.present("P12", &swapchain_image);

                batch.finish();
            }
            _ => (),
        }
    });
}
