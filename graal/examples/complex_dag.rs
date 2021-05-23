use ash::{
    version::DeviceV1_0,
    vk::{BufferUsageFlags, Rect2D, SampleCountFlags},
};
use graal::{
    ash::version::DeviceV1_1, extract_descriptor_set_layouts_from_shader_stages, vk,
    BufferDescriptor, BufferResourceCreateInfo, DescriptorSetInterface, FrameCreateInfo, ImageId,
    ImageInfo, ImageResourceCreateInfo, Norm, PipelineShaderStage, ResourceId, ResourceMemoryInfo,
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
                .vulkan_device()
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
                .vulkan_device()
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
                .vulkan_device()
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
            create_single_color_target_render_pass(context.vulkan_device(), vk::Format::B8G8R8A8_SRGB);

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
                .vulkan_device()
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
}

fn load_image(
    frame: &graal::Frame,
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
    } = frame.context().create_image(
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
    );

    let byte_size = width as u64 * height as u64 * bpp as u64;

    // create a staging buffer
    let mut staging_buffer = frame.alloc_upload_slice::<u8>(
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
                staging_buffer.mapped_ptr as *mut u8,
                bpp,
            )
            .expect("failed to read image");
    }

    let staging_buffer_handle = staging_buffer.handle;

    // build the upload pass
    frame.add_graphics_pass("image upload", |pass| {
        pass.register_image_access_2(
            image_id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        pass.register_buffer_access_2(
            staging_buffer.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        );

        pass.set_commands(move |context, command_buffer| unsafe {
            let device = context.vulkan_device();

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

fn create_transient_image(context: &mut graal::Context, name: &str, is_depth: bool) -> ImageId {
    let ImageInfo { id, .. } = context.create_image(
        name,
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::ImageResourceCreateInfo {
            image_type: graal::vk::ImageType::TYPE_2D,
            usage: if is_depth {
                graal::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT
            } else {
                graal::vk::ImageUsageFlags::COLOR_ATTACHMENT
            } | graal::vk::ImageUsageFlags::SAMPLED
                | graal::vk::ImageUsageFlags::TRANSFER_DST,
            format: if is_depth {
                graal::vk::Format::D32_SFLOAT
            } else {
                graal::vk::Format::R8G8B8A8_SRGB
            },
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
    );
    id
}

struct MeshData {
    vertex_buffer: graal::BufferId,
    vertex_count: usize,
}

fn load_mesh(batch: &graal::Frame, obj_file_path: &Path) -> MeshData {
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
        pass.register_buffer_access_2(
            staging_buffer.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.register_buffer_access_2(
            vertex_buffer_id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
        );
        pass.set_commands(move |context, command_buffer| unsafe {
            context.vulkan_device().cmd_copy_buffer(
                command_buffer,
                staging_buffer_handle,
                vertex_buffer_handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: byte_size,
                }],
            );
        });
    });

    MeshData {
        vertex_buffer: vertex_buffer_id,
        vertex_count: vertices.len(),
    }
}

fn test_pass(
    batch: &graal::Frame,
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
            pass.register_image_access_2(img, access_mask, input_stage, layout, layout);
        }
    });
}

fn depth_attachment(
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
        vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    )
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
        vk::AccessFlags::COLOR_ATTACHMENT_READ | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
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
    let device = graal::Device::new(Some(surface));
    let mut context = graal::Context::with_device(device);
    let swapchain = unsafe { context.create_swapchain(surface, window.inner_size().into()) };
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
                    context.resize_swapchain(swapchain.id, swapchain_size);
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let img_a = create_transient_image(&mut context, "A", false);
                let img_b = create_transient_image(&mut context, "B", false);
                let img_c = create_transient_image(&mut context, "C", false);
                let img_d1 = create_transient_image(&mut context, "D1", false);
                let img_d2 = create_transient_image(&mut context, "D2", false);
                let img_e = create_transient_image(&mut context, "E", false);
                let img_f = create_transient_image(&mut context, "F", false);
                let img_g = create_transient_image(&mut context, "G", false);
                let img_h = create_transient_image(&mut context, "H", false);
                let img_i = create_transient_image(&mut context, "I", false);
                let img_j = create_transient_image(&mut context, "J", false);
                let img_k = create_transient_image(&mut context, "K", false);
                let img_depth = create_transient_image(&mut context, "depth", true);

                let mut frame = context.start_frame(FrameCreateInfo {
                    collect_debug_info: false,
                    .. Default::default()
                });

                test_pass(&frame, "P0", &[color_attachment_output(img_a)]);
                test_pass(&frame, "P1", &[color_attachment_output(img_b)]);
                test_pass(
                    &frame,
                    "P2",
                    &[
                        compute_read(img_a),
                        compute_read(img_b),
                        compute_write(img_d1),
                        compute_write(img_d2),
                    ],
                );
                test_pass(
                    &frame,
                    "P3",
                    &[color_attachment_output(img_c), depth_attachment(img_depth)],
                );
                test_pass(
                    &frame,
                    "P4",
                    &[
                        compute_read(img_d2),
                        compute_read(img_c),
                        compute_write(img_e),
                    ],
                );
                test_pass(&frame, "P5", &[compute_read(img_d1), compute_write(img_f)]);
                test_pass(
                    &frame,
                    "P6",
                    &[
                        compute_read(img_e),
                        compute_read(img_f),
                        compute_write(img_g),
                    ],
                );
                test_pass(&frame, "P7", &[compute_read(img_g), compute_write(img_h)]);
                test_pass(&frame, "P8", &[compute_read(img_h), compute_write(img_i)]);
                test_pass(
                    &frame,
                    "P9",
                    &[
                        compute_read(img_i),
                        compute_read(img_g),
                        compute_write(img_j),
                    ],
                );
                test_pass(&frame, "P10", &[compute_read(img_j), compute_write(img_k)]);

                test_pass(
                    &frame,
                    "P11",
                    &[color_attachment_output(swapchain_image.image_info.id)],
                );

                frame.present("P12", &swapchain_image);
                frame.finish();
            }
            _ => (),
        }
    });
}
