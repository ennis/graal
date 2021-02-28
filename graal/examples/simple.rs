use ash::version::DeviceV1_0;
use graal::{vk, BufferResourceCreateInfo, ImageResourceCreateInfo, ResourceMemoryInfo, extract_descriptor_set_layouts_from_shader_stages, Norm, VertexBufferView, VertexInputInterface};
use raw_window_handle::HasRawWindowHandle;
use std::path::Path;
use std::{mem, ptr};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use graal::PipelineShaderStage;
use graal::VertexData;
use inline_spirv::include_spirv;

static BACKGROUND_SHADER_VERT : &[u32] = include_spirv!("shaders/background.vert", vert);
static BACKGROUND_SHADER_FRAG : &[u32] = include_spirv!("shaders/background.frag", frag);

#[derive(Copy,Clone,Debug)]
#[repr(C)]
struct Uniforms {
    u_resolution: [f32;2],
    u_scroll_offset: [f32;2],
    u_zoom: f32
}

#[derive(graal::DescriptorSetInterface)]
struct BackgroundShaderInterface {
    #[layout(binding=0, uniform_buffer, stages(fragment))]
    uniforms: vk::DescriptorBufferInfo,
}

#[derive(Copy,Clone,Debug,VertexData)]
#[repr(C)]
struct Vertex {
    position: [f32;2],
    texcoords: Norm<[u16;2]>,
}

#[derive(Copy,Clone,Debug,VertexInputInterface)]
struct VertexInput {
    #[layout(binding=0,location=0,per_vertex)]
    vertices: VertexBufferView<Vertex>
}


fn create_pipeline(device: &ash::Device, descriptor_set_layout_cache: &mut graal::DescriptorSetLayoutCache)
{
    let vert = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo {
            flags: Default::default(),
            code_size: BACKGROUND_SHADER_VERT.len()*4,
            p_code: BACKGROUND_SHADER_VERT.as_ptr(),
            .. Default::default()
        }, None).expect("failed to create shader module")
    };

    let frag = unsafe {
        device.create_shader_module(&vk::ShaderModuleCreateInfo {
            flags: Default::default(),
            code_size: BACKGROUND_SHADER_FRAG.len()*4,
            p_code: BACKGROUND_SHADER_FRAG.as_ptr(),
            .. Default::default()
        }, None).expect("failed to create shader module")
    };

    let shader_stages = [
        vk::PipelineShaderStageCreateInfo {
            flags: Default::default(),
            stage: vk::ShaderStageFlags::VERTEX,
            module: vert,
            p_name: b"main\0".as_ptr() as *const i8,
            p_specialization_info: ptr::null(),
            .. Default::default()
        },
        vk::PipelineShaderStageCreateInfo {
            flags: Default::default(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: frag,
            p_name: b"main\0".as_ptr() as *const i8,
            p_specialization_info: ptr::null(),
            .. Default::default()
        }
    ];

    let mut set_layouts = Vec::new();
    let mut set_layout_ids = Vec::new();

    let (layout_handle, layout_id) = descriptor_set_layout_cache.create_descriptor_set_layout_from_interface::<BackgroundShaderInterface>(device);
    set_layouts.push(layout_handle);
    set_layout_ids.push(layout_id);

    let pipeline_layout_create_info = vk::PipelineLayoutCreateInfo {
        flags: Default::default(),
        set_layout_count: set_layouts.len() as u32,
        p_set_layouts: set_layouts.as_ptr(),
        push_constant_range_count: 0,
        p_push_constant_ranges: ptr::null(),
        .. Default::default()
    };

    let pipeline_layout = unsafe {
        device.create_pipeline_layout(&pipeline_layout_create_info, None).unwrap()
    };

    let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
        flags: Default::default(),
        vertex_binding_description_count: VertexInput::BINDINGS.len(),
        p_vertex_binding_descriptions: VertexInput::BINDINGS.as_ptr(),
        vertex_attribute_description_count: VertexInput::ATTRIBUTES.len(),
        p_vertex_attribute_descriptions: VertexInput::ATTRIBUTES.as_ptr(),
        .. Default::default()
    };

    let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
        flags: Default::default(),
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: 0,
        .. Default::default()
    };

    let tessellation_state = vk::PipelineTessellationStateCreateInfo {
        flags: Default::default(),
        patch_control_points: 0,
        .. Default::default()
    };

    let viewport_state = vk::PipelineViewportStateCreateInfo {
        flags: Default::default(),
        viewport_count: 0,
        p_viewports: ptr::null(),
        scissor_count: 0,
        p_scissors: ptr::null(),
        .. Default::default()
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
        line_width: 0.0,
        .. Default::default()
    };

    let multisample_state = vk::PipelineMultisampleStateCreateInfo {
        flags: Default::default(),
        rasterization_samples: Default::default(),
        sample_shading_enable: 0,
        min_sample_shading: 0.0,
        p_sample_mask: ptr::null(),
        alpha_to_coverage_enable: vk::FALSE,
        alpha_to_one_enable: vk::FALSE,
        .. Default::default()
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
        .. Default::default()
    };

    let color_blend_attachments = &[
        vk::PipelineColorBlendAttachmentState {
            blend_enable: vk::FALSE,
            src_color_blend_factor: Default::default(),
            dst_color_blend_factor: Default::default(),
            color_blend_op: Default::default(),
            src_alpha_blend_factor: Default::default(),
            dst_alpha_blend_factor: Default::default(),
            alpha_blend_op: Default::default(),
            color_write_mask: Default::default()
        }
    ];
    
    let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
        flags: Default::default(),
        logic_op_enable: 0,
        logic_op: Default::default(),
        attachment_count: color_blend_attachments.len() as u32,
        p_attachments: color_blend_attachments.as_ptr(),
        blend_constants: [0.0f32;4],
        .. Default::default()
    };

    let dynamic_states = &[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

    let dynamic_state = vk::PipelineDynamicStateCreateInfo {
        flags: Default::default(),
        dynamic_state_count: dynamic_states.len() as u32,
        p_dynamic_states: dynamic_states.as_ptr(),
        .. Default::default()
    };

    let render_pass_attachments = &[vk::AttachmentDescription {
        flags: vk::AttachmentDescriptionFlags::MAY_ALIAS,
        format: vk::Format::R16G16B16A16_SFLOAT,      // TODO get from engine / pass as parameter
        samples: vk::SampleCountFlags::TYPE_1,
        load_op: vk::AttachmentLoadOp::DONT_CARE,
        store_op: vk::AttachmentStoreOp::STORE,
        stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
        stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
        initial_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        final_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    }];

    let color_attachments = &[vk::AttachmentReference {
        attachment: 0,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
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
        p_preserve_attachments: ptr::null()
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
        .. Default::default()
    };

    let render_pass = unsafe {
        device.create_render_pass(&render_pass_create_info, None).unwrap()
    };

    let gpci = vk::GraphicsPipelineCreateInfo {
        flags: Default::default(),
        stage_count: stages.len() as u32,
        p_stages: stage_create_infos.as_ptr(),
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
        .. Default::default()
    };

    let pipeline = unsafe {
        device.create_graphics_pipelines(vk::PipelineCache::null(), &[gpci], None).unwrap()[0]
    };

}

fn load_image(batch: &mut graal::Batch,
    path: &Path,
    usage: graal::vk::ImageUsageFlags,
    mipmaps: bool) -> (graal::ResourceId, u32, u32)
{
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
    let image_id = batch.context().create_image_resource(
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
            transient: false,
        },
    );

    let byte_size = width as u64 * height as u64 * bpp as u64;

    // create a staging buffer
    let (staging_buffer_id, mapped_ptr) = batch.context().create_buffer_resource(
        "staging",
        &ResourceMemoryInfo::HOST_VISIBLE_COHERENT,
        &BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            byte_size,
            transient: true,
            map_on_create: true,
        },
    );

    // read image data
    unsafe {
        image_input
            .read_unchecked(0, 0, 0..nchannels, format_typedesc, mapped_ptr, bpp)
            .expect("failed to read image");
    }

    // build the upload pass
    let mut pass = batch.build_graphics_pass("image upload");
    pass.add_image_usage(
        image_id,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
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
            context.buffer_handle(staging_buffer_id),
            context.image_handle(image_id),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions,
        );
    });
    pass.finish();
    (image_id, width, height)
}

fn create_transient_image(context: &mut graal::Context, name: &str) -> graal::ResourceId {
    context.create_image_resource(
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
            transient: true,
        },
    )
}

struct MeshData {
    vertex_data: graal::ResourceId,
    vertex_count: usize,
}

fn load_mesh(batch: &mut graal::Batch, obj_file_path: &Path) -> MeshData {
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

    let (id, _) = batch.context().create_buffer_resource(
        obj_file_path.to_str().unwrap(),
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            byte_size,
            transient: false,
            map_on_create: false,
        },
    );

    // staging
    let (staging_id, staging_ptr) = batch.context().create_buffer_resource(
        "staging",
        &graal::ResourceMemoryInfo::HOST_VISIBLE_COHERENT,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            byte_size,
            transient: true,
            map_on_create: true,
        },
    );

    unsafe {
        ptr::copy(
            vertices.as_ptr(),
            staging_ptr as *mut Vertex,
            vertices.len(),
        );
    }

    // upload
    let mut upload_pass = batch.build_transfer_pass("upload mesh", false);
    upload_pass.add_buffer_usage(
        staging_id,
        vk::AccessFlags::TRANSFER_READ,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
    );
    upload_pass.add_buffer_usage(
        id,
        vk::AccessFlags::TRANSFER_WRITE,
        vk::PipelineStageFlags::TRANSFER,
        vk::PipelineStageFlags::TRANSFER,
    );
    upload_pass.set_commands(move |context, command_buffer| {
        let src_buffer = context.buffer_handle(staging_id);
        let dst_buffer = context.buffer_handle(id);
        unsafe {
            context.device().cmd_copy_buffer(
                command_buffer,
                src_buffer,
                dst_buffer,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: byte_size,
                }],
            );
        }
    });
    upload_pass.finish();

    MeshData {
        vertex_data: id,
        vertex_count: vertices.len(),
    }
}

fn test_pass(
    batch: &mut graal::Batch,
    name: &str,
    images: &[(
        graal::ResourceId,
        graal::vk::AccessFlags,
        graal::vk::PipelineStageFlags,
        graal::vk::PipelineStageFlags,
        graal::vk::ImageLayout,
    )],
) {
    let mut pass_builder = batch.build_graphics_pass(name);
    for &(img, access_mask, input_stage, output_stage, layout) in images {
        pass_builder.add_image_usage(img, access_mask, input_stage, output_stage, layout);
    }
    pass_builder.finish();
}

fn color_attachment_output(
    img: graal::ResourceId,
) -> (
    graal::ResourceId,
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
    img: graal::ResourceId,
) -> (
    graal::ResourceId,
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
    img: graal::ResourceId,
) -> (
    graal::ResourceId,
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
    img: graal::ResourceId,
) -> (
    graal::ResourceId,
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

    let mut descriptor_set_layout_cache = graal::DescriptorSetLayoutCache::new();
    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle());
    let device = graal::Device::new(surface);
    let mut context = graal::Context::new(device);
    let swapchain = unsafe { context.create_swapchain(surface, window.inner_size().into()) };

    let mut init_batch = context.start_batch();
    let (file_image_id, file_image_width, file_image_height) = load_image(
        &mut init_batch,
        "../data/El4KUGDU0AAW64U.jpg".as_ref(),
        vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
        false,
    );
    let mesh = load_mesh(&mut init_batch, "../data/sphere.obj".as_ref());
    init_batch.finish();

    create_pipeline(context.device(), &mut descriptor_set_layout_cache);

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

                test_pass(&mut batch, "P0", &[color_attachment_output(img_a)]);
                test_pass(&mut batch, "P1", &[color_attachment_output(img_b)]);
                test_pass(
                    &mut batch,
                    "P2",
                    &[
                        compute_read(img_a),
                        compute_read(img_b),
                        compute_write(img_d1),
                        compute_write(img_d2),
                    ],
                );
                test_pass(&mut batch, "P3", &[color_attachment_output(img_c)]);
                test_pass(
                    &mut batch,
                    "P4",
                    &[
                        compute_read(img_d2),
                        compute_read(img_c),
                        compute_write(img_e),
                    ],
                );
                test_pass(
                    &mut batch,
                    "P5",
                    &[compute_read(img_d1), compute_write(img_f)],
                );
                test_pass(
                    &mut batch,
                    "P6",
                    &[
                        compute_read(img_e),
                        compute_read(img_f),
                        compute_write(img_g),
                    ],
                );
                test_pass(
                    &mut batch,
                    "P7",
                    &[compute_read(img_g), compute_write(img_h)],
                );
                test_pass(
                    &mut batch,
                    "P8",
                    &[compute_read(img_h), compute_write(img_i)],
                );
                test_pass(
                    &mut batch,
                    "P9",
                    &[
                        compute_read(img_i),
                        compute_read(img_g),
                        compute_write(img_j),
                    ],
                );
                test_pass(
                    &mut batch,
                    "P10",
                    &[compute_read(img_j), compute_write(img_k)],
                );

                test_pass(
                    &mut batch,
                    "P11",
                    &[color_attachment_output(swapchain_image.image_id)],
                );

                // blit pass
                let mut blit_pass = batch.build_graphics_pass("blit to screen");
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
                blit_pass.finish();

                batch.present("P12", &swapchain_image);

                batch.finish();
            }
            _ => (),
        }
    });
}

#[pass(graphics)]
fn blit_pass(
    context: &Context,
    command_buffer: vk::CommandBuffer,
    #[resource(
        usage=transfer_src,
        layout=transfer_src_optimal,
        input_stages(transfer),
        output_stages(transfer),
        access_mask(transfer_read),
    )]
    src_image: vk::Image,
    #[resource(
        usage=transfer_dst,
        layout=transfer_dst_optimal,
        input_stages(transfer),
        output_stages(transfer),
        access_mask(transfer_write),
    )]
    dst_image: vk::Image,
    filter: vk::Filter)
{
}


#[pass(graphics)]
fn main_pass(
    context: &Context,
    command_buffer: vk::CommandBuffer,


)