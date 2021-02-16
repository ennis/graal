use ash::version::DeviceV1_0;
use graal::{vk, BufferResourceCreateInfo, ImageResourceCreateInfo, ResourceMemoryInfo};
use raw_window_handle::HasRawWindowHandle;
use std::path::Path;
use std::{mem, ptr};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

fn load_image(
    batch: &mut graal::Batch,
    path: &Path,
    usage: graal::vk::ImageUsageFlags,
    mipmaps: bool,
) -> (graal::ResourceId, u32, u32) {
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
