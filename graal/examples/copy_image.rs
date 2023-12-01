use graal::{
    device::{BufferResourceCreateInfo, Device, ImageHandle, ImageResourceCreateInfo},
    queue,
    queue::{Queue, ResourceState, Submission},
    vk, MemoryLocation,
};
use raw_window_handle::HasRawWindowHandle;
use std::{mem, path::Path, ptr, time::Duration};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

struct LoadImageResult {
    image_info: ImageHandle,
    width: u32,
    height: u32,
}

fn load_image(queue: &mut Queue, path: impl AsRef<Path>, usage: vk::ImageUsageFlags, mipmaps: bool) -> LoadImageResult {
    use openimageio::{ImageInput, TypeDesc};

    let path = path.as_ref();
    let device = queue.device().clone();

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

    let mip_levels = graal::mip_level_count(width, height);

    // create the texture
    let ImageHandle {
        vk: image_handle,
        id: image_id,
    } = device.create_image(
        path.to_str().unwrap(),
        MemoryLocation::GpuOnly,
        &ImageResourceCreateInfo {
            type_: vk::ImageType::TYPE_2D,
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
    let staging_buffer = device.create_buffer(
        "staging",
        MemoryLocation::CpuToGpu,
        &BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            byte_size,
            map_on_create: true,
        },
    );

    // read image data
    unsafe {
        image_input
            .read_unchecked(
                0,
                0,
                0..nchannels,
                format_typedesc,
                staging_buffer.mapped_ptr.unwrap().as_ptr() as *mut u8,
                bpp,
            )
            .expect("failed to read image");
    }

    let staging_buffer_handle = staging_buffer.vk;

    let cb = queue.allocate_command_buffer();

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

    unsafe {
        device.begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            },
        );
        device.cmd_copy_buffer_to_image(
            cb,
            staging_buffer_handle,
            image_handle,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            regions,
        );
        device.end_command_buffer(cb).unwrap();

        let mut upload = Submission::new();
        upload.set_name("image upload");
        upload.use_buffer(staging_buffer.id, ResourceState::TRANSFER_SRC);
        upload.use_image(image_id, ResourceState::TRANSFER_DST);
        upload.push_command_buffer(cb);
        queue.submit(upload).expect("image upload failed");
        queue.device().destroy_buffer(staging_buffer.id);
    }

    LoadImageResult {
        image_info: ImageHandle {
            vk: image_handle,
            id: image_id,
        },
        width,
        height,
    }
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle().unwrap());

    let (device, mut queue) =
        unsafe { graal::create_device_and_queue(Some(surface)).expect("failed to create device") };
    let surface_format = unsafe { device.get_preferred_surface_format(surface) };
    let mut swapchain = unsafe { device.create_swapchain(surface, surface_format, window.inner_size().into()) };
    let mut swapchain_size = window.inner_size().into();

    event_loop
        .run(move |event, event_loop| {
            match event {
                Event::WindowEvent { window_id, event } => match event {
                    WindowEvent::CloseRequested => {
                        println!("The close button was pressed; stopping");
                        event_loop.exit();
                    }
                    WindowEvent::Resized(size) => unsafe {
                        swapchain_size = size.into();
                        device.resize_swapchain(&mut swapchain, swapchain_size);
                    },
                    WindowEvent::RedrawRequested => unsafe {
                        // SAFETY: swapchain is valid
                        let swapchain_image =
                            queue.acquire_next_swapchain_image(&swapchain, Duration::from_millis(100));

                        let swapchain_image = match swapchain_image {
                            Ok(image) => image,
                            Err(err) => {
                                eprintln!("vkAcquireNextImage failed: {}", err);
                                return;
                            }
                        };

                        let LoadImageResult {
                            image_info,
                            width,
                            height,
                        } = load_image(
                            &mut queue,
                            "data/haniyasushin_keiki.jpg",
                            vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
                            false,
                        );

                        let blit_w = width.min(swapchain_size.0);
                        let blit_h = height.min(swapchain_size.1);
                        let swapchain_image_id = swapchain_image.handle.id;

                        let cb = queue.allocate_command_buffer();

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

                        let device = queue.device();
                        device
                            .begin_command_buffer(
                                cb,
                                &vk::CommandBufferBeginInfo {
                                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                    ..Default::default()
                                },
                            )
                            .unwrap();
                        device.cmd_blit_image(
                            cb,
                            image_info.vk,
                            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                            swapchain_image.handle.vk,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            regions,
                            vk::Filter::NEAREST,
                        );
                        device.end_command_buffer(cb).unwrap();

                        let mut blit = Submission::new();
                        blit.set_name("blit");
                        blit.use_image(image_info.id, ResourceState::TRANSFER_SRC);
                        blit.use_image(swapchain_image_id, ResourceState::TRANSFER_DST);
                        blit.push_command_buffer(cb);
                        let render_finished = blit.signal(queue.get_or_create_semaphore());
                        queue.submit(blit).expect("blit failed");

                        // transition swapchain image to present
                        queue
                            .transition_image(swapchain_image_id, ResourceState::PRESENT)
                            .unwrap();
                        queue.present(render_finished, &swapchain_image).unwrap();
                        queue.end_frame().unwrap();

                        queue.device().destroy_image(image_info.id);
                    },
                    _ => {}
                },
                Event::AboutToWait => {
                    window.request_redraw();
                }
                _ => (),
            }
        })
        .expect("event loop run failed");
}
