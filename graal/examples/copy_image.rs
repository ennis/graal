use std::{path::Path, ptr, time::Duration};
use image::DynamicImage;

use raw_window_handle::HasRawWindowHandle;
use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    window::WindowBuilder,
};

use graal::{
    vk, BufferCreateInfo, BufferUsage, Image, ImageCopyBuffer, ImageCopyView, ImageCreateInfo, ImageDataLayout,
    ImageSubresourceLayers, ImageType, ImageUsage, MemoryLocation, Point3D, Queue, Rect3D,
};

fn load_image(queue: &mut Queue, path: impl AsRef<Path>, usage: ImageUsage, mipmaps: bool) -> Image {

    let path = path.as_ref();
    let device = queue.device().clone();

    let dyn_image = image::open(path).expect("could not open image file");

    let (vk_format, bpp) = match dyn_image {
        DynamicImage::ImageLuma8(_) => (vk::Format::R8_UNORM, 1usize),
        DynamicImage::ImageLumaA8(_) => (vk::Format::R8G8_UNORM, 2usize),
        DynamicImage::ImageRgb8(_) => (vk::Format::R8G8B8_SRGB, 3usize),
        DynamicImage::ImageRgba8(_) => (vk::Format::R8G8B8A8_SRGB, 4usize),
        _ => unimplemented!()
    };

    let width = dyn_image.width();
    let height = dyn_image.height();

    let mip_levels = graal::mip_level_count(width, height);

    // create the texture
    let image = device.create_image(
        path.to_str().unwrap(),
        &ImageCreateInfo {
            memory_location: MemoryLocation::GpuOnly,
            type_: ImageType::Image2D,
            usage: usage | ImageUsage::TRANSFER_DST,
            format: vk_format,
            width,
            height,
            depth: 1,
            mip_levels,
            array_layers: 1,
            samples: 1,
        },
    );

    let byte_size = width as u64 * height as u64 * bpp as u64;

    // create a staging buffer
    let staging_buffer = device.create_buffer(
        "staging",
        BufferUsage::TRANSFER_SRC,
        MemoryLocation::CpuToGpu,
        byte_size,
    );

    // read image data
    unsafe {
        ptr::copy_nonoverlapping(dyn_image.as_bytes().as_ptr(), staging_buffer.mapped_data().unwrap(), byte_size as usize);
    }

    let mut cmd_buf = queue.create_command_buffer();
    let mut encoder = cmd_buf.begin_blit();
    unsafe {
        encoder.copy_buffer_to_image(
            ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: ImageDataLayout {
                    offset: 0,
                    row_length: Some(width),
                    image_height: Some(height),
                },
            },
            ImageCopyView {
                image: &image,
                mip_level: 0,
                origin: vk::Offset3D { x: 0, y: 0, z: 0 },
                aspect: vk::ImageAspectFlags::COLOR,
            },
            vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        );
    }
    drop(encoder);

    queue.submit([cmd_buf]).unwrap();

    image
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::get_vulkan_surface(window.raw_window_handle().unwrap());

    let (device, mut queue) =
        unsafe { graal::create_device_and_queue(Some(surface)).expect("failed to create device") };
    let surface_format = unsafe { device.get_preferred_surface_format(surface) };
    let window_size = window.inner_size();
    let mut swapchain =
        unsafe { device.create_swapchain(surface, surface_format, window_size.width, window_size.height) };
    let mut swapchain_size: (u32, u32) = window.inner_size().into();

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
                        device.resize_swapchain(&mut swapchain, swapchain_size.0, swapchain_size.1);
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

                        let image = load_image(
                            &mut queue,
                            "data/yukari.png",
                            ImageUsage::TRANSFER_SRC | ImageUsage::SAMPLED,
                            false,
                        );

                        let blit_w = image.size().width.min(swapchain_size.0);
                        let blit_h = image.size().height.min(swapchain_size.1);

                        let mut cb = queue.create_command_buffer();
                        let mut encoder = cb.begin_blit();
                        unsafe {
                            encoder.blit_image(
                                &image,
                                ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                Rect3D {
                                    min: Point3D { x: 0, y: 0, z: 0 },
                                    max: Point3D {
                                        x: blit_w as i32,
                                        y: blit_h as i32,
                                        z: 1,
                                    },
                                },
                                &swapchain_image.image,
                                ImageSubresourceLayers {
                                    aspect_mask: vk::ImageAspectFlags::COLOR,
                                    mip_level: 0,
                                    base_array_layer: 0,
                                    layer_count: 1,
                                },
                                Rect3D {
                                    min: Point3D { x: 0, y: 0, z: 0 },
                                    max: Point3D {
                                        x: blit_w as i32,
                                        y: blit_h as i32,
                                        z: 1,
                                    },
                                },
                                vk::Filter::NEAREST,
                            );
                        }
                        drop(encoder);

                        queue.submit([cb]).expect("blit failed");
                        queue.present(&swapchain_image).unwrap();
                        queue.end_frame().unwrap();
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
