use ash::{
    vk::{BufferUsageFlags, Rect2D, SampleCountFlags},
};
use graal::{vk, BufferResourceCreateInfo, FrameCreateInfo, ImageId, ImageInfo, ImageResourceCreateInfo, ResourceId, ResourceMemoryInfo, MemoryLocation};
use inline_spirv::include_spirv;
use raw_window_handle::HasRawWindowHandle;
use std::{mem, path::Path, ptr};
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};
use graal::swapchain::Swapchain;

fn load_image(
    context: &mut graal::Context,
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
    } = context.create_image(
        path.to_str().unwrap(),
         MemoryLocation::GpuOnly,
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
    let mut staging_buffer = 
        context.create_buffer("staging", MemoryLocation::CpuToGpu, &BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::TRANSFER_SRC,
            byte_size,
            map_on_create: true
        });

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

    let staging_buffer_handle = staging_buffer.handle;

    // build the upload pass
    context.add_graphics_pass("image upload", |pass| {
        pass.reference_image(
            image_id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );
        pass.reference_buffer(
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

    context.destroy_buffer(staging_buffer.id);
    (image_id, width, height)
}

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .init();

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle());
    let device = graal::Device::new(Some(surface));
    let mut context = graal::Context::with_device(device);
    let mut swapchain = unsafe { Swapchain::new(&context, surface, window.inner_size().into()) };
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
                    swapchain.resize(&context, swapchain_size);
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let swapchain_image = unsafe { swapchain.acquire_next_image(&mut context) };

                context.start_frame(FrameCreateInfo { collect_debug_info: true, happens_after: Default::default() });

                let (file_image_id, file_image_width, file_image_height) = load_image(
                    &mut context,
                    "data/haniyasushin_keiki.jpg".as_ref(),
                    vk::ImageUsageFlags::TRANSFER_SRC | vk::ImageUsageFlags::SAMPLED,
                    false,
                );

                context.add_graphics_pass("blit to screen", |pass| {
                    pass.reference_image(
                        file_image_id,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    );
                    pass.reference_image(
                        swapchain_image.image_info.id,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TRANSFER,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    );

                    let blit_w = file_image_width.min(swapchain_size.0);
                    let blit_h = file_image_height.min(swapchain_size.1);

                    pass.set_commands(move |context, command_buffer| {
                        let dst_image_handle = context.image_handle(swapchain_image.image_info.id);
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
                            context.vulkan_device().cmd_blit_image(
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
                });

                context.present("P12", &swapchain_image);
                context.end_frame();

                context.destroy_image(file_image_id);
                context.destroy_image(swapchain_image.image_info.id);
            }
            _ => (),
        }
    });
}
