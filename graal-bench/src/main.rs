mod background;
mod blit;
mod bounding_box;
mod camera;
mod geometry_pass;
mod load_image;
mod mesh;
mod pipeline;
mod scene;
mod shader;
mod taa;

use crate::{
    background::BackgroundPass,
    camera::{CameraControl, CameraControlInput, CameraControlMouseButton},
    geometry_pass::GeometryPass,
    scene::Scene,
};
use graal::vk;
use raw_window_handle::HasRawWindowHandle;
use tracing_subscriber;
use winit::{
    event::{Event, MouseButton, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

/*fn create_transient_image(context: &mut graal::Context, name: &str) -> graal::ImageId {
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
}*/

fn main() {
    tracing_subscriber::fmt()
        .with_target(false)
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::ACTIVE)
        .init();

    // Ancient mantra of window and context creation
    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    let surface = graal::surface::get_vulkan_surface(window.raw_window_handle());
    let mut context = graal::Context::with_surface(surface);
    let swapchain = unsafe { context.create_swapchain(surface, window.inner_size().into()) };

    // Create a scene that will hold our objects and buffers.
    let mut scene = Scene::new();

    // Upload new objects and mesh data to the scene.
    {
        let mut scene_uploader = scene.start_upload(&mut context);
        scene_uploader.import_obj("data/reimu.obj");
        scene_uploader.finish();
    }

    let bkg_pass = BackgroundPass::new(&mut context);
    let geom_pass = GeometryPass::new(&mut context);
    let mut swapchain_size: (u32, u32) = window.inner_size().into();
    let mut camera_control = CameraControl::new(glam::dvec2(
        swapchain_size.0 as f64,
        swapchain_size.1 as f64,
    ));

    camera_control.center_on_bounds(&scene.bounds(), std::f64::consts::FRAC_PI_2);
    let mut dump_next_frame = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { window_id, event } => match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    *control_flow = ControlFlow::Exit
                }
                WindowEvent::MouseInput { button, state, .. } => {
                    let button = match button {
                        MouseButton::Left => CameraControlMouseButton::Left,
                        MouseButton::Right => CameraControlMouseButton::Right,
                        MouseButton::Middle => CameraControlMouseButton::Middle,
                        MouseButton::Other(_) => CameraControlMouseButton::Left,
                    };
                    let pressed = match state {
                        winit::event::ElementState::Pressed => true,
                        winit::event::ElementState::Released => false,
                    };
                    camera_control
                        .handle_input(&CameraControlInput::MouseInput { button, pressed });
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    if let Some(winit::event::VirtualKeyCode::F11) = input.virtual_keycode {
                        dump_next_frame = true;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    camera_control.handle_input(&CameraControlInput::CursorMoved {
                        position: glam::dvec2(position.x, position.y),
                    });
                }
                WindowEvent::Resized(size) => unsafe {
                    swapchain_size = size.into();
                    eprintln!("window resized: {},{}", swapchain_size.0, swapchain_size.1);
                    context.resize_swapchain(swapchain, swapchain_size);
                    camera_control.set_screen_size(glam::dvec2(
                        swapchain_size.0 as f64,
                        swapchain_size.1 as f64,
                    ));
                },
                _ => {}
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let swapchain_image = unsafe { context.acquire_next_image(swapchain) };
                let camera = camera_control.camera();
                let mut frame = context.start_frame(Default::default());

                // draw background
                bkg_pass.run(
                    &frame,
                    swapchain_image.image_info,
                    vk::Format::B8G8R8A8_SRGB,
                    swapchain_size,
                );

                // draw our mesh to G-buffers
                let gbuffers = geom_pass.run(&frame, &scene, swapchain_size, &camera);

                // blit?
                blit::blit_images(
                    &frame,
                    gbuffers.color,
                    swapchain_image.image_info,
                    swapchain_size,
                    vk::ImageAspectFlags::COLOR,
                );

                frame.present("present", &swapchain_image);

                if dump_next_frame {
                    frame.dump(Some("bench"));
                    dump_next_frame = false;
                }
                frame.finish();
            }
            _ => (),
        }
    });
}
