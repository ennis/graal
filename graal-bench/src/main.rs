use std::hash::{Hash, Hasher};

use egui::{Key, TouchDeviceId};
use raw_window_handle::HasRawWindowHandle;
use tracing_subscriber;
use winit::{
    event::{Event, Force, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use graal::swapchain::Swapchain;
use graal::vk;

use crate::{
    background::BackgroundPass,
    camera::{CameraControl, CameraControlInput, CameraControlMouseButton},
    geometry_pass::GeometryPass,
    scene::Scene,
};

mod background;
mod bounding_box;
mod camera;
mod egui_renderer;
mod geometry_pass;
mod load_image;
mod mesh;
mod pipeline;
mod scene;
mod shader;
mod taa;
pub mod vertex;
pub mod fragment_output;
pub mod descriptor;
pub mod buffer;

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
    let mut swapchain = unsafe { Swapchain::new(&context, surface, window.inner_size().into()) };

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

    let mut egui_ctx = egui::CtxRef::default();
    let mut egui_renderer = egui_renderer::EguiRenderer::new(&mut context, swapchain.format);
    let mut winit_input_state = WinitInputState::from_pixels_per_point(1.0);

    camera_control.center_on_bounds(&scene.bounds(), std::f64::consts::FRAC_PI_2);
    let mut dump_next_frame = false;

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        let mut egui_input_state = egui::RawInput::default();

        match event {
            Event::WindowEvent { window_id, event } => {
                input_to_egui(1.0, &event, &mut winit_input_state, control_flow);

                match event {
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
                        swapchain.resize(&context, swapchain_size);
                        camera_control.set_screen_size(glam::dvec2(
                            swapchain_size.0 as f64,
                            swapchain_size.1 as f64,
                        ));
                    },
                    _ => {}
                }
            },
            Event::MainEventsCleared => {
                window.request_redraw();
            }

            Event::RedrawRequested(_) => {
                let swapchain_image = unsafe { swapchain.acquire_next_image(&mut context) };

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
                graal::utils::blit_images(
                    &frame,
                    gbuffers.color,
                    swapchain_image.image_info,
                    swapchain_size,
                    vk::ImageAspectFlags::COLOR,
                );

                // draw GUI
                winit_input_state.raw.screen_rect = Some(egui::Rect::from_min_size(
                    Default::default(),
                    egui::Vec2::new(swapchain_size.0 as f32, swapchain_size.1 as f32),
                ));
                egui_ctx.begin_frame(winit_input_state.raw.take());

                egui::SidePanel::left("my_side_panel", 300.0).show(&egui_ctx, |ui| {
                    ui.heading("Hello World!");
                    if ui.button("Quit").clicked() {
                        //quit = true;
                    }

                    egui::ComboBox::from_label("Version")
                        .width(150.0)
                        .selected_text("foo")
                        .show_ui(ui, |ui| {
                            egui::CollapsingHeader::new("Dev")
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.label("contents");
                                });
                        });
                });

                let (_output , clipped_shapes) = egui_ctx.end_frame();
                let clipped_meshes = egui_ctx.tessellate(clipped_shapes);
                egui_renderer.render(&frame, swapchain_image.image_info, swapchain_size, clipped_meshes, &egui_ctx.texture());

                // present
                frame.present("present", &swapchain_image);

                if dump_next_frame {
                    frame.dump(Some("bench"));
                    dump_next_frame = false;
                }

                frame.finish();

                context.destroy_image(swapchain_image.image_info.id);
            }
            _ => (),
        }
    });
}
