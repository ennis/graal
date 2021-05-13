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

use crate::{
    background::BackgroundPass,
    camera::{CameraControl, CameraControlInput, CameraControlMouseButton},
    geometry_pass::GeometryPass,
    scene::Scene,
};
use egui::{Key, TouchDeviceId};
use graal::vk;
use raw_window_handle::HasRawWindowHandle;
use std::hash::{Hash, Hasher};
use tracing_subscriber;
use winit::{
    event::{Event, Force, MouseButton, VirtualKeyCode, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

pub struct WinitInputState {
    pub pointer_pos_in_points: Option<egui::Pos2>,
    pub raw: egui::RawInput,
}

impl WinitInputState {
    pub fn from_pixels_per_point(pixels_per_point: f32) -> Self {
        Self {
            pointer_pos_in_points: Default::default(),
            raw: egui::RawInput {
                pixels_per_point: Some(pixels_per_point),
                ..Default::default()
            },
        }
    }
}

fn input_to_egui(
    pixels_per_point: f32,
    event: &WindowEvent,
    input_state: &mut WinitInputState,
    control_flow: &mut ControlFlow,
) {
    match event {
        WindowEvent::CloseRequested | WindowEvent::Destroyed => *control_flow = ControlFlow::Exit,
        WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
            input_state.raw.pixels_per_point = Some(*scale_factor as f32);
        }
        WindowEvent::MouseInput { state, button, .. } => {
            if let Some(pos_in_points) = input_state.pointer_pos_in_points {
                if let Some(button) = translate_mouse_button(*button) {
                    input_state.raw.events.push(egui::Event::PointerButton {
                        pos: pos_in_points,
                        button,
                        pressed: state == &winit::event::ElementState::Pressed,
                        modifiers: input_state.raw.modifiers,
                    });
                }
            }
        }
        WindowEvent::CursorMoved {
            position: pos_in_pixels,
            ..
        } => {
            let pos_in_points = egui::Pos2::new(
                pos_in_pixels.x as f32 / pixels_per_point,
                pos_in_pixels.y as f32 / pixels_per_point,
            );
            input_state.pointer_pos_in_points = Some(pos_in_points);
            input_state
                .raw
                .events
                .push(egui::Event::PointerMoved(pos_in_points));
        }
        WindowEvent::CursorLeft { .. } => {
            input_state.pointer_pos_in_points = None;
            input_state.raw.events.push(egui::Event::PointerGone);
        }
        WindowEvent::ReceivedCharacter(ch) => {
            if is_printable_char(*ch)
                && !input_state.raw.modifiers.ctrl
                && !input_state.raw.modifiers.mac_cmd
            {
                input_state.raw.events.push(egui::Event::Text(ch.to_string()));
            }
        }
        WindowEvent::KeyboardInput { input, .. } => {
            if let Some(keycode) = input.virtual_keycode {
                let pressed = input.state == winit::event::ElementState::Pressed;

                // We could also use `WindowEvent::ModifiersChanged` instead, I guess.
                if matches!(keycode, VirtualKeyCode::LAlt | VirtualKeyCode::RAlt) {
                    input_state.raw.modifiers.alt = pressed;
                }
                if matches!(keycode, VirtualKeyCode::LControl | VirtualKeyCode::RControl) {
                    input_state.raw.modifiers.ctrl = pressed;
                    if !cfg!(target_os = "macos") {
                        input_state.raw.modifiers.command = pressed;
                    }
                }
                if matches!(keycode, VirtualKeyCode::LShift | VirtualKeyCode::RShift) {
                    input_state.raw.modifiers.shift = pressed;
                }
                if cfg!(target_os = "macos")
                    && matches!(keycode, VirtualKeyCode::LWin | VirtualKeyCode::RWin)
                {
                    input_state.raw.modifiers.mac_cmd = pressed;
                    input_state.raw.modifiers.command = pressed;
                }

                if pressed {
                    if cfg!(target_os = "macos")
                        && input_state.raw.modifiers.mac_cmd
                        && keycode == VirtualKeyCode::Q
                    {
                        *control_flow = ControlFlow::Exit;
                    }

                    // VirtualKeyCode::Paste etc in winit are broken/untrustworthy,
                    // so we detect these things manually:
                    if input_state.raw.modifiers.command && keycode == VirtualKeyCode::X {
                        input_state.raw.events.push(egui::Event::Cut);
                    } else if input_state.raw.modifiers.command && keycode == VirtualKeyCode::C {
                        input_state.raw.events.push(egui::Event::Copy);
                    } else if input_state.raw.modifiers.command && keycode == VirtualKeyCode::V {
                        /*if let Some(clipboard) = clipboard {
                            match clipboard.get_contents() {
                                Ok(contents) => {
                                    input_state.raw.events.push(Event::Text(contents));
                                }
                                Err(err) => {
                                    eprintln!("Paste error: {}", err);
                                }
                            }
                        }*/
                    }
                }

                if let Some(key) = translate_virtual_key_code(keycode) {
                    input_state.raw.events.push(egui::Event::Key {
                        key,
                        pressed,
                        modifiers: input_state.raw.modifiers,
                    });
                }
            }
        }
        WindowEvent::MouseWheel { delta, .. } => {
            let mut delta = match delta {
                winit::event::MouseScrollDelta::LineDelta(x, y) => {
                    let line_height = 8.0; // magic value!
                    egui::Vec2::new(*x, *y) * line_height
                }
                winit::event::MouseScrollDelta::PixelDelta(delta) => {
                    egui::Vec2::new(delta.x as f32, delta.y as f32) / pixels_per_point
                }
            };
            if cfg!(target_os = "macos") {
                // This is still buggy in winit despite
                // https://github.com/rust-windowing/winit/issues/1695 being closed
                delta.x *= -1.0;
            }

            if input_state.raw.modifiers.ctrl {
                // Treat as zoom instead:
                input_state.raw.zoom_delta *= (delta.y / 200.0).exp();
            } else {
                input_state.raw.scroll_delta += delta;
            }
        }
        WindowEvent::TouchpadPressure {
            // device_id,
            // pressure,
            // stage,
            ..
        } => {
            // TODO
        }
        WindowEvent::Touch(touch) => {
            let pixels_per_point_recip = 1. / pixels_per_point;
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            touch.device_id.hash(&mut hasher);
            input_state.raw.events.push(egui::Event::Touch {
                device_id: TouchDeviceId(hasher.finish()),
                id: egui::TouchId::from(touch.id),
                phase: match touch.phase {
                    winit::event::TouchPhase::Started => egui::TouchPhase::Start,
                    winit::event::TouchPhase::Moved => egui::TouchPhase::Move,
                    winit::event::TouchPhase::Ended => egui::TouchPhase::End,
                    winit::event::TouchPhase::Cancelled => egui::TouchPhase::Cancel,
                },
                pos: egui::Pos2::new(touch.location.x as f32 * pixels_per_point_recip,
                          touch.location.y as f32 * pixels_per_point_recip),
                force: match touch.force {
                    Some(Force::Normalized(force)) => force as f32,
                    Some(Force::Calibrated {
                             force,
                             max_possible_force,
                             ..
                         }) => (force / max_possible_force) as f32,
                    None => 0_f32,
                },
            });
        }
        _ => {
            // dbg!(event);
        }
    }
}

/// Glium sends special keys (backspace, delete, F1, ...) as characters.
/// Ignore those.
/// We also ignore '\r', '\n', '\t'.
/// Newlines are handled by the `Key::Enter` event.
fn is_printable_char(chr: char) -> bool {
    let is_in_private_use_area = '\u{e000}' <= chr && chr <= '\u{f8ff}'
        || '\u{f0000}' <= chr && chr <= '\u{ffffd}'
        || '\u{100000}' <= chr && chr <= '\u{10fffd}';

    !is_in_private_use_area && !chr.is_ascii_control()
}

pub fn translate_mouse_button(button: winit::event::MouseButton) -> Option<egui::PointerButton> {
    match button {
        winit::event::MouseButton::Left => Some(egui::PointerButton::Primary),
        winit::event::MouseButton::Right => Some(egui::PointerButton::Secondary),
        winit::event::MouseButton::Middle => Some(egui::PointerButton::Middle),
        _ => None,
    }
}

pub fn translate_virtual_key_code(key: VirtualKeyCode) -> Option<egui::Key> {
    use VirtualKeyCode::*;

    Some(match key {
        Down => Key::ArrowDown,
        Left => Key::ArrowLeft,
        Right => Key::ArrowRight,
        Up => Key::ArrowUp,

        Escape => Key::Escape,
        Tab => Key::Tab,
        Back => Key::Backspace,
        Return => Key::Enter,
        Space => Key::Space,

        Insert => Key::Insert,
        Delete => Key::Delete,
        Home => Key::Home,
        End => Key::End,
        PageUp => Key::PageUp,
        PageDown => Key::PageDown,

        Key0 | Numpad0 => Key::Num0,
        Key1 | Numpad1 => Key::Num1,
        Key2 | Numpad2 => Key::Num2,
        Key3 | Numpad3 => Key::Num3,
        Key4 | Numpad4 => Key::Num4,
        Key5 | Numpad5 => Key::Num5,
        Key6 | Numpad6 => Key::Num6,
        Key7 | Numpad7 => Key::Num7,
        Key8 | Numpad8 => Key::Num8,
        Key9 | Numpad9 => Key::Num9,

        A => Key::A,
        B => Key::B,
        C => Key::C,
        D => Key::D,
        E => Key::E,
        F => Key::F,
        G => Key::G,
        H => Key::H,
        I => Key::I,
        J => Key::J,
        K => Key::K,
        L => Key::L,
        M => Key::M,
        N => Key::N,
        O => Key::O,
        P => Key::P,
        Q => Key::Q,
        R => Key::R,
        S => Key::S,
        T => Key::T,
        U => Key::U,
        V => Key::V,
        W => Key::W,
        X => Key::X,
        Y => Key::Y,
        Z => Key::Z,

        _ => {
            return None;
        }
    })
}

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
                        context.resize_swapchain(swapchain.id, swapchain_size);
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
                let swapchain_image = unsafe { context.acquire_next_image(swapchain.id) };
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
            }
            _ => (),
        }
    });
}
