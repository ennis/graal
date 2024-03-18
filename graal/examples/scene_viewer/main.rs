mod aabb;
mod app;
mod camera_control;
mod mesh;
mod scene;

use winit::{
    event::{Event, WindowEvent},
    event_loop::EventLoop,
    raw_window_handle::HasRawWindowHandle,
    window::WindowBuilder,
};

fn main() {
    let event_loop = EventLoop::new().expect("failed to create event loop");
    let window = WindowBuilder::new().build(&event_loop).unwrap();

    let surface = graal::get_vulkan_surface(window.raw_window_handle().unwrap());

    let (device, mut queue) =
        unsafe { graal::create_device_and_command_stream(Some(surface)).expect("failed to create device") };
    let surface_format = unsafe { device.get_preferred_surface_format(surface) };
    let mut swapchain = unsafe { device.create_swapchain(surface, surface_format, window.inner_size().into()) };

    let mut swapchain_size = window.inner_size().into();

    event_loop
        .run(move |event, event_loop| match event {
            Event::WindowEvent { window_id, event } => match event {
                WindowEvent::CloseRequested => {
                    println!("The close button was pressed; stopping");
                    event_loop.exit();
                }
                WindowEvent::Resized(size) => unsafe {
                    swapchain_size = size.into();
                    device.resize_swapchain(&mut swapchain, swapchain_size);
                },
                WindowEvent::RedrawRequested => unsafe {},
                _ => {}
            },
            Event::AboutToWait => {
                window.request_redraw();
            }
            _ => (),
        })
        .expect("event loop run failed");
}
