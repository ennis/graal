use graal::CommandStream;
use winit::{
    event::MouseButton,
    keyboard::{Key, NamedKey},
};

pub struct App {}

impl App {
    pub fn new(width: u32, height: u32) -> App {
        App {}
    }

    pub fn resize(&mut self, w: u32, h: u32) {}

    pub fn mouse_input(&mut self, button: MouseButton, pos: glam::DVec2, pressed: bool) {}

    pub fn cursor_moved(&mut self, pos: glam::DVec2) {}

    pub fn key_input(&mut self, key: winit::keyboard::Key, pressed: bool) {
        if key == Key::Named(NamedKey::F5) && pressed {
            self.reload_shaders();
        }
    }

    pub fn reload_shaders(&mut self) {
        // TODO
    }

    pub fn render(&mut self, queue: &mut CommandStream) {}
    pub fn on_exit(&mut self) {}
}
