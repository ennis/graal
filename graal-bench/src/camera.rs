use crate::bounding_box::BoundingBox;
use std::f64::consts::TAU;

#[derive(Copy, Clone, Debug, Default)]
pub struct Frustum {
    pub left: f32,
    pub right: f32,
    pub top: f32,
    pub bottom: f32,
    // near clip plane position
    pub near_plane: f32,
    // far clip plane position
    pub far_plane: f32,
}

/// Represents a camera (a view of a scene).
#[derive(Copy, Clone, Debug)]
pub struct Camera {
    // Projection parameters
    // frustum (for culling)
    pub frustum: Frustum,
    // view matrix
    // (World -> View)
    pub view: glam::Mat4,
    // projection matrix
    // (View -> clip?)
    pub projection: glam::Mat4,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            frustum: Default::default(),
            view: glam::Mat4::look_at_rh(
                glam::vec3(0.0, 0.0, -1.0),
                glam::vec3(0.0, 0.0, 0.0),
                glam::Vec3::Y,
            ),
            projection: glam::Mat4::perspective_rh(std::f32::consts::PI / 2.0, 1.0, 0.001, 10.0),
        }
    }
}

#[derive(Copy, Clone, Debug)]
struct CameraFrame {
    eye: glam::DVec3,
    up: glam::DVec3,
    center: glam::DVec3,
}

#[derive(Copy, Clone, Debug)]
enum CameraInputMode {
    None,
    Pan {
        anchor_screen: glam::DVec2,
        orig_frame: CameraFrame,
    },
    Tumble {
        anchor_screen: glam::DVec2,
        orig_frame: CameraFrame,
    },
}

/// A camera controller that generates `Camera` instances.
///
/// TODO describe parameters
#[derive(Clone, Debug)]
pub struct CameraControl {
    fov_y_radians: f64,
    z_near: f64,
    z_far: f64,
    zoom: f32,
    screen_size: glam::DVec2,

    cursor_pos: Option<glam::DVec2>,

    frame: CameraFrame,
    input_mode: CameraInputMode,
}

#[derive(Copy, Clone, Debug)]
pub enum CameraControlMouseButton {
    Left,
    Right,
    Middle,
}

#[derive(Copy, Clone, Debug)]
pub enum CameraControlInput {
    MouseInput {
        button: CameraControlMouseButton,
        pressed: bool,
    },
    CursorMoved {
        position: glam::DVec2,
    },
}

impl CameraControl {
    pub fn new(screen_size: glam::DVec2) -> CameraControl {
        CameraControl {
            fov_y_radians: std::f64::consts::PI / 2.0,
            z_near: 0.001,
            z_far: 10.0,
            zoom: 1.0,
            screen_size,
            cursor_pos: None,
            frame: CameraFrame {
                eye: glam::dvec3(0.0, 0.0, 2.0),
                up: glam::dvec3(0.0, 1.0, 0.0),
                center: glam::dvec3(0.0, 0.0, 0.0),
            },
            input_mode: CameraInputMode::None,
        }
    }

    pub fn set_screen_size(&mut self, size: glam::DVec2) {
        self.screen_size = size;
    }

    fn handle_pan(&mut self, orig: &CameraFrame, delta_screen: glam::DVec2) {
        let delta = delta_screen / self.screen_size;
        let dir = orig.center - orig.eye;
        let right = dir.normalize().cross(orig.up);
        let dist = dir.length();

        self.frame.eye = orig.eye + dist * (-delta.x * right + delta.y * orig.up);
        self.frame.center = orig.center + dist * (-delta.x * right + delta.y * orig.up);
    }

    fn to_ndc(&self, p: glam::DVec2) -> glam::DVec2 {
        2.0 * (p / self.screen_size) - glam::dvec2(1.0, 1.0)
    }

    fn handle_tumble(&mut self, orig: &CameraFrame, from: glam::DVec2, to: glam::DVec2) {
        let delta = (to - from) / self.screen_size;
        let eye_dir = orig.eye - orig.center;
        let right = eye_dir.normalize().cross(orig.up);
        let r = glam::DQuat::from_rotation_y(-delta.x * TAU)
            * glam::DQuat::from_axis_angle(right, delta.y * TAU);
        let new_eye = orig.center + r * eye_dir;
        let new_up = r * orig.up;
        self.frame.eye = new_eye;
        self.frame.up = new_up;
    }

    pub fn handle_input(&mut self, input: &CameraControlInput) {
        match *input {
            CameraControlInput::MouseInput { button, pressed } => {
                match button {
                    CameraControlMouseButton::Middle => {
                        if let Some(pos) = self.cursor_pos {
                            match self.input_mode {
                                CameraInputMode::None | CameraInputMode::Pan { .. } if pressed => {
                                    self.input_mode = CameraInputMode::Pan {
                                        anchor_screen: pos,
                                        orig_frame: self.frame,
                                    }
                                }
                                CameraInputMode::Pan {
                                    orig_frame,
                                    anchor_screen,
                                } if !pressed => {
                                    self.handle_pan(&orig_frame, pos - anchor_screen);
                                    self.input_mode = CameraInputMode::None;
                                }
                                _ => {}
                            }
                        }
                    }
                    CameraControlMouseButton::Left => {
                        if let Some(pos) = self.cursor_pos {
                            match self.input_mode {
                                CameraInputMode::None | CameraInputMode::Tumble { .. }
                                    if pressed =>
                                {
                                    self.input_mode = CameraInputMode::Tumble {
                                        anchor_screen: pos,
                                        orig_frame: self.frame,
                                    }
                                }
                                CameraInputMode::Tumble {
                                    orig_frame,
                                    anchor_screen,
                                } if !pressed => {
                                    self.handle_tumble(&orig_frame, anchor_screen, pos);
                                    self.input_mode = CameraInputMode::None;
                                }
                                _ => {}
                            }
                        }
                    }
                    _ => {
                        // TODO
                    }
                }
            }
            CameraControlInput::CursorMoved { position } => {
                self.cursor_pos = Some(position);
                match self.input_mode {
                    CameraInputMode::Tumble {
                        orig_frame,
                        anchor_screen,
                    } => {
                        self.handle_tumble(&orig_frame, anchor_screen, position);
                    }
                    CameraInputMode::Pan {
                        orig_frame,
                        anchor_screen,
                    } => {
                        self.handle_pan(&orig_frame, position - anchor_screen);
                    }
                    _ => {}
                }
            }
        }
    }

    /// Centers the camera on the given axis-aligned bounding box.
    /// Orbit angles are reset.
    pub fn center_on_bounds(&mut self, bounds: &BoundingBox, fov_y_radians: f64) {
        let size = bounds.size().max_element() as f64;
        let new_center: glam::DVec3 = bounds.center().as_f64();
        let cam_dist = (0.5 * size) / f64::tan(0.5 * fov_y_radians);

        let new_front = glam::dvec3(0.0, 0.0, -1.0).normalize();
        let new_eye = new_center + (-new_front * cam_dist);

        let new_right = new_front.cross(self.frame.up);
        let new_up = new_right.cross(new_front);

        self.frame.center = new_center;
        self.frame.eye = new_eye;
        self.frame.up = new_up;

        self.z_near = 0.1 * cam_dist;
        self.z_far = 10.0 * cam_dist;
        self.fov_y_radians = fov_y_radians;

        eprintln!(
            "center_on_bounds: eye={}, center={}, z_near={}, z_far={}",
            self.frame.eye, self.frame.center, self.z_near, self.z_far
        );
    }

    /// Returns the look-at matrix
    fn get_look_at(&self) -> glam::Mat4 {
        glam::Mat4::look_at_rh(
            self.frame.eye.as_f32(),
            self.frame.center.as_f32(),
            self.frame.up.as_f32(),
        )
    }

    /// Returns a `Camera` for the current viewpoint.
    pub fn camera(&self) -> Camera {
        let aspect_ratio = self.screen_size.x / self.screen_size.y;
        Camera {
            frustum: Default::default(),
            view: self.get_look_at(),
            projection: glam::Mat4::perspective_rh(
                self.fov_y_radians as f32,
                aspect_ratio as f32,
                self.z_near as f32,
                self.z_far as f32,
            ),
        }
    }
}
