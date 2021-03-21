use crate::bounding_box::BoundingBox;

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

/// A camera controller that generates `Camera` instances.
///
/// TODO describe parameters
#[derive(Clone, Debug)]
pub struct CameraControl {
    fov_y_radians: f32,
    aspect_ratio: f32,
    z_near: f32,
    z_far: f32,
    zoom: f32,
    orbit_radius: f32,
    theta: f32,
    phi: f32,
    target: glam::Vec3A,
}

impl Default for CameraControl {
    fn default() -> CameraControl {
        CameraControl {
            fov_y_radians: f32::consts::PI / 2.0,
            aspect_ratio: 1.0,
            z_near: 0.001,
            z_far: 10.0,
            zoom: 1.0,
            orbit_radius: 1.0,
            theta: 0.0,
            phi: f32::consts::PI / 2.0,
            target: glam::Vec3A::new(0.0, 0.0, 0.0),
        }
    }
}

impl CameraControl {
    pub fn set_aspect_ratio(&mut self, aspect_ratio: f32) {
        self.aspect_ratio = aspect_ratio;
    }

    /// Centers the camera on the given axis-aligned bounding box.
    /// Orbit angles are not reset.
    pub fn center_on_bounds(&mut self, bounds: &BoundingBox, fov_y_radians: f32) {
        let size = bounds.size().max_element();
        let center = bounds.center();
        let cam_dist = (0.5 * size) / f32::tan(0.5 * fovy);

        self.orbit_radius = cam_dist;
        self.target = center;
        self.z_near = 0.1 * cam_dist;
        self.z_far = 10.0 * cam_dist;
        self.fov_y_radians = fov_y_radians;

        //debug!("Center on AABB: {:?} -> {:?}", &aabb, &self);
    }

    fn orbit_to_cartesian(&self) -> glam::Vec3A {
        glam::Vec3A::new(
            self.orbit_radius * f32::sin(self.theta) * f32::sin(self.phi),
            self.orbit_radius * f32::cos(self.phi),
            self.orbit_radius * f32::cos(self.theta) * f32::sin(self.phi),
        )
    }

    /// Returns the look-at matrix
    fn get_look_at(&self) -> glam::Mat4 {
        let dir = self.orbit_to_cartesian();
        glam::Mat4::look_at_rh((self.target + dir).into(), self.target.into(), glam::Vec3::Y)
    }

    /// Returns a `Camera` for the current viewpoint.
    pub fn camera(&self) -> Camera {
        Camera {
            frustum: Default::default(),
            view: self.get_look_at(),
            projection: glam::Mat4::perspective_rh(
                self.fov_y_radians,
                self.aspect_ratio,
                self.z_near,
                self.z_far,
            ),
        }
    }
}