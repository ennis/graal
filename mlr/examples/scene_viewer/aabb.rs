/// Axis-aligned bounding boxes
#[derive(Copy, Clone, Debug)]
pub struct AABB {
    pub min: glam::Vec3A,
    pub max: glam::Vec3A,
}

impl AABB {
    /// Returns an empty bounding box.
    pub fn new() -> AABB {
        AABB {
            min: Default::default(),
            max: Default::default(),
        }
    }

    /// Returns the size of the bounding box.
    pub fn size(&self) -> glam::Vec3A {
        self.max - self.min
    }

    /// Transforms the bounding box with the provided matrix.
    ///
    /// Reference:
    /// http://dev.theomader.com/transform-bounding-boxes/
    pub fn transform(&self, tr: &glam::Mat4) -> AABB {
        let xa = tr.x_axis * self.min.x;
        let xb = tr.x_axis * self.max.x;
        let ya = tr.y_axis * self.min.y;
        let yb = tr.y_axis * self.max.y;
        let za = tr.z_axis * self.min.z;
        let zb = tr.z_axis * self.max.z;

        let min = xa.min(xb) + ya.min(yb) + za.min(zb) + tr.w_axis;
        let max = xa.max(xb) + ya.max(yb) + za.max(zb) + tr.w_axis;

        AABB {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Returns the center of the bounding box.
    pub fn center(&self) -> glam::Vec3A {
        0.5 * (self.min + self.max)
    }

    /// Returns the union of this bounding box with another.
    pub fn union(&self, other: &AABB) -> AABB {
        AABB {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

impl Default for AABB {
    fn default() -> Self {
        AABB::new()
    }
}
