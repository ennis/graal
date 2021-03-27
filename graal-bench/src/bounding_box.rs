/// Axis-aligned bounding boxes
#[derive(Copy, Clone, Debug)]
pub struct BoundingBox {
    pub min: glam::Vec3A,
    pub max: glam::Vec3A,
}

impl BoundingBox {
    pub fn size(&self) -> glam::Vec3A {
        self.max - self.min
    }

    /// Reference:
    /// http://dev.theomader.com/transform-bounding-boxes/
    pub fn transform(&self, tr: &glam::Mat4) -> BoundingBox {
        let xa = tr.x_axis * self.min.x;
        let xb = tr.x_axis * self.max.x;
        let ya = tr.y_axis * self.min.y;
        let yb = tr.y_axis * self.max.y;
        let za = tr.z_axis * self.min.z;
        let zb = tr.z_axis * self.max.z;

        let min = xa.min(xb) + ya.min(yb) + za.min(zb) + tr.w_axis;
        let max = xa.max(xb) + ya.max(yb) + za.max(zb) + tr.w_axis;

        BoundingBox {
            min: min.into(),
            max: max.into(),
        }
    }

    /// Returns the center of the bounding box.
    pub fn center(&self) -> glam::Vec3A {
        0.5 * (self.min + self.max)
    }

    /*pub fn union_with(&mut self, other: &AABB>) {
        // This is a tad verbose
        self.min = Point3::from_coordinates(cw_min3(&self.min.coords, &other.min.coords));
        self.max = Point3::from_coordinates(cw_max3(&self.max.coords, &other.max.coords));
    }

    pub fn empty() -> AABB<N> {
        AABB {
            max: Point3::new(N::min_value(), N::min_value(), N::min_value()),
            min: Point3::new(N::max_value(), N::max_value(), N::max_value()),
        }
    }*/
}
