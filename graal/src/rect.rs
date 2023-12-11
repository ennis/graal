#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect2D {
    pub min: Point2D,
    pub max: Point2D,
}

impl Rect2D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect3D {
    pub min: Point3D,
    pub max: Point3D,
}

impl Rect3D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }
    pub const fn depth(&self) -> u32 {
        (self.max.z - self.min.z) as u32
    }
}
