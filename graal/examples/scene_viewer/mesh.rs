use crate::aabb::AABB;
use glam::Vec3;
use graal::{Buffer, Vertex};

/// Represents a vertex with position, normal and tangent attributes.
#[derive(Copy, Clone, Debug, Vertex)]
#[repr(C)]
pub struct Vertex3D {
    pub position: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
}

/// A buffer containing vertices.
#[derive(Copy, Clone, Debug)]
pub struct MeshData {
    /// GPU vertex buffer containing vertex attributes.
    pub vertex_buffer: Buffer<Vertex3D>,

    /// GPU index buffer containing vertex indices.
    pub index_buffer: Buffer<u32>,

    /// Number of vertices.
    pub vertex_count: usize,

    /// Number of indices.
    pub index_count: usize,

    /// Bounds of the vertex data.
    pub bounds: AABB,
}
