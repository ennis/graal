//! Vertex-related types
use crate::vk;
use std::marker::PhantomData;

pub use mlr_macros::Vertex;

/// Trait implemented by types that represent vertex data in a vertex buffer.
pub unsafe trait Vertex: Copy + 'static {
    const ATTRIBUTES: &'static [VertexAttributeDescriptor];
}

/// Trait implemented by types that can serve as indices.
pub unsafe trait VertexIndex: Copy + 'static {
    /// Index type.
    const FORMAT: VertexIndexFormat;
}

/// Describes the type of indices contained in an index buffer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum VertexIndexFormat {
    /// 16-bit unsigned integer indices
    U16,
    /// 32-bit unsigned integer indices
    U32,
}

/// Description of a vertex attribute within a vertex layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VertexAttributeDescriptor {
    pub format: vk::Format,
    pub offset: u32,
}

/// Trait implemented by types that can serve as a vertex attribute.
pub unsafe trait VertexAttribute {
    /// Returns the corresponding data format (the layout of the data in memory).
    const FORMAT: vk::Format;
}

/// Wrapper type for normalized integer attributes.
///
/// Helper for `normalized` in `derive(Vertex)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Norm<T>(T);

// Vertex attribute types --------------------------------------------------------------------------
macro_rules! impl_vertex_attr {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexAttribute for $t {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

/*
macro_rules! impl_vertex_attrib_ {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexAttribute for $t {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

macro_rules! impl_attrib_vector_type {
    ([$t:ty; $len:expr], $fmt:ident) => {
        unsafe impl VertexAttribute for [$t; $len] {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}*/

// F32
impl_vertex_attr!(f32, R32_SFLOAT);
impl_vertex_attr!([f32; 2], R32G32_SFLOAT);
impl_vertex_attr!([f32; 3], R32G32B32_SFLOAT);
impl_vertex_attr!([f32; 4], R32G32B32A32_SFLOAT);

// U32
impl_vertex_attr!(u32, R32_UINT);
impl_vertex_attr!([u32; 2], R32G32_UINT);
impl_vertex_attr!([u32; 3], R32G32B32_UINT);
impl_vertex_attr!([u32; 4], R32G32B32A32_UINT);

impl_vertex_attr!(i32, R32_SINT);
impl_vertex_attr!([i32; 2], R32G32_SINT);
impl_vertex_attr!([i32; 3], R32G32B32_SINT);
impl_vertex_attr!([i32; 4], R32G32B32A32_SINT);

// U16
impl_vertex_attr!(u16, R16_UINT);
impl_vertex_attr!([u16; 2], R16G16_UINT);
impl_vertex_attr!([u16; 3], R16G16B16_UINT);
impl_vertex_attr!([u16; 4], R16G16B16A16_UINT);

impl_vertex_attr!(i16, R16_SINT);
impl_vertex_attr!([i16; 2], R16G16_SINT);
impl_vertex_attr!([i16; 3], R16G16B16_SINT);
impl_vertex_attr!([i16; 4], R16G16B16A16_SINT);

// UNORM16
impl_vertex_attr!(Norm<u16>, R16_UNORM);
impl_vertex_attr!(Norm<[u16; 2]>, R16G16_UNORM);
impl_vertex_attr!(Norm<[u16; 3]>, R16G16B16_UNORM);
impl_vertex_attr!(Norm<[u16; 4]>, R16G16B16A16_UNORM);

// SNORM16
impl_vertex_attr!(Norm<i16>, R16_SNORM);
impl_vertex_attr!(Norm<[i16; 2]>, R16G16_SNORM);
impl_vertex_attr!(Norm<[i16; 3]>, R16G16B16_SNORM);
impl_vertex_attr!(Norm<[i16; 4]>, R16G16B16A16_SNORM);

// U8
impl_vertex_attr!(u8, R8_UINT);
impl_vertex_attr!([u8; 2], R8G8_UINT);
impl_vertex_attr!([u8; 3], R8G8B8_UINT);
impl_vertex_attr!([u8; 4], R8G8B8A8_UINT);

impl_vertex_attr!(Norm<u8>, R8_UNORM);
impl_vertex_attr!(Norm<[u8; 2]>, R8G8_UNORM);
impl_vertex_attr!(Norm<[u8; 3]>, R8G8B8_UNORM);
impl_vertex_attr!(Norm<[u8; 4]>, R8G8B8A8_UNORM);

impl_vertex_attr!(i8, R8_SINT);
impl_vertex_attr!([i8; 2], R8G8_SINT);
impl_vertex_attr!([i8; 3], R8G8B8_SINT);
impl_vertex_attr!([i8; 4], R8G8B8A8_SINT);

// Vertex types from glam --------------------------------------------------------------------------

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec2,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 2
    },
    R32G32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec3,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 3
    },
    R32G32B32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec4,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 4
    },
    R32G32B32A32_SFLOAT
);

// Index data types --------------------------------------------------------------------------------
macro_rules! impl_index_data {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexIndex for $t {
            const FORMAT: VertexIndexFormat = VertexIndexFormat::$fmt;
        }
    };
}

impl_index_data!(u16, U16);
impl_index_data!(u32, U32);

// --------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferView<T: Vertex> {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub _phantom: PhantomData<*const T>,
}

/*
pub trait VertexBindingInterface {
    const ATTRIBUTES: &'static [VertexAttributeDescriptor];
    const STRIDE: usize;
}

impl<T: VertexData> VertexBindingInterface for VertexBufferView<T> {
    const ATTRIBUTES: &'static [VertexAttribute] = T::ATTRIBUTES;
    const STRIDE: usize = mem::size_of::<T>();
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct VertexInputBindingAttributes<'a> {
    pub base_location: u32,
    pub attributes: &'a [VertexAttribute],
}

pub trait VertexInputInterface {
    const BINDINGS: &'static [vk::VertexInputBindingDescription];
    const ATTRIBUTES: &'static [vk::VertexInputAttributeDescription];
}

/// Extension trait for VertexInputInterface
pub trait VertexInputInterfaceExt: VertexInputInterface {
    /// Helper function to get a `vk::PipelineVertexInputStateCreateInfo` from this vertex input struct.
    fn get_pipeline_vertex_input_state_create_info() -> vk::PipelineVertexInputStateCreateInfo;
}

impl<T: VertexInputInterface> VertexInputInterfaceExt for T {
    fn get_pipeline_vertex_input_state_create_info() -> vk::PipelineVertexInputStateCreateInfo {
        vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: Self::BINDINGS.len() as u32,
            p_vertex_binding_descriptions: Self::BINDINGS.as_ptr(),
            vertex_attribute_description_count: Self::ATTRIBUTES.len() as u32,
            p_vertex_attribute_descriptions: Self::ATTRIBUTES.as_ptr(),
            ..Default::default()
        }
    }
}*/
/*
pub mod vertex_macro_helpers {
    use graal::vk;
    use mlr::vertex::VertexAttribute;

    pub const fn append_attributes<const N: usize>(
        head: &'static [vk::VertexInputAttributeDescription],
        binding: u32,
        base_location: u32,
        tail: &'static [VertexAttribute],
    ) -> [vk::VertexInputAttributeDescription; N] {
        const NULL_ATTR: vk::VertexInputAttributeDescription = vk::VertexInputAttributeDescription {
            location: 0,
            binding: 0,
            format: vk::Format::UNDEFINED,
            offset: 0,
        };
        let mut result = [NULL_ATTR; N];
        let mut i = 0;
        while i < head.len() {
            result[i] = head[i];
            i += 1;
        }
        while i < N {
            let j = i - head.len();
            result[i] = vk::VertexInputAttributeDescription {
                location: base_location + j as u32,
                binding,
                format: tail[j].format,
                offset: tail[j].offset,
            };
            i += 1;
        }

        result
    }
}
*/
