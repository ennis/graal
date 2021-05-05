use crate::{
    typedesc::{PrimitiveType, TypeDesc},
    BufferData,
};
use ash::vk;
use std::{marker::PhantomData, mem};

/// Describes the type of indices contained in an index buffer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum IndexFormat {
    /// 16-bit unsigned integer indices
    U16,
    /// 32-bit unsigned integer indices
    U32,
}

/// Description of a vertex attribute within a vertex layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VertexAttribute {
    pub format: vk::Format,
    pub offset: u32,
}

/// Trait implemented by types that represent vertex data in a vertex buffer.
pub unsafe trait VertexData: BufferData {
    const ATTRIBUTES: &'static [VertexAttribute];
}

/// Trait implemented by types that can serve as indices.
pub unsafe trait IndexData: BufferData {
    /// Index type.
    const FORMAT: IndexFormat;
}

/// Trait implemented by types that can serve as a vertex attribute.
pub unsafe trait VertexAttributeType {
    /// The equivalent type descriptor (the type seen by the shader).
    const EQUIVALENT_TYPE: TypeDesc<'static>;
    /// Returns the corresponding data format (the layout of the data in memory).
    const FORMAT: vk::Format;
}

/// Wrapper type for normalized integer attributes.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Norm<T>(pub T);

impl From<f32> for Norm<u8> {
    fn from(v: f32) -> Self {
        Norm((v * u8::MAX as f32) as u8)
    }
}

impl From<f64> for Norm<u8> {
    fn from(v: f64) -> Self {
        Norm((v * u8::MAX as f64) as u8)
    }
}

impl From<f32> for Norm<u16> {
    fn from(v: f32) -> Self {
        Norm((v * u16::MAX as f32) as u16)
    }
}

impl From<f64> for Norm<u16> {
    fn from(v: f64) -> Self {
        Norm((v * u16::MAX as f64) as u16)
    }
}

// Vertex attribute types --------------------------------------------------------------------------
/*macro_rules! impl_attrib_type {
    ($t:ty, $equiv:expr, $fmt:ident) => {
        unsafe impl VertexAttributeType for $t {
            const EQUIVALENT_TYPE: TypeDesc<'static> = $equiv;
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}*/

macro_rules! impl_attrib_prim_type {
    ($t:ty, $prim:ident, $fmt:ident) => {
        unsafe impl VertexAttributeType for $t {
            const EQUIVALENT_TYPE: TypeDesc<'static> = TypeDesc::Primitive(PrimitiveType::$prim);
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

macro_rules! impl_attrib_vector_type {
    ([$t:ty; $len:expr], $prim:ident, $fmt:ident) => {
        unsafe impl VertexAttributeType for [$t; $len] {
            const EQUIVALENT_TYPE: TypeDesc<'static> = TypeDesc::Vector {
                elem_ty: PrimitiveType::$prim,
                len: $len,
            };
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

// F32
impl_attrib_prim_type!(f32, Float, R32_SFLOAT);
impl_attrib_vector_type!([f32; 2], Float, R32G32_SFLOAT);
impl_attrib_vector_type!([f32; 3], Float, R32G32B32_SFLOAT);
impl_attrib_vector_type!([f32; 4], Float, R32G32B32A32_SFLOAT);

// U32
impl_attrib_prim_type!(u32, UnsignedInt, R32_UINT);
impl_attrib_vector_type!([u32; 2], UnsignedInt, R32G32_UINT);
impl_attrib_vector_type!([u32; 3], UnsignedInt, R32G32B32_UINT);
impl_attrib_vector_type!([u32; 4], UnsignedInt, R32G32B32A32_UINT);

impl_attrib_prim_type!(i32, Int, R32_SINT);
impl_attrib_vector_type!([i32; 2], Int, R32G32_SINT);
impl_attrib_vector_type!([i32; 3], Int, R32G32B32_SINT);
impl_attrib_vector_type!([i32; 4], Int, R32G32B32A32_SINT);

// U16
impl_attrib_prim_type!(u16, UnsignedInt, R16_UINT);
impl_attrib_vector_type!([u16; 2], UnsignedInt, R16G16_UINT);
impl_attrib_vector_type!([u16; 3], UnsignedInt, R16G16B16_UINT);
impl_attrib_vector_type!([u16; 4], UnsignedInt, R16G16B16A16_UINT);

impl_attrib_prim_type!(i16, Int, R16_SINT);
impl_attrib_vector_type!([i16; 2], Int, R16G16_SINT);
impl_attrib_vector_type!([i16; 3], Int, R16G16B16_SINT);
impl_attrib_vector_type!([i16; 4], Int, R16G16B16A16_SINT);

// UNORM16
impl_attrib_prim_type!(Norm<u16>, Float, R16_UNORM);
impl_attrib_vector_type!([Norm<u16>; 2], Float, R16G16_UNORM);
impl_attrib_vector_type!([Norm<u16>; 3], Float, R16G16B16_UNORM);
impl_attrib_vector_type!([Norm<u16>; 4], Float, R16G16B16A16_UNORM);

// SNORM16
impl_attrib_prim_type!(Norm<i16>, Float, R16_SNORM);
impl_attrib_vector_type!([Norm<i16>; 2], Float, R16G16_SNORM);
impl_attrib_vector_type!([Norm<i16>; 3], Float, R16G16B16_SNORM);
impl_attrib_vector_type!([Norm<i16>; 4], Float, R16G16B16A16_SNORM);

// U8
impl_attrib_prim_type!(u8, UnsignedInt, R8_UINT);
impl_attrib_vector_type!([u8; 2], UnsignedInt, R8G8_UINT);
impl_attrib_vector_type!([u8; 3], UnsignedInt, R8G8B8_UINT);
impl_attrib_vector_type!([u8; 4], UnsignedInt, R8G8B8A8_UINT);

impl_attrib_prim_type!(i8, Int, R8_SINT);
impl_attrib_vector_type!([i8; 2], Int, R8G8_SINT);
impl_attrib_vector_type!([i8; 3], Int, R8G8B8_SINT);
impl_attrib_vector_type!([i8; 4], Int, R8G8B8A8_SINT);

// Vertex types from glam --------------------------------------------------------------------------

#[cfg(feature = "graal-glam")]
impl_attrib_type!(
    glam::Vec2,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 2
    },
    R32G32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_attrib_type!(
    glam::Vec3,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 3
    },
    R32G32B32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_attrib_type!(
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
        unsafe impl IndexData for $t {
            const FORMAT: IndexFormat = IndexFormat::$fmt;
        }
    };
}

impl_index_data!(u16, U16);
impl_index_data!(u32, U32);

// --------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferView<T: VertexData> {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub _phantom: PhantomData<*const T>,
}

pub trait VertexBindingInterface {
    const ATTRIBUTES: &'static [VertexAttribute];
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
}

pub mod vertex_macro_helpers {
    use crate::{vk, VertexAttribute};

    pub const fn append_attributes<const N: usize>(
        head: &'static [vk::VertexInputAttributeDescription],
        binding: u32,
        base_location: u32,
        tail: &'static [VertexAttribute],
    ) -> [vk::VertexInputAttributeDescription; N] {
        const NULL_ATTR: vk::VertexInputAttributeDescription =
            vk::VertexInputAttributeDescription {
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
