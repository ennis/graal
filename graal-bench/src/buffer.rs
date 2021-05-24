//--------------------------------------------------------------------------------------------------

/// Marker trait for data that can be uploaded to a GPU buffer
pub trait BufferData: 'static {
    type Element;
    fn len(&self) -> usize;
}

impl<T: Copy + 'static> BufferData for T {
    type Element = T;
    fn len(&self) -> usize {
        1
    }
}

impl<U: BufferData> BufferData for [U] {
    type Element = U;
    fn len(&self) -> usize {
        (&self as &[U]).len()
    }
}

/*impl<U: BufferData, const N: usize> BufferData for [U; N] {
    type Element = U;
    fn len(&self) -> usize {
        N
    }
}*/

/// Trait implemented by types that are layout-compatible with an specific
/// to GLSL/SPIR-V type.
///
/// An implementation is provided for most primitive types and arrays of primitive types.
/// Structs can derive it automatically with `#[derive(StructuredBufferData)]`
///
/// Unresolved issue: a struct may have alignment requirements
pub unsafe trait StructuredBufferData: BufferData {
    const TYPE: TypeDesc<'static>;
    const LAYOUT: Layout<'static>;
}

macro_rules! impl_structured_type {
    ($t:ty, $tydesc:expr) => {
        unsafe impl StructuredBufferData for $t {
            const TYPE: TypeDesc<'static> = $tydesc;
            const LAYOUT: Layout<'static> =
                Layout::with_size_align(std::mem::size_of::<$t>(), std::mem::align_of::<$t>());
        }
    };
}

// 32-bit-sized boolean type for use in shader interfaces
#[repr(u32)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BoolU32 {
    False = 0,
    True = 1,
}

impl Default for BoolU32 {
    fn default() -> Self {
        BoolU32::False
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec2f(pub [f32; 2]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec3f(pub [f32; 3]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Vec4f(pub [f32; 4]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Vec2i(pub [i32; 2]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Vec3i(pub [i32; 3]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct Vec4i(pub [i32; 4]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Mat2x2f(pub [[f32; 2]; 2]);

#[repr(transparent)]
#[derive(Copy, Clone, Debug, PartialEq, PartialOrd)]
pub struct Mat4x4f(pub [[f32; 4]; 4]);

unsafe impl<T: StructuredBufferData + Copy, const N: usize> StructuredBufferData for [T; N] {
    const TYPE: TypeDesc<'static> = TypeDesc::Array {
        elem_ty: &T::TYPE,
        len: N,
    };
    const LAYOUT: Layout<'static> = Layout {
        size: std::mem::size_of::<Self>(),
        align: std::mem::align_of::<Self>(),
        inner: InnerLayout::Array(ArrayLayout {
            elem_layout: &T::LAYOUT,
            stride: T::LAYOUT.size,
        }),
    };
}

impl_structured_type!(BoolU32, TypeDesc::Primitive(PrimitiveType::UnsignedInt));
impl_structured_type!(f32, TypeDesc::Primitive(PrimitiveType::Float));
impl_structured_type!(
    Vec2f,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 2
    }
);
impl_structured_type!(
    Vec3f,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 3
    }
);
impl_structured_type!(
    Vec4f,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 4
    }
);
impl_structured_type!(i32, TypeDesc::Primitive(PrimitiveType::Int));
impl_structured_type!(
    Vec2i,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Int,
        len: 2
    }
);
impl_structured_type!(
    Vec3i,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Int,
        len: 3
    }
);
impl_structured_type!(
    Vec4i,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Int,
        len: 4
    }
);
impl_structured_type!(
    Mat2x2f,
    TypeDesc::Matrix {
        elem_ty: PrimitiveType::Float,
        rows: 2,
        columns: 2
    }
);
impl_structured_type!(
    Mat4x4f,
    TypeDesc::Matrix {
        elem_ty: PrimitiveType::Float,
        rows: 4,
        columns: 4
    }
);
