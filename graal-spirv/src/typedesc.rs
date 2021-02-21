//--------------------------------------------------------------------------------------------------
use crate::spv;

/// Primitive SPIR-V data types.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PrimitiveType {
    /// 32-bit signed integer
    Int,
    /// 32-bit unsigned integer
    UnsignedInt,
    /// 16-bit half float (unused)
    Half,
    /// 32-bit floating-point value
    Float,
    /// 64-bit floating-point value
    Double,
    /// Boolean.
    /// Cannot be used with externally-visible storage classes.
    Bool,
}

/// SPIR-V image type
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ImageType<'a> {
    pub sampled_ty: &'a TypeDesc<'a>,
    pub format: spv::ImageFormat,
    pub dim: spv::Dim,
    pub arrayed: bool,
    pub ms: bool,
    pub depth: Option<bool>,
    pub sampled: Option<bool>,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum MatrixLayout {
    RowMajor,
    ColumnMajor,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ObjectOrMemberInfo {
    pub no_perspective: bool,
    pub builtin: bool,
    pub uniform: bool,
}

impl Default for ObjectOrMemberInfo {
    fn default() -> Self {
        ObjectOrMemberInfo {
            no_perspective: false,
            builtin: false,
            uniform: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructField<'a> {
    /// The type of the field
    pub ty: &'a TypeDesc<'a>,
    /// Decorations attached to this field.
    pub decorations: &'a [(spv::Decoration, &'a [u32])],
    /// Matrix layout (RowMajor or ColMajor decorations).
    pub matrix_layout: Option<MatrixLayout>,
    pub matrix_stride: Option<u32>,
    pub offset: Option<u32>,
    /// Additional information
    pub member_info: ObjectOrMemberInfo,
}

/// SPIR-V variable information.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Variable<'a> {
    pub id: u32,
    /// Type of the variable
    pub ty: &'a TypeDesc<'a>,
    /// Decorations attached to the variable
    pub decorations: &'a [(spv::Decoration, &'a [u32])],
    /// Storage class
    pub storage_class: spv::StorageClass,
    pub descriptor_set: Option<u32>,
    pub binding: Option<u32>,
    pub location: Option<u32>,
    /// Additional information
    pub info: ObjectOrMemberInfo,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum StructLayout {
    GLSLShared,
    GLSLPacked,
    CPacked,
}

/// Declaration of a SPIR-V structure type.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct StructType<'a> {
    /// Fields.
    pub fields: &'a [StructField<'a>],
    /// Decorations attached to the type
    pub decorations: &'a [(spv::Decoration, &'a [u32])],
    /// Whether the struct has a `Block` decoration
    pub block: bool,
    /// Whether the struct has a `BufferBlock` decoration
    pub buffer_block: bool,
    ///
    pub struct_layout: Option<StructLayout>,
}

/// Describes a data type used inside a SPIR-V shader
/// (e.g. the type of a uniform, or the type of vertex attributes as seen by the shader).
///
/// TypeDescs are slightly different from Formats:
/// the latter describes the precise bit layout, packing, numeric format, and interpretation
/// of individual data elements, while the former describes unpacked data as seen inside shaders.
// TODO move all types related to type descriptions into a separate module
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum TypeDesc<'a> {
    /// Primitive type.
    Primitive(PrimitiveType),
    /// Array type. (typedesc + length + stride)
    Array {
        elem_ty: &'a TypeDesc<'a>,
        len: usize,
    },
    /// Vector type (ty,size).
    Vector {
        elem_ty: PrimitiveType,
        len: u8,
    },
    /// Matrix type (ty,rows,cols).
    Matrix {
        elem_ty: PrimitiveType,
        rows: u8,
        columns: u8,
    },
    /// Structure type (array of (offset, type) tuples).
    Struct(StructType<'a>),
    /// Image type.
    Image(ImageType<'a>),
    /// Combination of an image and sampling information.
    SampledImage(ImageType<'a>),
    Void,
    /// Pointer to data.
    Pointer(&'a TypeDesc<'a>),
    /// Sampler
    Sampler,
    Unknown,
}

impl<'a> TypeDesc<'a> {
    /// The array element type, if this TypeDesc describes an array type.
    pub fn element_type(&self) -> Option<&'a TypeDesc<'a>> {
        match self {
            TypeDesc::Array { elem_ty, .. } => Some(*elem_ty),
            TypeDesc::Pointer(elem_ty) => Some(*elem_ty),
            _ => None,
        }
    }

    /// The type of the pointed-to element, if this TypeDesc describes a pointer.
    pub fn pointee_type(&self) -> Option<&'a TypeDesc<'a>> {
        match self {
            TypeDesc::Pointer(elem_ty) => Some(*elem_ty),
            _ => None,
        }
    }
}
