use crate::{PrimitiveType, TypeDesc, Arena, StructField, StructType};
use std::iter;

fn round_up(value: usize, multiple: usize) -> usize {
    if multiple == 0 {
        return value;
    }
    let remainder = value % multiple;
    if remainder == 0 {
        return value;
    }
    value + multiple - remainder
}

/// Contains information about the layout of a SPIR-V type.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Layout<'a> {
    /// Alignment
    pub align: usize,
    /// Byte size
    pub size: usize,
    /// Layout of the contents of the type, for array or structs
    pub inner: InnerLayout<'a>,
}

impl<'a> Layout<'a> {
    /// Creates a new layout for a scalar element with the specified size and alignment.
    pub const fn with_size_align(size: usize, align: usize) -> Layout<'a> {
        Layout {
            align,
            size,
            inner: InnerLayout::None,
        }
    }
}

/// Layout of the fields of a SPIR-V struct type.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct FieldsLayout<'a> {
    /// Offsets of each field.
    pub offsets: &'a [usize],
    /// Individual layout information of each field.
    pub layouts: &'a [&'a Layout<'a>],
}

/// Layout of the array elements of a SPIR-V array type.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct ArrayLayout<'a> {
    /// Layout of individual array elements.
    pub elem_layout: &'a Layout<'a>,
    /// Number of bytes between consecutive array elements.
    pub stride: usize,
}

/// Layout information for SPIR-V arrays or structs.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum InnerLayout<'a> {
    None,
    Array(ArrayLayout<'a>),
    Struct(FieldsLayout<'a>),
}

fn std140_array_layout<'a>(
    arena: &'a Arena,
    elem_ty: &TypeDesc,
    arraylen: usize,
) -> &'a Layout<'a> {
    let elem_layout = std140_layout(arena, elem_ty);
    // alignment = column type align rounded up to vec4 align (16 bytes)
    let base_align = round_up(elem_layout.align, 16);
    let stride = round_up(elem_layout.size, elem_layout.align);
    // total array size = num columns * stride, rounded up to the next multiple of the base alignment.
    // actually the spec says nothing about the 'size' of an element, only about the alignment
    // of the next element in the structure.
    let array_size = round_up(arraylen * stride, base_align);
    arena.0.alloc(Layout {
        align: base_align,
        size: array_size,
        inner: InnerLayout::Array(ArrayLayout {
            stride,
            elem_layout,
        }),
    })
}

fn std140_struct_layout<'a>(arena: &'a Arena, fields: &[StructField]) -> &'a Layout<'a> {
    /* If the member is a structure, the base alignment of the structure is N,
    where N is the largest base alignment value of any of its members,
    and rounded up to the base alignment of a vec4.
    The individual members of this sub-structure are then assigned offsets by applying this set of rules recursively,
    where the base offset of the first member of the sub-structure is equal to the aligned offset of the structure.
    The structure may have padding at the end;
    the base offset of the member following the sub-structure is rounded up to the next multiple of the base alignment of the structure.
    */
    // TODO: zero-sized structures?

    let layouts: Vec<_> = fields.iter().map(|&field| std140_layout(arena, field.ty)).collect();
    let layouts = arena.0.alloc_slice_fill_iter(layouts);
    let n = layouts.iter().map(|l| l.align).max().unwrap_or(0);
    if n == 0 {
        // skip, no members
        return arena.0.alloc(Layout {
            align: 0,
            size: 0,
            inner: InnerLayout::Struct(FieldsLayout {
                offsets: &[],
                layouts: &[],
            }),
        });
    }

    // round up to base alignment of vec4
    let n = round_up(n, 16);

    // compute field offsets
    let offsets = arena.0.alloc_slice_fill_copy(fields.len(), 0);
    let mut off = 0;
    for i in 0..fields.len() {
        offsets[i] = off;
        off += layouts[i].size;
    }

    // round up total size to base align
    let size = round_up(off, n);

    arena.0.alloc(Layout {
        align: n,
        size,
        inner: InnerLayout::Struct(FieldsLayout { layouts, offsets }),
    })
}

fn std140_primitive_layout(prim_ty: PrimitiveType) -> Layout<'static> {
    match prim_ty {
        PrimitiveType::Int | PrimitiveType::UnsignedInt | PrimitiveType::Float => Layout {
            size: 4,
            align: 4,
            inner: InnerLayout::None,
        },
        _ => unimplemented!(),
    }
}

fn std140_vector_layout(prim_ty: PrimitiveType, len: u8) -> Layout<'static> {
    let Layout { size: n, .. } = std140_primitive_layout(prim_ty);
    match len {
        2 => Layout {
            align: 2 * n,
            size: 2 * n,
            inner: InnerLayout::None,
        },
        3 => Layout {
            align: 4 * n,
            size: 3 * n,
            inner: InnerLayout::None,
        },
        4 => Layout {
            align: 4 * n,
            size: 4 * n,
            inner: InnerLayout::None,
        },
        _ => panic!("unsupported vector size"),
    }
}

fn std140_layout<'a>(arena: &'a Arena, ty: &TypeDesc) -> &'a Layout<'a> {
    match *ty {
        TypeDesc::Primitive(p) => arena.0.alloc(std140_primitive_layout(p)),
        TypeDesc::Vector { elem_ty, len } => arena.0.alloc(std140_vector_layout(elem_ty, len)),
        TypeDesc::Matrix {
            elem_ty,
            rows,
            columns,
        } => std140_array_layout(
            arena,
            &TypeDesc::Vector { elem_ty, len: rows },
            columns as usize,
        ),
        TypeDesc::Array { elem_ty, len } => match elem_ty {
            TypeDesc::Primitive(_) | TypeDesc::Vector { .. } | TypeDesc::Struct { .. } => {
                std140_array_layout(arena, elem_ty, len)
            }
            ty => panic!("unsupported array element type: {:?}", ty),
        },
        TypeDesc::Struct(ty) => std140_struct_layout(arena, ty.fields),
        ty => panic!("unsupported type: {:?}", ty),
    }
}

impl<'a> Layout<'a> {
    pub fn std140(arena: &'a Arena, ty: &TypeDesc) -> &'a Layout<'a> {
        std140_layout(arena, ty)
    }
}
