//! SPIR-V parsing and manipulation utilities.
pub mod inst;
mod layout;

use std::{error, fmt};

use crate::inst::{
    decode_raw_instruction, DecodedInstruction, IDecorate, IMemberDecorate, ITypeArray, ITypeBool,
    ITypeFloat, ITypeImage, ITypeInt, ITypeMatrix, ITypeOpaque, ITypePointer, ITypeRuntimeArray,
    ITypeSampledImage, ITypeSampler, ITypeStruct, ITypeVector, ITypeVoid, IVariable, Instruction,
    RawInstruction,
};
pub use crate::layout::{ArrayLayout, FieldsLayout, InnerLayout, Layout};
pub use spirv_headers as spv;
use std::collections::HashMap;

/// An arena allocator used to store parsed SPIR-V structures.
#[derive(Debug)]
pub struct Arena(bumpalo::Bump);

impl Arena {
    pub fn new() -> Arena {
        Arena(bumpalo::Bump::new())
    }
}

//--------------------------------------------------------------------------------------------------

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

pub enum ImageSamplingType {
    Unknown,
    Sampled,
    NotSampled,
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

/*/// Variable decoration
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Decoration {
    Block,
    BufferBlock,
    Constant,
    Location(u32),
    Index(u32),
    Binding(u32),
    DescriptorSet(u32),
    Uniform,
    Other(spv::Decoration),
}*/

/// Errors that can occur during parsing of SPIR-V modules.
#[derive(Debug, Clone)]
pub enum ParseError {
    MissingHeader,
    WrongHeader,
    IncompleteInstruction,
    UnknownConstant(&'static str, u32),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f, "SPIR-V parse error")
    }
}

impl error::Error for ParseError {}

/// Decodes a SPIR-V stream into raw (unparsed) instructions.
fn raw_inst_iter<'a>(module: &'a [u32]) -> impl Iterator<Item = (usize, RawInstruction<'a>)> + 'a {
    struct RawInstIter<'m> {
        i: &'m [u32],
        ptr: usize,
    }

    impl<'m> Iterator for RawInstIter<'m> {
        type Item = (usize, RawInstruction<'m>);

        fn next(&mut self) -> Option<(usize, RawInstruction<'m>)> {
            if self.i.len() >= 1 {
                let (inst, rest) = decode_raw_instruction(self.i).unwrap();
                let ptr = self.ptr;
                self.i = rest;
                self.ptr += inst.word_count as usize;
                Some((ptr, inst))
            } else {
                None
            }
        }
    }

    // 5 is beginning of instruction stream
    RawInstIter {
        i: &module[5..],
        ptr: 5,
    }
}

/// Returns an iterator of all instructions of type `T` in the given SPIR-V stream.
fn inst_by_type_iter<'a, T: DecodedInstruction<'a>>(
    module: &'a [u32],
) -> impl Iterator<Item = (usize, T)> + 'a {
    raw_inst_iter(module).filter_map(|(iptr, inst)| {
        if inst.opcode == T::OPCODE as u16 {
            Some((iptr, T::decode(inst.operands).into()))
        } else {
            None
        }
    })
}

/// Decodes all instructions in the given SPIR-V stream.
fn inst_iter<'a>(module: &'a [u32]) -> impl Iterator<Item = (usize, Instruction)> + 'a {
    raw_inst_iter(module).map(|(iptr, inst)| (iptr, inst.decode()))
}

fn decode_raw_inst_at(module: &[u32], iptr: usize) -> Result<RawInstruction, ParseError> {
    decode_raw_instruction(&module[iptr..]).map(|(inst, _)| inst)
}

fn next_inst<'a>(module: &'a [u32], ptr: usize) -> Result<usize, ParseError> {
    Ok(ptr + decode_raw_inst_at(module, ptr)?.word_count as usize)
}

/// Returns an iterator of all decorations on id.
fn decorations_iter<'a>(
    module: &'a [u32],
    id: u32,
) -> impl Iterator<Item = (usize, IDecorate)> + 'a {
    inst_by_type_iter::<IDecorate>(module).filter(move |(_, d)| d.target_id == id)
}

/// Returns an iterator of all decorations on a member of a struct type.
fn member_decorations_iter<'a>(
    module: &'a [u32],
    struct_type_id: u32,
    member: u32,
) -> impl Iterator<Item = (usize, IMemberDecorate)> + 'a {
    inst_by_type_iter::<IMemberDecorate>(module)
        .filter(move |(_, d)| d.target_id == struct_type_id && d.member == member)
}

fn parse_types<'a>(arena: &'a Arena, module: &'a [u32]) -> HashMap<u32, &'a TypeDesc<'a>> {
    // build a map from id to type
    let mut tymap = HashMap::<u32, &'a TypeDesc<'a>>::new();

    // can process types in order, since the spec specifies that:
    // "Types are built bottom up: A parameterizing operand in a type must be defined before being used."
    inst_iter(module).for_each(|(_, inst)| {
        match inst {
            Instruction::TypeVoid(ITypeVoid { result_id }) => {
                tymap.insert(result_id, arena.0.alloc(TypeDesc::Void));
            }
            Instruction::TypeBool(ITypeBool { result_id }) => {
                tymap.insert(
                    result_id,
                    arena.0.alloc(TypeDesc::Primitive(PrimitiveType::Bool)),
                );
            }
            Instruction::TypeSampler(ITypeSampler{ result_id }) => {
                tymap.insert(result_id, arena.0.alloc(TypeDesc::Sampler));
            }
            Instruction::TypeInt(ITypeInt {
                result_id,
                width,
                signedness,
            }) => {
                assert_eq!(width, 32, "unsupported bit width");
                match signedness {
                    true => tymap.insert(
                        result_id,
                        arena.0.alloc(TypeDesc::Primitive(PrimitiveType::Int)),
                    ),
                    false => tymap.insert(
                        result_id,
                        arena
                            .0
                            .alloc(TypeDesc::Primitive(PrimitiveType::UnsignedInt)),
                    ),
                };
            }
            Instruction::TypeFloat(ITypeFloat { result_id, width }) => {
                assert_eq!(width, 32, "unsupported bit width");
                tymap.insert(
                    result_id,
                    arena.0.alloc(TypeDesc::Primitive(PrimitiveType::Float)),
                );
            }
            Instruction::TypeVector(ITypeVector {
                result_id,
                component_id,
                count,
            }) => {
                let elem_ty = tymap[&component_id];
                if let &TypeDesc::Primitive(elem_ty) = &*elem_ty {
                    tymap.insert(
                        result_id,
                        arena.0.alloc(TypeDesc::Vector {
                            elem_ty,
                            len: count as u8,
                        }),
                    );
                } else {
                    panic!("expected primitive type");
                }
            }
            Instruction::TypeMatrix(ITypeMatrix {
                result_id,
                column_type_id,
                column_count,
            }) => {
                let colty = tymap[&column_type_id];
                if let &TypeDesc::Vector { elem_ty, len } = colty {
                    tymap.insert(
                        result_id,
                        arena.0.alloc(TypeDesc::Matrix {
                            elem_ty,
                            rows: len,
                            columns: column_count as u8,
                        }),
                    );
                } else {
                    panic!("expected vector type");
                }
            }
            Instruction::TypeImage(ITypeImage {
                result_id,
                sampled_type_id,
                dim,
                depth,
                arrayed,
                ms,
                sampled,
                format,
                access: _,
            }) => {
                let sampled_ty = tymap[&sampled_type_id];
                tymap.insert(
                    result_id,
                    arena.0.alloc(TypeDesc::Image(ImageType {
                        sampled_ty,
                        format,
                        dim,
                        ms,
                        arrayed,
                        depth,
                        sampled,
                    })),
                );
            }
            Instruction::TypeSampledImage(ITypeSampledImage {
                result_id,
                image_type_id,
            }) => {
                let image_ty = tymap[&image_type_id];
                if let &TypeDesc::Image(ref img_ty) = image_ty {
                    tymap.insert(result_id, arena.0.alloc(TypeDesc::SampledImage(*img_ty)));
                } else {
                    panic!("expected image type")
                };
            }
            Instruction::TypeArray(ITypeArray {
                result_id,
                type_id,
                length_id: _,
            }) => {
                let elem_ty = tymap[&type_id];
                tymap.insert(
                    result_id,
                    arena.0.alloc(TypeDesc::Array { elem_ty, len: 0 }),
                );
            }
            Instruction::TypeRuntimeArray(ITypeRuntimeArray { result_id, type_id }) => {
                let elem_ty = tymap[&type_id];
                tymap.insert(
                    result_id,
                    arena.0.alloc(TypeDesc::Array { elem_ty, len: 0 }),
                );
            }
            Instruction::TypeStruct(ITypeStruct {
                result_id,
                member_types,
            }) => {
                let fields = arena
                    .0
                    .alloc_slice_fill_iter(member_types.iter().enumerate().map(
                        |(member, &tyid)| {
                            parse_struct_member(
                                arena,
                                module,
                                &tymap,
                                result_id,
                                member as u32,
                                tyid
                            )
                        },
                    ));

                let mut struct_type = StructType {
                    fields,
                    decorations: &[],
                    block: false,
                    buffer_block: false,
                    struct_layout: None,
                };

                let mut decorations = Vec::new();
                for (_, d) in decorations_iter(module, result_id) {
                    match d.decoration {
                        spv::Decoration::Block => struct_type.block = true,
                        spv::Decoration::BufferBlock => struct_type.buffer_block = true,
                        other => {
                            // TODO
                        }
                    }
                    decorations.push((d.decoration, d.params));
                }

                struct_type.decorations = arena.0.alloc_slice_fill_iter(decorations);

                tymap.insert(result_id, arena.0.alloc(TypeDesc::Struct(struct_type)));
            }
            Instruction::TypeOpaque(ITypeOpaque {
                result_id: _,
                name: _,
            }) => unimplemented!(),
            Instruction::TypePointer(ITypePointer {
                result_id,
                storage_class: _,
                type_id,
            }) => {
                let ty = tymap[&type_id];
                tymap.insert(result_id, arena.0.alloc(TypeDesc::Pointer(ty)));
            }
            _ => {}
        };
    });

    tymap
}

fn parse_object_or_member_decoration(
    decoration: spv::Decoration,
    params: &[u32],
    out_info: &mut ObjectOrMemberInfo,
) {
    match decoration {
        spv::Decoration::NoPerspective => out_info.no_perspective = true,
        spv::Decoration::BuiltIn => out_info.builtin = true,
        _ => {}
    }
}

fn parse_struct_member<'a>(
    arena: &'a Arena,
    module: &'a [u32],
    tymap: &HashMap<u32, &'a TypeDesc<'a>>,
    struct_type_id: u32,
    member: u32,
    member_type_id: u32,
) -> StructField<'a> {
    let mut field = StructField {
        ty: tymap[&member_type_id],
        decorations: &[],
        matrix_layout: None,
        matrix_stride: None,
        offset: None,
        member_info: ObjectOrMemberInfo {
            no_perspective: false,
            builtin: false,
            uniform: false,
        },
    };

    let mut decorations = Vec::new();

    for (_, d) in member_decorations_iter(module, struct_type_id, member) {
        match d.decoration {
            spv::Decoration::MatrixStride => field.matrix_stride = Some(d.params[0]),
            spv::Decoration::RowMajor => field.matrix_layout = Some(MatrixLayout::RowMajor),
            spv::Decoration::ColMajor => field.matrix_layout = Some(MatrixLayout::ColumnMajor),
            spv::Decoration::Offset => field.offset = Some(d.params[0]),
            other => parse_object_or_member_decoration(other, d.params, &mut field.member_info),
        }
        decorations.push((d.decoration, d.params));
    }

    field.decorations = arena.0.alloc_slice_fill_iter(decorations);
    field
}

fn parse_variables<'a>(
    arena: &'a Arena,
    module: &'a [u32],
    tymap: &HashMap<u32, &'a TypeDesc<'a>>,
) -> &'a [Variable<'a>] {
    let vars: Vec<_> = inst_by_type_iter::<IVariable>(module)
        .map(|(_iptr, v)| {
            let mut variable = Variable {
                id: v.result_id,
                ty: tymap[&v.result_type_id],
                decorations: &[],
                storage_class: v.storage_class,
                descriptor_set: None,
                binding: None,
                location: None,
                info: Default::default(),
            };

            let mut decorations = Vec::new();
            for (_, d) in decorations_iter(module, v.result_id) {
                match d.decoration {
                    spv::Decoration::DescriptorSet => variable.descriptor_set = Some(d.params[0]),
                    spv::Decoration::Binding => variable.binding = Some(d.params[0]),
                    spv::Decoration::Location => variable.location = Some(d.params[0]),
                    other => parse_object_or_member_decoration(other, d.params, &mut variable.info),
                }
                decorations.push((d.decoration, d.params));
            }
            variable.decorations = arena.0.alloc_slice_fill_iter(decorations);
            variable
        })
        .collect();

    arena.0.alloc_slice_fill_iter(vars)
}

/// A SPIR-V module.
#[derive(Debug, Clone)]
pub struct Module<'a> {
    arena: &'a Arena,
    pub data: &'a [u32],
    _tymap: HashMap<u32, &'a TypeDesc<'a>>,
    pub variables: &'a [Variable<'a>],
    pub version: (u8, u8),
    pub bound: u32,
}

impl<'a> Module<'a> {
    /// Parses a SPIR-V module from a slice of bytes, possibly converting it to the native byte order if necessary.
    pub fn from_bytes(arena: &'a Arena, data: &[u8]) -> Result<Module<'a>, ParseError> {
        if data.len() < 20 {
            return Err(ParseError::MissingHeader);
        }

        // we need to determine whether we are in big endian order or little endian order depending
        // on the magic number at the start of the file
        let data = if data[0] == 0x07 && data[1] == 0x23 && data[2] == 0x02 && data[3] == 0x03 {
            // big endian
            arena.0.alloc_slice_fill_iter(data.chunks(4).map(|c| {
                ((c[0] as u32) << 24) | ((c[1] as u32) << 16) | ((c[2] as u32) << 8) | c[3] as u32
            }))
        } else if data[3] == 0x07 && data[2] == 0x23 && data[1] == 0x02 && data[0] == 0x03 {
            // little endian
            arena.0.alloc_slice_fill_iter(data.chunks(4).map(|c| {
                ((c[3] as u32) << 24) | ((c[2] as u32) << 16) | ((c[1] as u32) << 8) | c[0] as u32
            }))
        } else {
            return Err(ParseError::MissingHeader);
        };

        Self::from_words(arena, data)
    }

    /// Parses a SPIR-V module from machine words.
    pub fn from_words(arena: &'a Arena, module: &'a [u32]) -> Result<Module<'a>, ParseError> {
        if module.len() < 5 {
            return Err(ParseError::MissingHeader);
        }

        if module[0] != 0x07230203 {
            return Err(ParseError::WrongHeader);
        }

        let version = (
            ((module[1] & 0x00ff0000) >> 16) as u8,
            ((module[1] & 0x0000ff00) >> 8) as u8,
        );

        let tymap = parse_types(arena, module);
        let variables = parse_variables(arena, module, &tymap);

        Ok(Module {
            arena,
            data: module,
            _tymap: tymap,
            variables,
            version,
            bound: module[3],
        })
    }
}
