use graal::StructuredBufferData;

#[repr(C)]
#[derive(StructuredBufferData, Copy, Clone)]
struct TestLayout1 {
    a: i32,
    b: i32,
}

#[repr(C)]
#[derive(StructuredBufferData, Copy, Clone)]
struct TestLayout2 {
    a: [i32; 3],
    b: f32,
    c: [f32; 3],
}

#[test]
fn test_structured_buffer_data() {
    use graal::layout::*;
    use graal::typedesc::*;

    assert_eq!(
        <TestLayout1 as StructuredBufferData>::TYPE,
        TypeDesc::Struct(StructType {
            fields: &[
                StructField {
                    ty: &TypeDesc::Primitive(PrimitiveType::Int),
                    decorations: &[],
                    matrix_layout: None,
                    matrix_stride: None,
                    offset: None,
                    member_info: Default::default()
                },
                StructField {
                    ty: &TypeDesc::Primitive(PrimitiveType::Int),
                    decorations: &[],
                    matrix_layout: None,
                    matrix_stride: None,
                    offset: None,
                    member_info: Default::default()
                }
            ],
            decorations: &[],
            block: false,
            buffer_block: false,
            struct_layout: None
        })
    );

    assert_eq!(
        <TestLayout1 as StructuredBufferData>::LAYOUT,
        Layout {
            align: 4,
            size: 8,
            inner: InnerLayout::Struct(FieldsLayout {
                offsets: &[0, 4],
                layouts: &[
                    &Layout {
                        align: 4,
                        size: 4,
                        inner: InnerLayout::None
                    },
                    &Layout {
                        align: 4,
                        size: 4,
                        inner: InnerLayout::None
                    }
                ]
            })
        }
    );

    assert_eq!(
        <TestLayout2 as StructuredBufferData>::TYPE,
        TypeDesc::Struct(StructType {
            fields: &[
                StructField {
                    ty: &TypeDesc::Array {
                        elem_ty: &TypeDesc::Primitive(PrimitiveType::Int),
                        len: 3
                    },
                    decorations: &[],
                    matrix_layout: None,
                    matrix_stride: None,
                    offset: None,
                    member_info: Default::default()
                },
                StructField {
                    ty: &TypeDesc::Primitive(PrimitiveType::Float),
                    decorations: &[],
                    matrix_layout: None,
                    matrix_stride: None,
                    offset: None,
                    member_info: Default::default()
                },
                StructField {
                    ty: &TypeDesc::Array {
                        elem_ty: &TypeDesc::Primitive(PrimitiveType::Float),
                        len: 3
                    },
                    decorations: &[],
                    matrix_layout: None,
                    matrix_stride: None,
                    offset: None,
                    member_info: Default::default()
                }
            ],
            decorations: &[],
            block: false,
            buffer_block: false,
            struct_layout: None
        })
    );

    assert_eq!(
        <TestLayout2 as StructuredBufferData>::LAYOUT,
        Layout {
            align: 4,
            size: 28,
            inner: InnerLayout::Struct(FieldsLayout {
                offsets: &[0, 12, 16],
                layouts: &[
                    &Layout {
                        align: 4,
                        size: 12,
                        inner: InnerLayout::Array(ArrayLayout {
                            elem_layout: &Layout::with_size_align(4, 4),
                            stride: 4
                        })
                    },
                    &Layout {
                        align: 4,
                        size: 4,
                        inner: InnerLayout::None
                    },
                    &Layout {
                        align: 4,
                        size: 12,
                        inner: InnerLayout::Array(ArrayLayout {
                            elem_layout: &Layout::with_size_align(4, 4),
                            stride: 4
                        })
                    }
                ]
            })
        }
    );
}
