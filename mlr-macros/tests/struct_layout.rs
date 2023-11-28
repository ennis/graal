use mlr::utils::FieldLayout;

#[repr(C)]
#[derive(StructLayout, Copy, Clone)]
struct TestLayout2 {
    a: [i32; 3],
    b: f32,
    c: [f32; 3],
}

#[test]
fn test_vertex_layout() {
    assert_eq!(TestLayout2::LAYOUT.a.offset, 0);
}
