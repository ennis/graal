use mlr::{
    vertex::{Norm, Vertex, VertexAttributeDescription},
    vk,
};

#[repr(C)]
#[derive(Vertex, Copy, Clone)]
struct VertexPNT {
    pos: [f32; 3],
    norm: [f32; 3],
    #[normalized]
    tex: [u16; 2],
}

#[test]
fn test_vertex_layout() {
    assert_eq!(
        <VertexPNT as Vertex>::ATTRIBUTES,
        &[
            VertexAttributeDescription {
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0
            },
            VertexAttributeDescription {
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12
            },
            VertexAttributeDescription {
                format: vk::Format::R16G16_UNORM,
                offset: 24
            }
        ]
    );
}
