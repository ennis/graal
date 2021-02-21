use graal::Norm;
use graal::VertexAttribute;
use graal::VertexBufferView;
use graal::VertexData;
use graal::VertexInputInterface;
use graal::{vk, VertexInputBindingAttributes};
use std::mem;

#[repr(C)]
#[derive(VertexData, Copy, Clone)]
struct Vertex {
    pos: [f32; 3],
    norm: [f32; 3],
    tex: [Norm<u16>; 2],
}

#[derive(VertexInputInterface)]
struct VertexInput {
    #[layout(binding = 0, location = 0, per_vertex)]
    vertices: VertexBufferView<Vertex>,
    #[layout(binding = 2, location = 3, per_vertex)]
    previous_vertices: VertexBufferView<Vertex>,
}

#[test]
fn test_vertex_layout() {
    assert_eq!(
        <Vertex as VertexData>::ATTRIBUTES,
        &[
            VertexAttribute {
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 0
            },
            VertexAttribute {
                format: vk::Format::R32G32B32_SFLOAT,
                offset: 12
            },
            VertexAttribute {
                format: vk::Format::R16G16_UNORM,
                offset: 24
            }
        ]
    );

    assert_eq!(<VertexInput as VertexInputInterface>::ATTRIBUTES.len(), 6);

    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[0].binding,
        0
    );
    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[0].stride,
        mem::size_of::<Vertex>() as u32
    );
    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[0].input_rate,
        vk::VertexInputRate::VERTEX
    );
    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[1].binding,
        2
    );
    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[1].stride,
        mem::size_of::<Vertex>() as u32
    );
    assert_eq!(
        <VertexInput as VertexInputInterface>::BINDINGS[1].input_rate,
        vk::VertexInputRate::VERTEX
    );

    let a0 = <VertexInput as VertexInputInterface>::ATTRIBUTES[0];
    let a1 = <VertexInput as VertexInputInterface>::ATTRIBUTES[1];
    let a2 = <VertexInput as VertexInputInterface>::ATTRIBUTES[2];
    let a3 = <VertexInput as VertexInputInterface>::ATTRIBUTES[3];
    let a4 = <VertexInput as VertexInputInterface>::ATTRIBUTES[4];
    let a5 = <VertexInput as VertexInputInterface>::ATTRIBUTES[5];

    assert_eq!((a0.location, a0.binding, a0.format, a0.offset), (0, 0, vk::Format::R32G32B32_SFLOAT, 0));
    assert_eq!((a1.location, a1.binding, a1.format, a1.offset), (1, 0, vk::Format::R32G32B32_SFLOAT, 12));
    assert_eq!((a2.location, a2.binding, a2.format, a2.offset), (2, 0, vk::Format::R16G16_UNORM, 24));
    assert_eq!((a3.location, a3.binding, a3.format, a3.offset), (3, 2, vk::Format::R32G32B32_SFLOAT, 0));
    assert_eq!((a4.location, a4.binding, a4.format, a4.offset), (4, 2, vk::Format::R32G32B32_SFLOAT, 12));
    assert_eq!((a5.location, a5.binding, a5.format, a5.offset), (5, 2, vk::Format::R16G16_UNORM, 24));
}
