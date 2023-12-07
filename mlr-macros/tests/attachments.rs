use mlr::{
    attachments::{Attachments, StaticAttachments},
    image::{ImageAny, ImageHandle},
    vk::Format,
};

#[derive(Attachments)]
struct GBuffers<'a> {
    #[attachment(color, format=R16G16B16A16_SFLOAT)]
    color: &'a ImageAny,
    // assume color by default
    #[attachment(format=R16G16_SFLOAT)]
    normal: &'a ImageAny,
    #[attachment(color, format=R16G16_SFLOAT)]
    tangent: &'a ImageAny,
    #[attachment(depth, format=R16G16_SFLOAT)]
    depth: &'a ImageAny,
}

/*
#[derive(Attachments)]
struct TestClearDepth<'a> {
    #[attachment(depth, format=R16G16_SFLOAT, clear_depth=1.0)]
    depth: &'a ImageAny,
}

#[derive(Attachments)]
struct TestClearDepthStencil<'a> {
    #[attachment(depth, format=R16G16_SFLOAT, clear_depth_stencil=(1.0, 0))]
    depth: &'a ImageAny,
}

#[derive(Attachments)]
struct TestClearStencil<'a> {
    #[attachment(depth, format=R16G16_SFLOAT, clear_stencil=0)]
    depth: &'a ImageAny,
}

#[derive(Attachments)]
struct TestClearColorFloat<'a> {
    #[attachment(color, format=R16G16_SFLOAT, clear_color=[1.0, 0.0, 0.0, 1.0])]
    color: &'a ImageAny,
}

#[derive(Attachments)]
struct TestClearColorUInt<'a> {
    #[attachment(color, format=R16G16B16A16_UINT, clear_color=[128u32, 255u32, 255u32, 255u32])]
    color: &'a ImageAny,
}

#[derive(Attachments)]
struct TestClearColorInt<'a> {
    #[attachment(color, format=R16G16B16A16_SINT, clear_color=[-1, 0, -50, 254])]
    color: &'a ImageAny,
}*/

#[test]
fn test_attachments() {
    assert_eq!(
        <GBuffers as StaticAttachments>::COLOR,
        &[
            Format::R16G16B16A16_SFLOAT,
            Format::R16G16_SFLOAT,
            Format::R16G16_SFLOAT
        ]
    );
}
