use mlr::{attachments::Attachments, vk::Format};

#[derive(Attachments)]
struct GBuffers {
    #[attachment(color, format=R16G16B16A16_SFLOAT)]
    color: mlr::image::ImageHandle,
    // assume color by default
    #[attachment(format=R16G16_SFLOAT)]
    normal: mlr::image::ImageHandle,
    #[attachment(color, format=R16G16_SFLOAT)]
    tangent: mlr::image::ImageHandle,
    #[attachment(depth, format=R16G16_SFLOAT)]
    depth: mlr::image::ImageHandle,
}

#[test]
fn test_attachments() {
    assert_eq!(
        <GBuffers as Attachments>::COLOR,
        &[
            Format::R16G16B16A16_SFLOAT,
            Format::R16G16_SFLOAT,
            Format::R16G16_SFLOAT
        ]
    );
}
