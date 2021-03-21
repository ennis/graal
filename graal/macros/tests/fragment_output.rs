use graal::{vk, FragmentOutputInterface};
use std::ptr;

#[derive(FragmentOutputInterface)]
struct GBuffers {
    /// Color buffer, R16G16B16A16_SFLOAT
    #[attachment(
        color,
        format = "R16G16B16A16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    color: graal::ImageInfo,

    /// Normals, RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    normal: graal::ImageInfo,

    /// Tangents: RG16_SFLOAT
    #[attachment(
        color,
        format = "R16G16_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "COLOR_ATTACHMENT_OPTIMAL"
    )]
    tangent: graal::ImageInfo,

    /// Depth: D32_SFLOAT
    #[attachment(
        depth,
        format = "D32_SFLOAT",
        samples = 1,
        load_op = "CLEAR",
        store_op = "STORE",
        layout = "DEPTH_STENCIL_ATTACHMENT_OPTIMAL"
    )]
    depth: graal::ImageInfo,
}
/*    const ATTACHMENTS: &'static [vk::AttachmentDescription];
    const COLOR_ATTACHMENTS: &'static [vk::AttachmentReference];
    const DEPTH_ATTACHMENT: Option<&'static vk::AttachmentReference>;
    const RENDER_PASS_CREATE_INFO: &'static vk::RenderPassCreateInfo;
*/

#[test]
fn test_fragment_output() {
    let attachments = <GBuffers as FragmentOutputInterface>::ATTACHMENTS;

    assert_eq!(
        attachments[0].flags,
        vk::AttachmentDescriptionFlags::empty()
    );
    assert_eq!(attachments[0].format, vk::Format::R16G16B16A16_SFLOAT);
    assert_eq!(attachments[0].samples, vk::SampleCountFlags::TYPE_1);
    assert_eq!(attachments[0].load_op, vk::AttachmentLoadOp::CLEAR);
    assert_eq!(attachments[0].store_op, vk::AttachmentStoreOp::STORE);
    assert_eq!(
        attachments[0].stencil_load_op,
        vk::AttachmentLoadOp::DONT_CARE
    );
    assert_eq!(
        attachments[0].stencil_store_op,
        vk::AttachmentStoreOp::DONT_CARE
    );
    assert_eq!(
        attachments[0].initial_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        attachments[0].final_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );

    assert_eq!(
        attachments[1].flags,
        vk::AttachmentDescriptionFlags::empty()
    );
    assert_eq!(attachments[1].format, vk::Format::R16G16_SFLOAT);
    assert_eq!(attachments[1].samples, vk::SampleCountFlags::TYPE_1);
    assert_eq!(attachments[1].load_op, vk::AttachmentLoadOp::CLEAR);
    assert_eq!(attachments[1].store_op, vk::AttachmentStoreOp::STORE);
    assert_eq!(
        attachments[1].stencil_load_op,
        vk::AttachmentLoadOp::DONT_CARE
    );
    assert_eq!(
        attachments[1].stencil_store_op,
        vk::AttachmentStoreOp::DONT_CARE
    );
    assert_eq!(
        attachments[1].initial_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        attachments[1].final_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );

    assert_eq!(
        attachments[2].flags,
        vk::AttachmentDescriptionFlags::empty()
    );
    assert_eq!(attachments[2].format, vk::Format::R16G16_SFLOAT);
    assert_eq!(attachments[2].samples, vk::SampleCountFlags::TYPE_1);
    assert_eq!(attachments[2].load_op, vk::AttachmentLoadOp::CLEAR);
    assert_eq!(attachments[2].store_op, vk::AttachmentStoreOp::STORE);
    assert_eq!(
        attachments[2].stencil_load_op,
        vk::AttachmentLoadOp::DONT_CARE
    );
    assert_eq!(
        attachments[2].stencil_store_op,
        vk::AttachmentStoreOp::DONT_CARE
    );
    assert_eq!(
        attachments[2].initial_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        attachments[2].final_layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );

    assert_eq!(
        attachments[3].flags,
        vk::AttachmentDescriptionFlags::empty()
    );
    assert_eq!(attachments[3].format, vk::Format::D32_SFLOAT);
    assert_eq!(attachments[3].samples, vk::SampleCountFlags::TYPE_1);
    assert_eq!(attachments[3].load_op, vk::AttachmentLoadOp::CLEAR);
    assert_eq!(attachments[3].store_op, vk::AttachmentStoreOp::STORE);
    assert_eq!(
        attachments[3].stencil_load_op,
        vk::AttachmentLoadOp::DONT_CARE
    );
    assert_eq!(
        attachments[3].stencil_store_op,
        vk::AttachmentStoreOp::DONT_CARE
    );
    assert_eq!(
        attachments[3].initial_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );
    assert_eq!(
        attachments[3].final_layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );

    let color_att = <GBuffers as FragmentOutputInterface>::COLOR_ATTACHMENTS;
    assert_eq!(color_att[0].attachment, 0);
    assert_eq!(
        color_att[0].layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(color_att[1].attachment, 1);
    assert_eq!(
        color_att[1].layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );
    assert_eq!(color_att[2].attachment, 2);
    assert_eq!(
        color_att[2].layout,
        vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
    );

    let depth_att = <GBuffers as FragmentOutputInterface>::DEPTH_ATTACHMENT;
    assert!(depth_att.is_some());
    assert_eq!(depth_att.unwrap().attachment, 3);
    assert_eq!(
        depth_att.unwrap().layout,
        vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL
    );

    /*assert_eq!(
        <GBuffers as FragmentOutputInterface>::RENDER_PASS_CREATE_INFO,
        vk::RenderPassCreateInfo {
            attachment_count: <GBuffers as FragmentOutputInterface>::COLOR_ATTACHMENTS.len(),
            p_attachments: <GBuffers as FragmentOutputInterface>::COLOR_ATTACHMENTS.as_ptr(),
            subpass_count: 1,
            p_subpasses: <GBuffers as FragmentOutputInterface>::RENDER_PASS_CREATE_INFO.p_subpasses, // TODO
            dependency_count: 0,
            p_dependencies: ptr::null(),
            ..Default::default()
        }
    );*/
}
