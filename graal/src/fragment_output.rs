use crate::vk;

pub trait FragmentOutputInterface {
    const ATTACHMENTS: &'static [vk::AttachmentDescription];
    const COLOR_ATTACHMENTS: &'static [vk::AttachmentReference];
    const DEPTH_ATTACHMENT: Option<vk::AttachmentReference>;
    const RENDER_PASS_CREATE_INFO: &'static vk::RenderPassCreateInfo;
}