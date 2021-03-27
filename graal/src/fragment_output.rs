use crate::{
    ash::version::DeviceV1_0,
    context::{CommandContext, RenderPassId},
    vk,
    vk::RenderPass,
    Frame, Context,
};

/// Types that describe a fragment output interface (a set of images that act as attachments).
/// They contain the images that should be bound as attachments and metadata to create
/// a compatible single-subpass render pass.
///
/// You should probably use the derive macro `#[derive(FragmentOutputInterface)]` to derive this trait.
///
/// Safety requirements: vk::RenderPassCreateInfo must contain valid pointers and describe a valid
/// render pass.
pub unsafe trait FragmentOutputInterface {
    /// List of attachments, directly usable in `vk::RenderPassCreateInfo`.
    const ATTACHMENTS: &'static [vk::AttachmentDescription];

    /// References to all color attachments in `ATTACHMENTS`. Directly usable in `vk::SubpassDescription`.
    const COLOR_ATTACHMENTS: &'static [vk::AttachmentReference];

    /// An optional reference to the depth attachment within `ATTACHMENTS`.
    const DEPTH_ATTACHMENT: Option<vk::AttachmentReference>;

    /// Creation info for a single-subpass render pass with the declared attachments.
    const RENDER_PASS_CREATE_INFO: &'static vk::RenderPassCreateInfo;

    fn get_or_init_render_pass(init: impl FnOnce() -> RenderPassId) -> RenderPassId;

    /// Creates an instance of this fragment output interface by allocating transient images for each
    /// attachments using the specified size and image usage flags.
    fn new(batch: &Frame, additional_usage: vk::ImageUsageFlags, size: (u32, u32)) -> Self
    where
        Self: Sized;

    /// Creates a framebuffer composed of all attachments in `self`.
    fn create_framebuffer(&self, cctx: &mut CommandContext, size: (u32, u32)) -> vk::Framebuffer;
}

pub trait FragmentOutputInterfaceExt: FragmentOutputInterface {
    fn create_render_pass(context: &mut Context) -> vk::RenderPass;
}

impl<T: FragmentOutputInterface> FragmentOutputInterfaceExt for T {
    fn create_render_pass(context: &mut Context) -> RenderPass {
        // Safety: through the requirements of the implementations FragmentOutputInterface
        unsafe {
            context
                .device()
                .create_render_pass(Self::RENDER_PASS_CREATE_INFO, None)
                .unwrap()
        }
    }
}
