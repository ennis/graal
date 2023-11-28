use crate::vk;

pub use mlr_macros::Attachments;

/// Trait implemented for types that contain attachments.
pub trait Attachments {
    /// Color attachment formats.
    const COLOR: &'static [vk::Format];
    /// Depth attachment format.
    const DEPTH: Option<vk::Format>;
    /// Stencil attachment format.
    const STENCIL: Option<vk::Format>;
}
