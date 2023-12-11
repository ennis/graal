use crate::{
    image::{ImageAny, ImageViewInfo},
    vk, ImageView,
};

pub use graal_macros::Attachments;

/// Trait implemented for types that contain attachments.
pub trait StaticAttachments {
    /// Color attachment formats.
    const COLOR: &'static [vk::Format];
    /// Depth attachment format.
    const DEPTH: Option<vk::Format>;
    /// Stencil attachment format.
    const STENCIL: Option<vk::Format>;
}

/// Describes a color, depth, or stencil attachment.
pub struct Attachment<'a> {
    pub image_view: &'a ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: Option<vk::ClearValue>,
}

impl<'a> Attachment<'a> {
    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color(mut self, color: [f32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color_int(mut self, color: [i32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { int32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color_uint(mut self, color: [u32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { uint32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear depth of this attachment.
    pub fn clear_depth(mut self, depth: f32) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
        });
        self
    }

    pub fn clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        });
        self
    }

    /// Sets `load_op` to DONT_CARE.
    pub fn load_discard(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    /// Sets `store_op` to DONT_CARE.
    pub fn store_discard(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }
}

pub trait Attachments {
    /// Returns an iterator over the color attachments in this object.
    fn color_attachments(&self) -> impl Iterator<Item = Attachment<'_>> + '_;

    /// Returns the depth attachment.
    fn depth_attachment(&self) -> Option<Attachment>;

    /// Returns the stencil attachment.
    fn stencil_attachment(&self) -> Option<Attachment>;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Types that describe a color, depth, or stencil attachment to a rendering operation.
pub trait AsAttachment<'a> {
    /// Returns an object describing the attachment.
    fn as_attachment(&self) -> Attachment<'a>;
}

/// References to images can be used as attachments.
impl<'a> AsAttachment<'a> for &'a ImageView {
    fn as_attachment(&self) -> Attachment<'a> {
        Attachment {
            image_view: self,
            load_op: Default::default(),
            store_op: Default::default(),
            clear_value: None,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// macro helper types

#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    Float([f32; 4]),
    Int([i32; 4]),
    Uint([u32; 4]),
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(v: [f32; 4]) -> Self {
        Self::Float(v)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(v: [i32; 4]) -> Self {
        Self::Int(v)
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(v: [u32; 4]) -> Self {
        Self::Uint(v)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(v: ClearColorValue) -> Self {
        match v {
            ClearColorValue::Float(v) => vk::ClearColorValue { float32: v },
            ClearColorValue::Int(v) => vk::ClearColorValue { int32: v },
            ClearColorValue::Uint(v) => vk::ClearColorValue { uint32: v },
        }
    }
}

#[doc(hidden)]
pub struct AttachmentOverride<A>(
    pub A,
    pub Option<vk::AttachmentLoadOp>,
    pub Option<vk::AttachmentStoreOp>,
    pub Option<vk::ClearValue>,
);

impl<'a, A: AsAttachment<'a>> AsAttachment<'a> for AttachmentOverride<A> {
    fn as_attachment(&self) -> Attachment<'a> {
        let mut desc = self.0.as_attachment();
        if let Some(load_op) = self.1 {
            desc.load_op = load_op;
        }
        if let Some(store_op) = self.2 {
            desc.store_op = store_op;
        }
        if let Some(clear_value) = self.3 {
            desc.clear_value = Some(clear_value);
        }
        desc
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
