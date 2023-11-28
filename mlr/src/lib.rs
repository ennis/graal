//pub mod argument;
pub mod attachments;
pub mod buffer;
pub mod image;
pub mod vertex;

/*
pub use graal::{
    self,
    device::{
        BufferHandle, BufferId, BufferResourceCreateInfo, ImageHandle, ImageId, ImageResourceCreateInfo, ResourceId,
    },
    vk,
};*/

// Re-exports
pub use graal::vk;

#[doc(hidden)]
pub use memoffset::offset_of as __offset_of;
#[doc(hidden)]
pub use memoffset::offset_of_tuple as __offset_of_tuple;
