#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub use crate::{
    context::{
        BufferId, BufferInfo, BufferResourceCreateInfo, CommandContext, Context,
        format_aspect_mask, frame::FrameCreateInfo, get_mip_level_count,
        GpuFuture, ImageId, ImageInfo, ImageResourceCreateInfo,
        ResourceId, ResourceMemoryInfo, TypedBufferInfo,
    },
    device::Device,
};
pub use gpu_allocator::MemoryLocation;

pub(crate) use crate::{
    device::MAX_QUEUES,
    instance::{VULKAN_ENTRY, VULKAN_INSTANCE},
};

mod context;
pub mod device;
pub(crate) mod instance;
pub mod surface;
pub mod swapchain;
pub mod utils;
mod platform_impl;

pub mod platform;
