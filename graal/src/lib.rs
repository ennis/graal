#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub use crate::{
    context::{format_aspect_mask, Frame, frame::FrameCreateInfo, CommandContext, Context, GpuFuture},
    device::Device,
    resource::{
        get_mip_level_count, AllocationRequirements, BufferId, BufferInfo, BufferRegistrationInfo,
        BufferResourceCreateInfo, ImageId, ImageInfo, ImageRegistrationInfo,
        ImageResourceCreateInfo, ResourceGroupId, ResourceId, ResourceOwnership,
        ResourceRegistrationInfo,
    },
    serial::{QueueSerialNumbers, SubmissionNumber},
};
pub use gpu_allocator::MemoryLocation;

pub(crate) use crate::{
    device::MAX_QUEUES,
    instance::{VULKAN_ENTRY, VULKAN_INSTANCE},
};

mod context;
pub mod device;
mod instance;
pub mod platform;
mod platform_impl;
mod resource;
mod serial;
pub mod surface;
pub mod swapchain;
pub mod utils;
mod allocator;
