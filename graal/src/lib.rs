#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub use crate::{
    context::{
        format_aspect_mask,
        frame::{FrameCreateInfo, PassBuilder},
        is_depth_and_stencil_format, is_depth_only_format, is_stencil_only_format, is_write_access,
        Context, Frame, GpuFuture, RecordingContext,
    },
    device::{create_device_and_context, Device},
    resource::{
        get_mip_level_count, AllocationRequirements, BufferId, BufferInfo, BufferRegistrationInfo,
        BufferResourceCreateInfo, DescriptorSetLayoutId, ImageId, ImageInfo, ImageRegistrationInfo,
        ImageResourceCreateInfo, PipelineId, PipelineLayoutId, ResourceGroupId, ResourceId,
        ResourceOwnership, ResourceRegistrationInfo, SamplerId,
    },
    serial::{FrameNumber, QueueSerialNumbers, SubmissionNumber},
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
pub mod serial;
pub mod surface;
pub mod swapchain;
pub mod utils;
