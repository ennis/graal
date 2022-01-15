#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub use crate::{
    context::{
        Context, format_aspect_mask, Frame,
        frame::FrameCreateInfo, frame::PassBuilder, GpuFuture, is_depth_and_stencil_format, is_depth_only_format, is_stencil_only_format,
        is_write_access, RecordingContext,
    },
    device::{Device, create_device_and_context},
    resource::{
        AllocationRequirements, BufferId, BufferInfo, BufferRegistrationInfo, BufferResourceCreateInfo,
        get_mip_level_count, ImageId, ImageInfo, ImageRegistrationInfo,
        ImageResourceCreateInfo, ResourceGroupId, ResourceId, ResourceOwnership,
        ResourceRegistrationInfo, DescriptorSetLayoutId, PipelineLayoutId, PipelineId, SamplerId
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
