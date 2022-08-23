//#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub use crate::{
    context::{
        format_aspect_mask, is_depth_and_stencil_format, is_depth_only_format, is_stencil_only_format, is_write_access,
        CommandCallback, Context, Frame, GpuFuture, PassBuilder, QueueCallback, RecordingContext, SubmitInfo,
    },
    device::{
        create_device_and_context, get_mip_level_count, AllocationRequirements, BufferId, BufferInfo,
        BufferRegistrationInfo, BufferResourceCreateInfo, DescriptorSetLayoutId, Device, ImageId, ImageInfo,
        ImageRegistrationInfo, ImageResourceCreateInfo, PipelineId, PipelineLayoutId, ResourceGroupId, ResourceId,
        ResourceOwnership, ResourceRegistrationInfo, SamplerId, Swapchain, SwapchainImage,
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
pub mod serial;
pub mod surface;
pub mod utils;
