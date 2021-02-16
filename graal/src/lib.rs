#![feature(new_uninit)]

mod context;
pub(crate) mod descriptor;
pub mod device;
pub(crate) mod instance;
pub(crate) mod pass;
pub mod pipeline;
pub(crate) mod resource;
pub mod surface;
pub(crate) mod swapchain;

pub(crate) use crate::device::MAX_QUEUES;
pub(crate) use crate::instance::VULKAN_ENTRY;
pub(crate) use crate::instance::VULKAN_INSTANCE;

pub use crate::resource::get_mip_level_count;
pub use crate::context::Batch;
pub use crate::resource::BufferResourceCreateInfo;
pub use crate::resource::ImageResourceCreateInfo;
pub use crate::resource::ResourceMemoryInfo;
pub use crate::context::Context;
pub use crate::context::{ResourceId, SwapchainId};
pub use crate::device::Device;
pub use ash::vk;
