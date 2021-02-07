mod context;
pub mod device;
pub(crate) mod handle;
pub(crate) mod instance;
pub(crate) mod pass;
pub mod surface;
pub mod swapchain;

pub(crate) use crate::device::MAX_QUEUES;
pub(crate) use crate::instance::VULKAN_ENTRY;
pub(crate) use crate::instance::VULKAN_INSTANCE;
pub(crate) use crate::instance::VULKAN_SURFACE_KHR;

pub use crate::device::Device;
pub use crate::context::Context;
pub use crate::context::ResourceId;
pub use crate::context::ResourceMemoryInfo;
pub use crate::context::ImageResourceCreateInfo;
pub use crate::context::BufferResourceCreateInfo;
pub use crate::context::Batch;
pub use crate::context::get_mip_level_count;
pub use ash::vk;