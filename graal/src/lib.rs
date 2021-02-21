//#![feature(new_uninit)]

mod context;
pub(crate) mod buffer;
pub(crate) mod vertex;
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

pub use crate::buffer::BoolU32;
pub use crate::buffer::BufferData;
pub use crate::buffer::StructuredBufferData;
pub use crate::vertex::IndexData;
pub use crate::vertex::Norm;
pub use crate::vertex::VertexData;
pub use crate::vertex::VertexAttributeType;
pub use crate::vertex::VertexAttribute;
pub use crate::vertex::VertexBindingInterface;
pub use crate::vertex::VertexInputInterface;
pub use crate::vertex::VertexBufferView;
pub use crate::vertex::VertexInputBindingAttributes;
pub use crate::vertex::vertex_macro_helpers;

pub use crate::descriptor::PipelineShaderStage;
pub use crate::descriptor::DescriptorSource;
pub use crate::descriptor::DescriptorSetInterface;
pub use crate::descriptor::DescriptorSetAllocator;
pub use crate::descriptor::DescriptorSetLayoutId;
pub use crate::descriptor::DescriptorSetLayoutInfo;
pub use crate::descriptor::DescriptorSetLayoutBindingInfo;
pub use crate::descriptor::DescriptorSetLayoutCache;
pub use crate::descriptor::extract_descriptor_set_layouts_from_shader_stages;

pub use graal_macros::DescriptorSetInterface;
pub use graal_macros::VertexInputInterface;
pub use graal_macros::VertexData;

pub use crate::resource::get_mip_level_count;
pub use crate::context::Batch;
pub use crate::resource::BufferResourceCreateInfo;
pub use crate::resource::ImageResourceCreateInfo;
pub use crate::resource::ResourceMemoryInfo;
pub use crate::context::Context;
pub use crate::context::{ResourceId, SwapchainId};
pub use crate::device::Device;

pub use graal_spirv::typedesc;
pub use graal_spirv::layout;

pub use ash;
pub use ash::vk;
