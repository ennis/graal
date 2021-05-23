#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_ref)]

pub use ash::{self, vk};
pub use graal_macros::{
    DescriptorSetInterface, FragmentOutputInterface, StructuredBufferData, VertexData,
    VertexInputInterface,
};
pub use graal_spirv::{layout, typedesc};

pub use crate::{
    buffer::{BoolU32, BufferData, StructuredBufferData},
    context::{
        format_aspect_mask,
        frame::FrameCreateInfo,
        get_mip_level_count, BufferId, BufferInfo, BufferResourceCreateInfo, ImageId,
        ImageInfo, ImageResourceCreateInfo, ResourceId, ResourceMemoryInfo, TypedBufferInfo,
        AccessType, AccessTypeInfo, CommandContext, Context, DescriptorSetAllocatorId, Frame,
        GpuFuture, PipelineLayoutId, RenderPassId,
    },
    descriptor::{
        extract_descriptor_set_layouts_from_shader_stages, BufferDescriptor,
        DescriptorSetInterface, DescriptorSetLayoutBindingInfo, DescriptorSetLayoutInfo,
        DescriptorSource, PipelineShaderStage,
    },
    device::Device,
    fragment_output::{FragmentOutputInterface, FragmentOutputInterfaceExt},
    vertex::{
        vertex_macro_helpers, IndexData, Norm, VertexAttribute, VertexAttributeType,
        VertexBindingInterface, VertexBufferView, VertexData, VertexInputBindingAttributes,
        VertexInputInterface, VertexInputInterfaceExt,
    },
};

pub(crate) use crate::{
    device::MAX_QUEUES,
    instance::{VULKAN_ENTRY, VULKAN_INSTANCE},
};

pub use instance::{get_instance_extensions, get_vulkan_entry, get_vulkan_instance};

pub(crate) mod buffer;
pub mod cache;
mod context;
pub(crate) mod descriptor;
pub mod device;
pub(crate) mod fragment_output;
pub(crate) mod instance;
pub mod pipeline;
pub mod surface;
pub mod swapchain;
pub mod utils;
pub(crate) mod vertex;
mod platform_impl;

pub mod platform;

/// For internal use by `graal_macros`.
pub mod internal {
    pub use once_cell::sync::OnceCell;
}
