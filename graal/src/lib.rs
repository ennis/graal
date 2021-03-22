#![feature(maybe_uninit_slice)]
#![feature(maybe_uninit_ref)]

pub use ash::{self, vk};
pub use graal_macros::{
    DescriptorSetInterface, StructuredBufferData, VertexData, VertexInputInterface, FragmentOutputInterface
};
pub use graal_spirv::{layout, typedesc};

pub use crate::{
    buffer::{BoolU32, BufferData, StructuredBufferData},
    context::{
        resource::{
            get_mip_level_count, BufferResourceCreateInfo, ImageResourceCreateInfo, ResourceId,
            ResourceMemoryInfo, BufferInfo, ImageInfo, ImageId, BufferId, TypedBufferInfo
        },
        Batch, Context, DescriptorSetAllocatorId, SwapchainId, RenderPassId, CommandContext,
        format_aspect_mask
    },
    descriptor::{
        extract_descriptor_set_layouts_from_shader_stages, BufferDescriptor,
        DescriptorSetInterface, DescriptorSetLayoutBindingInfo, DescriptorSetLayoutInfo,
        DescriptorSource, PipelineShaderStage,
    },
    device::Device,
    vertex::{
        vertex_macro_helpers, IndexData, Norm, VertexAttribute, VertexAttributeType,
        VertexBindingInterface, VertexBufferView, VertexData, VertexInputBindingAttributes,
        VertexInputInterface, VertexInputInterfaceExt
    },
    fragment_output::{FragmentOutputInterface, FragmentOutputInterfaceExt}
};
pub(crate) use crate::{
    device::MAX_QUEUES,
    instance::{VULKAN_ENTRY, VULKAN_INSTANCE},
};

pub(crate) mod buffer;
pub mod cache;
mod context;
pub(crate) mod descriptor;
pub mod device;
pub(crate) mod instance;
pub mod pipeline;
pub mod surface;
pub(crate) mod swapchain;
pub(crate) mod vertex;
pub(crate) mod fragment_output;

/// For internal use by `graal_macros`.
pub mod internal {
    pub use once_cell::sync::OnceCell;
}
