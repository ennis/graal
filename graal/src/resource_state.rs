use crate::vk;

#[derive(Copy, Clone, Debug)]
pub struct ResourceState {
    /// Stages that will access the resource.
    pub stages: vk::PipelineStageFlags2,
    /// Access flags for the resource.
    pub access: vk::AccessFlags2,
    /// Requested layout for the resource.
    pub layout: vk::ImageLayout,
}

impl ResourceState {
    pub const TRANSFER_SRC: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    };
    pub const TRANSFER_DST: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    };
    pub const SHADER_READ: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::ALL_GRAPHICS,
        access: vk::AccessFlags2::SHADER_READ,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    pub const COLOR_ATTACHMENT_OUTPUT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    pub const DEPTH_STENCIL_ATTACHMENT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    pub const VERTEX_BUFFER: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::VERTEX_INPUT,
        access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        layout: vk::ImageLayout::UNDEFINED,
    };
    pub const PRESENT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
        layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };
}
