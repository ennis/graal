use std::cell::Cell;
use crate::{
    buffer::BufferAny,
    context::FrameResources,
    image::ImageAny,
    sampler::SamplerType,
    vk,
    vk::{DescriptorType, ShaderStageFlags},
};
use mlr::{
    context::{Context, ContextResources},
    Frame,
};
use graal::descriptor::DescriptorSetLayoutId;

pub trait ResourceVisitor {
    fn visit_image(
        &mut self,
        image: &ImageAny,
        access_mask: graal::vk::AccessFlags,
        stage_mask: graal::vk::PipelineStageFlags,
        layout: graal::vk::ImageLayout,
    ) -> bool;
    fn visit_buffer(
        &mut self,
        buffer: &BufferAny,
        access_mask: graal::vk::AccessFlags,
        stage_mask: graal::vk::PipelineStageFlags,
    ) -> bool;
}

/// Trait implemented by types that hold references to resources.
pub trait ResourceHolder {
    /// Visits all resources referenced by this object.
    fn walk_resources(&self, visitor: &mut dyn ResourceVisitor);
}

/// Arguments with statically known descriptor set layout.
pub trait StaticArguments {
    const TYPE_ID: std::any::TypeId;
    const LAYOUT: &'static [vk::DescriptorSetLayoutBinding];
    const UPDATE_TEMPLATE_ENTRIES: Option<&'static [vk::DescriptorUpdateTemplateEntry]>;
}

/// Shader arguments (uniforms, textures, etc.).
pub trait Arguments: ResourceHolder {
    /// Returns a unique ID for the type of this structure, or None if it's unique.
    fn unique_type_id(&self) -> Option<std::any::TypeId>;

    /// Returns the descriptor set layout for this argument.
    fn get_descriptor_set_layout_bindings(&self) -> &[vk::DescriptorSetLayoutBinding];

    /// Returns the descriptor set update template entries for this argument.
    fn get_descriptor_set_update_template_entries(
        &self,
    ) -> Option<&[vk::DescriptorUpdateTemplateEntry]>;

    /// Updates a descriptor set with the data contained in the arguments.
    unsafe fn update_descriptor_set(
        &mut self,
        frame: &mut FrameResources,
        set: vk::DescriptorSet,
        update_template: Option<vk::DescriptorUpdateTemplate>,
    );
}

pub unsafe trait DescriptorBinding {
    /// Descriptor type.
    const DESCRIPTOR_TYPE: vk::DescriptorType;
    /// Which shader stages can access a resource for this binding.
    const SHADER_STAGES: vk::ShaderStageFlags;
    /// Offset to the descriptor update data within this object.
    const UPDATE_OFFSET: usize;
    /// Stride of the descriptor update data within this object.
    const UPDATE_STRIDE: usize;
    /// Number of descriptors represented in this object.
    const DESCRIPTOR_COUNT: u32;

    /// Prepares the descriptor update data during pass evaluation.
    ///
    /// # Note
    /// Implementations can access the submission context to upload uniform data, create image views,
    /// create samplers, etc.
    /// This cannot be done before evaluation since resources may not have memory bound to them at that point.
    fn prepare_descriptors(&mut self, frame: &mut FrameResources);

    fn visit(&self, visitor: &mut dyn ResourceVisitor);
}


//--------------------------------------------------------------------------------------------------

/// Sampled image descriptor.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[derive(mlr::StructLayout)]
pub struct SampledImage2D<'a> {
    pub(crate) image: &'a ImageAny,
    pub(crate) descriptor: vk::DescriptorImageInfo,
}

unsafe impl<'a> DescriptorBinding for SampledImage2D<'a> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLED_IMAGE;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;
    const UPDATE_OFFSET: usize = Self::layout().descriptor.offset;
    const UPDATE_STRIDE: usize = Self::layout().descriptor.size;

    fn prepare_descriptors(&mut self, frame: &mut FrameResources) {
        // SAFETY: TODO
        let image_view = unsafe {
            let create_info = vk::ImageViewCreateInfo {
                flags: vk::ImageViewCreateFlags::empty(),
                image: self.image.handle(),
                view_type: vk::ImageViewType::TYPE_2D,
                format: self.image.format(),
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            };
            /// FIXME: this should probably be cached into the image
            frame.create_transient_image_view(&create_info)
        };

        self.descriptor = vk::DescriptorImageInfo {
            sampler: Default::default(),
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };
    }

    fn visit(&self, visitor: &mut dyn ResourceVisitor) {
        visitor.visit_image(
            self.image,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
}

impl<'a> From<&'a ImageAny> for SampledImage2D<'a> {
    fn from(img: &'a ImageAny) -> Self {
        img.to_sampled_image_2d()
    }
}

//--------------------------------------------------------------------------------------------------

/// Combined image/sampler descriptor.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[derive(mlr::StructLayout)]
pub struct CombinedImageSampler2D<'a, S: SamplerType> {
    pub(crate) image: &'a ImageAny,
    pub(crate) sampler: S,
    pub(crate) descriptor: vk::DescriptorImageInfo,
}

unsafe impl<'a, S: SamplerType> DescriptorBinding for CombinedImageSampler2D<'a, S> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;
    const UPDATE_OFFSET: usize = Self::layout().descriptor.offset;
    const UPDATE_STRIDE: usize = Self::layout().descriptor.size;

    fn prepare_descriptors(&mut self, frame: &mut FrameResources) {
        // SAFETY: TODO
        let image_view = unsafe {
            let create_info = vk::ImageViewCreateInfo {
                flags: vk::ImageViewCreateFlags::empty(),
                image: self.image.handle(),
                view_type: vk::ImageViewType::TYPE_2D,
                format: self.image.format(),
                components: vk::ComponentMapping::default(),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                },
                ..Default::default()
            };
            /// FIXME: this should probably be cached into the image
            frame.create_transient_image_view(&create_info)
        };

        let sampler = self.sampler.to_sampler(frame.vulkan_device());
        self.descriptor = vk::DescriptorImageInfo {
            sampler,
            image_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }
    }

    fn visit(&self, visitor: &mut dyn ResourceVisitor) {
        visitor.visit_image(
            self.image,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
}

//--------------------------------------------------------------------------------------------------

/// Uniform buffer slice.
#[derive(Copy, Clone, Debug)]
#[repr(C)]
#[derive(mlr::StructLayout)]
pub struct UniformBuffer<'a> {
    pub(crate) buffer: &'a BufferAny,
    pub(crate) offset: vk::DeviceSize,
    pub(crate) range: vk::DeviceSize,
    pub(crate) descriptor: vk::DescriptorBufferInfo,
}

unsafe impl<'a> DescriptorBinding for UniformBuffer<'a> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;
    const UPDATE_OFFSET: usize = Self::layout().descriptor.offset;
    const UPDATE_STRIDE: usize = Self::layout().descriptor.size;
    const DESCRIPTOR_COUNT: u32 = 1;

    fn prepare_descriptors(&mut self, frame: &mut FrameResources) {
        self.descriptor = vk::DescriptorBufferInfo {
            buffer: self.buffer.handle(),
            offset: self.offset,
            range: self.range
        }
    }

    fn visit(&self, visitor: &mut dyn ResourceVisitor) {
        visitor.visit_buffer(
            self.buffer,
            vk::AccessFlags::UNIFORM_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
    }
}


//--------------------------------------------------------------------------------------------------

/// Argument blocks
///
/// Actually they are just descriptor sets.
pub struct ArgumentBlock<T: Arguments> {
    pub(crate) args: T,
    pub(crate) set_layout_id: graal::DescriptorSetLayoutId,
    pub(crate) set_layout: vk::DescriptorSetLayout,
    pub(crate) update_template: vk::DescriptorUpdateTemplate,
    pub(crate) descriptor_set: Cell<vk::DescriptorSet>, // allocated on first use
}
