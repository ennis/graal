use crate::{
    vk,
    vk::{DescriptorType, ShaderStageFlags},
};
use graal::device::{BufferHandle, ImageHandle};
use std::cell::Cell;

enum Binding {}

pub trait ArgumentVisitor {
    /// Image resource + descriptor.
    fn visit_image(
        &mut self,
        binding: usize,
        image: ImageHandle,
        image_view: vk::ImageView,
        descriptor_type: vk::DescriptorType,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
        layout: vk::ImageLayout,
    );

    /// Buffer resource + descriptor.
    fn visit_buffer(
        &mut self,
        binding: usize,
        buffer: BufferHandle,
        descriptor_type: vk::DescriptorType,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    );

    /// Sampler descriptor.
    fn visit_sampler(&mut self, binding: usize, sampler: Sampler);

    /// Inline uniform data.
    fn visit_data(&mut self, binding: usize, data: &[u8]);
}

/*
/// Arguments with statically known descriptor set layout.
pub trait StaticArguments {
    const TYPE_ID: std::any::TypeId;
}*/

/// Shader arguments (uniforms, textures, etc.).
pub trait Arguments {
    /// The descriptor set layout of this argument.
    ///
    /// This can be used to create/fetch a DescriptorSetLayout without needing
    /// an instance.
    const LAYOUT: &'static [vk::DescriptorSetLayoutBinding];
}

pub unsafe trait DescriptorBinding {
    /// Descriptor type.
    const DESCRIPTOR_TYPE: vk::DescriptorType;
    /// Number of descriptors represented in this object.
    const DESCRIPTOR_COUNT: u32;
    /// Which shader stages can access a resource for this binding.
    const SHADER_STAGES: vk::ShaderStageFlags;

    fn visit<A: ArgumentVisitor>(&self, binding: usize, visitor: &mut A);
}

//--------------------------------------------------------------------------------------------------

/// Sampled image descriptor.
#[derive(Debug)]
pub struct SampledImage2D {
    pub(crate) image: ImageHandle,
    pub(crate) view: vk::ImageView,
    pub(crate) descriptor: vk::DescriptorImageInfo,
}

unsafe impl DescriptorBinding for SampledImage2D {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLED_IMAGE;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn visit<A: ArgumentVisitor>(&self, binding: usize, visitor: &mut A) {
        visitor.visit_image(
            binding,
            self.image,
            self.view,
            vk::DescriptorType::SAMPLED_IMAGE,
            vk::AccessFlags::SHADER_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        );
    }
}

//--------------------------------------------------------------------------------------------------

/// Combined image/sampler descriptor.
#[derive(Debug)]
pub struct CombinedImageSampler2D<S: Sampler> {
    pub image: ImageHandle,
    pub sampler: S,
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

#[derive(Copy, Clone, Debug)]
#[repr(C)]
pub struct UniformBuffer {
    pub(crate) buffer: BufferHandle,
    pub(crate) offset: vk::DeviceSize,
    pub(crate) range: vk::DeviceSize,
}

unsafe impl DescriptorBinding for UniformBuffer {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;
    const UPDATE_OFFSET: usize = Self::layout().descriptor.offset;
    const UPDATE_STRIDE: usize = Self::layout().descriptor.size;
    const DESCRIPTOR_COUNT: u32 = 1;

    fn prepare_descriptors(&mut self, frame: &mut FrameResources) {
        self.descriptor = vk::DescriptorBufferInfo {
            buffer: self.buffer.handle(),
            offset: self.offset,
            range: self.range,
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

// used internally by `#[derive(Arguments)]`
//#[doc(hidden)]
//pub struct ShaderStageDescriptorBinding<D, const SHADER_STAGES: u32>(pub D);

//--------------------------------------------------------------------------------------------------

pub struct DescriptorSet<T: Arguments> {
    descriptor_set: vk::DescriptorSet,
}

pub struct SampledImage(graal::device::ImageHandle);
