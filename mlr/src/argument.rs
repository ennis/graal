use crate::{device::Device, sampler::Sampler, vk};
use graal::device::{BufferHandle, ImageHandle};
use std::{borrow::Cow, ffi::c_void};

pub use mlr_macros::{Arguments, PushConstants};

pub trait ArgumentVisitor {
    /// Image resource + descriptor.
    fn visit_image(
        &mut self,
        binding: u32,
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
        binding: u32,
        buffer: BufferHandle,
        descriptor_type: vk::DescriptorType,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    );

    /// Sampler descriptor.
    fn visit_sampler(&mut self, binding: u32, sampler: Sampler);

    /// Inline data.
    ///
    /// # Safety
    ///
    /// `data` must be a valid pointer to a block of memory of size `byte_size`.
    ///
    // NOTE: I tried to make it a `&[u8]` instead of `*const c_void`+size but this requires bytemuck and I couldn't make
    // the bytemuck derives work inside the `#[derive(Arguments)]` macro (https://github.com/Lokathor/bytemuck/issues/159).
    unsafe fn visit_inline_uniform_block(&mut self, binding: u32, byte_size: usize, data: *const c_void);
}

#[derive(Debug, Clone)]
pub struct ArgumentsLayout<'a> {
    pub bindings: Cow<'a, [vk::DescriptorSetLayoutBinding]>,
}

/// Shader arguments (uniforms, textures, etc.).
pub trait StaticArguments: Arguments {
    /// The descriptor set layout of this argument.
    ///
    /// This can be used to create/fetch a DescriptorSetLayout without needing
    /// an instance.
    const LAYOUT: ArgumentsLayout<'static>;

    // Creates the VkDescriptorSetLayout object that represents `LAYOUT`.
    //
    // The returned object is expected to live for the duration of the application
    // (it is leaked, essentially).
    //fn create_descriptor_set_layout(device: &Device) -> vk::DescriptorSetLayout;
}

pub trait Arguments {
    /// Visit all arguments in the block.
    fn visit<A: ArgumentVisitor>(&self, visitor: &mut A);
}

pub unsafe trait DescriptorBinding {
    /// Descriptor type.
    const DESCRIPTOR_TYPE: vk::DescriptorType;
    /// Number of descriptors represented by this object.
    ///
    /// This is `1` for objects that don't represent a descriptor array, or the array size otherwise.
    const DESCRIPTOR_COUNT: u32;
    /// Which shader stages can access a resource for this binding.
    const SHADER_STAGES: vk::ShaderStageFlags;

    fn visit<A: ArgumentVisitor>(&self, binding: u32, visitor: &mut A);
}

pub trait StaticPushConstants {
    /// The push constant ranges of this argument.
    const PUSH_CONSTANT_RANGES: &'static [vk::PushConstantRange];
}

pub trait PushConstants {
    /// Returns the push constant ranges.
    fn push_constant_ranges(&self) -> Cow<'static, [vk::PushConstantRange]>;
}

impl StaticPushConstants for () {
    const PUSH_CONSTANT_RANGES: &'static [vk::PushConstantRange] = &[];
}

impl<T> PushConstants for T
where
    T: StaticPushConstants,
{
    fn push_constant_ranges(&self) -> Cow<'static, [vk::PushConstantRange]> {
        Cow::Borrowed(Self::PUSH_CONSTANT_RANGES)
    }
}

//--------------------------------------------------------------------------------------------------

/// Sampled image descriptor.
#[derive(Debug)]
pub struct SampledImage {
    pub(crate) image: ImageHandle,
    pub(crate) view: vk::ImageView,
}

unsafe impl DescriptorBinding for SampledImage {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLED_IMAGE;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn visit<A: ArgumentVisitor>(&self, binding: u32, visitor: &mut A) {
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

/*
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
*/

//--------------------------------------------------------------------------------------------------

#[derive(Copy, Clone, Debug)]
pub struct UniformBuffer<T> {
    pub buffer: BufferHandle,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    _phantom: std::marker::PhantomData<fn() -> T>,
}

unsafe impl<T> DescriptorBinding for UniformBuffer<T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn visit<A: ArgumentVisitor>(&self, binding: u32, visitor: &mut A) {
        visitor.visit_buffer(
            binding,
            self.buffer,
            vk::DescriptorType::UNIFORM_BUFFER,
            vk::AccessFlags::UNIFORM_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
        );
    }
}

//--------------------------------------------------------------------------------------------------

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation details of `#[derive(Arguments)]`

#[doc(hidden)]
pub fn create_descriptor_set_layout(device: &Device, layout: &ArgumentsLayout<'_>) -> vk::DescriptorSetLayout {
    let create_info = vk::DescriptorSetLayoutCreateInfo {
        flags: vk::DescriptorSetLayoutCreateFlags::empty(),
        binding_count: layout.bindings.len() as u32,
        p_bindings: layout.bindings.as_ptr(),
        ..Default::default()
    };
    unsafe {
        device
            .device()
            .create_descriptor_set_layout(&create_info, None)
            .expect("failed to create descriptor set layout")
    }
}

pub trait ArgumentSet<const INDEX: usize> {
    type Arguments: Arguments;
}

fn test<P, const N: usize>(pipeline: &P, args: P::Arguments)
where
    P: ArgumentSet<{ N }>,
{
}
