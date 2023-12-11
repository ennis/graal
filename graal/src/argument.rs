use crate::{resource_state::ResourceState, vk, BufferAny, ImageAny, ImageView, Sampler};
use std::borrow::Cow;

pub use graal_macros::{Arguments, PushConstants};

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
}

/// Description of one argument in an argument block.
pub struct ArgumentDescription<'a> {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub kind: ArgumentKind<'a>,
}

/// Kind of argument.
#[derive(Debug, Clone)]
pub enum ArgumentKind<'a> {
    Image {
        image_view: &'a ImageView,
        resource_state: ResourceState,
    },
    Buffer {
        buffer: &'a BufferAny,
        resource_state: ResourceState,
        offset: usize,
        size: usize,
    },
    Sampler {
        sampler: &'a Sampler,
    },
}

pub trait Arguments {
    /// The type of inline data for this argument.
    type InlineData: Copy + 'static;

    /// Returns an iterator over all descriptors contained in this object.
    fn arguments(&self) -> impl Iterator<Item = ArgumentDescription> + '_;

    /// Returns the inline data for this argument.
    fn inline_data(&self) -> Cow<Self::InlineData>;
}

pub unsafe trait Argument {
    /// Descriptor type.
    const DESCRIPTOR_TYPE: vk::DescriptorType;
    /// Number of descriptors represented by this object.
    ///
    /// This is `1` for objects that don't represent a descriptor array, or the array size otherwise.
    const DESCRIPTOR_COUNT: u32;
    /// Which shader stages can access a resource for this binding.
    const SHADER_STAGES: vk::ShaderStageFlags;

    /// Returns the argument description for this object.
    ///
    /// Used internally by the `Arguments` derive macro.
    ///
    /// # Arguments
    ///
    /// * `binding`: the binding number of this argument.
    fn argument_description(&self, binding: u32) -> ArgumentDescription;
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

/// Sampled image descriptor.
#[derive(Debug)]
pub struct SampledImage<'a> {
    pub image_view: &'a ImageView,
}

unsafe impl<'a> Argument for SampledImage<'a> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLED_IMAGE;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
            kind: ArgumentKind::Image {
                image_view: self.image_view,
                resource_state: ResourceState {
                    layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    access: vk::AccessFlags2::SHADER_READ,
                    stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                },
            },
        }
    }
}

unsafe impl<'a> Argument for &'a Sampler {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::SAMPLER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::SAMPLER,
            kind: ArgumentKind::Sampler { sampler: self },
        }
    }
}

/// Uniform buffer descriptor.
#[derive(Copy, Clone, Debug)]
pub struct UniformBuffer<'a, T> {
    pub buffer: &'a BufferAny,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    _phantom: std::marker::PhantomData<fn() -> T>,
}

unsafe impl<'a, T> Argument for UniformBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.buffer,
                resource_state: ResourceState {
                    layout: vk::ImageLayout::UNDEFINED,
                    access: vk::AccessFlags2::UNIFORM_READ,
                    stages: vk::PipelineStageFlags2::ALL_COMMANDS,
                },
                offset: self.offset as usize,
                size: self.range as usize,
            },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation details of `#[derive(Arguments)]`

/*
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
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////
