use std::{
    borrow::Cow,
    convert::TryInto,
    marker::PhantomData,
    mem,
    ops::{Bound, RangeBounds},
};

// reexports
pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;
// reexports for macro internals
#[doc(hidden)]
pub use memoffset::offset_of as __offset_of;
#[doc(hidden)]
pub use memoffset::offset_of_tuple as __offset_of_tuple;
pub use ordered_float;

pub use device::*;
// proc-macros
pub use graal_macros::{Arguments, Attachments, Vertex};
pub use instance::*;
//pub use platform::*;
pub use surface::*;
pub use types::*;

mod device;
mod instance;
mod platform;
mod platform_impl;
mod surface;
mod types;
pub mod util;

pub mod prelude {
    pub use crate::{
        util::{DeviceExt, QueueExt},
        vk, Arguments, Attachments, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CompareOp,
        DepthStencilState, Device, Format, FragmentOutputInterfaceDescriptor, FrontFace, GraphicsPipeline,
        GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageType, ImageUsage, ImageView, IndexType,
        LineRasterization, LineRasterizationMode, MemoryLocation, PipelineBindPoint, PipelineLayoutDescriptor, Point2D,
        PolygonMode, PreRasterizationShaders, PrimitiveTopology, Queue, RasterizationState, Rect2D, SampledImage,
        Sampler, SamplerCreateInfo, ShaderCode, ShaderEntryPoint, ShaderSource, Size2D, StaticArguments,
        StaticAttachments, StencilState, TypedBuffer, Vertex, VertexBufferDescriptor, VertexBufferLayoutDescription,
        VertexInputAttributeDescription, VertexInputRate, VertexInputState,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to create device")]
    DeviceCreationFailed(#[from] DeviceCreateError),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("compilation error: {0}")]
    Shaderc(#[from] shaderc::Error),
    #[error("Vulkan error: {0}")]
    Vulkan(#[from] vk::Result),
}

#[derive(Copy, Clone, Debug)]
pub struct ImageCopyBuffer<'a> {
    pub buffer: &'a Buffer,
    pub layout: ImageDataLayout,
}

#[derive(Copy, Clone, Debug)]
pub struct ImageCopyView<'a> {
    pub image: &'a Image,
    pub mip_level: u32,
    pub origin: vk::Offset3D,
    pub aspect: vk::ImageAspectFlags,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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
        use_: ResourceUse,
    },
    Buffer {
        buffer: &'a Buffer,
        use_: ResourceUse,
        offset: u64,
        size: u64,
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
                use_: ResourceUse::SAMPLED_READ,
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
    pub buffer: &'a Buffer,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    _phantom: PhantomData<fn() -> T>,
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
                use_: ResourceUse::UNIFORM,
                offset: self.offset,
                size: self.range,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone, Debug)]
pub struct ReadOnlyStorageBuffer<'a, T: ?Sized> {
    pub buffer: &'a Buffer,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    _phantom: PhantomData<fn() -> T>,
}

unsafe impl<'a, T: ?Sized> Argument for ReadOnlyStorageBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.buffer,
                use_: ResourceUse::STORAGE_READ,
                offset: self.offset,
                size: self.range,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone, Debug)]
pub struct ReadWriteStorageBuffer<'a, T: ?Sized> {
    pub buffer: &'a Buffer,
    pub offset: vk::DeviceSize,
    pub range: vk::DeviceSize,
    _phantom: PhantomData<fn() -> T>,
}

unsafe impl<'a, T: ?Sized> Argument for ReadWriteStorageBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.buffer,
                use_: ResourceUse::STORAGE_READ_WRITE,
                offset: self.offset,
                size: self.range,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone, Debug)]
pub struct ReadWriteStorageImage<'a> {
    pub image_view: &'a ImageView,
}

unsafe impl<'a> Argument for ReadWriteStorageImage<'a> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_IMAGE;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
            kind: ArgumentKind::Image {
                image_view: self.image_view,
                use_: ResourceUse::IMAGE_READ_WRITE,
            },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Buffer {
    /// Byte range
    pub fn byte_range(&self, range: impl RangeBounds<u64>) -> BufferRangeAny {
        let byte_size = self.byte_size();
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => byte_size,
            Bound::Excluded(end) => *end,
            Bound::Included(end) => *end + 1,
        };
        let size = end - start;
        assert!(start <= byte_size && end <= byte_size);
        BufferRangeAny {
            buffer: self,
            offset: start,
            size,
        }
    }
}

/// Typed buffers.
pub struct TypedBuffer<T: ?Sized> {
    buffer: Buffer,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> TypedBuffer<T> {
    fn new(buffer: Buffer) -> Self {
        Self {
            buffer,
            _marker: PhantomData,
        }
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.buffer.byte_size()
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.buffer.usage()
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.buffer.handle()
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        self.buffer.device()
    }

    pub fn any(&self) -> &Buffer {
        &self.buffer
    }

    pub fn as_read_only_storage_buffer(&self) -> ReadOnlyStorageBuffer<T> {
        ReadOnlyStorageBuffer {
            buffer: &self.buffer,
            offset: 0,
            range: self.byte_size(),
            _phantom: PhantomData,
        }
    }

    pub fn as_read_write_storage_buffer(&self) -> ReadWriteStorageBuffer<T> {
        ReadWriteStorageBuffer {
            buffer: &self.buffer,
            offset: 0,
            range: self.byte_size(),
            _phantom: PhantomData,
        }
    }
}

impl<T> TypedBuffer<[T]> {
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        let len = self.byte_size() / mem::size_of::<T>() as u64;
        // I suppose this can fail on 32-bit platforms connected to GPUs with more than 4GB of VRAM
        len.try_into().expect("buffer too large")
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut T> {
        self.buffer.mapped_data().map(|ptr| ptr as *mut T)
    }

    /// Element range.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferRange<T> {
        let elem_size = mem::size_of::<T>();
        let start = match range.start_bound() {
            Bound::Unbounded => 0,
            Bound::Included(start) => *start,
            Bound::Excluded(start) => *start + 1,
        };
        let end = match range.end_bound() {
            Bound::Unbounded => self.len(),
            Bound::Excluded(end) => *end,
            Bound::Included(end) => *end + 1,
        };
        let start = (start * elem_size) as u64;
        let end = (end * elem_size) as u64;

        BufferRange {
            any: self.buffer.byte_range(start..end),
            _phantom: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BufferRangeAny<'a> {
    pub buffer: &'a Buffer,
    pub offset: u64,
    pub size: u64,
}

#[derive(Copy, Clone)]
pub struct BufferRange<'a, T> {
    any: BufferRangeAny<'a>,
    _phantom: std::marker::PhantomData<T>,
}

impl<'a, T> BufferRange<'a, T> {
    // TODO rename this to "untyped" or something
    pub fn any(&self) -> BufferRangeAny<'a> {
        self.any
    }

    pub fn as_read_only_storage_buffer(&self) -> ReadOnlyStorageBuffer<T> {
        ReadOnlyStorageBuffer {
            buffer: self.any.buffer,
            offset: self.any.offset,
            range: self.any.size,
            _phantom: PhantomData,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Describes a color, depth, or stencil attachment.
pub struct Attachment<'a> {
    pub image_view: &'a ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: Option<vk::ClearValue>,
}

impl<'a> Attachment<'a> {
    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color(mut self, color: [f32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { float32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color_int(mut self, color: [i32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { int32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear color of this attachment.
    pub fn clear_color_uint(mut self, color: [u32; 4]) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            color: vk::ClearColorValue { uint32: color },
        });
        self
    }

    /// Sets `load_op` to CLEAR, and sets the clear depth of this attachment.
    pub fn clear_depth(mut self, depth: f32) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
        });
        self
    }

    pub fn clear_depth_stencil(mut self, depth: f32, stencil: u32) -> Self {
        self.load_op = vk::AttachmentLoadOp::CLEAR;
        self.clear_value = Some(vk::ClearValue {
            depth_stencil: vk::ClearDepthStencilValue { depth, stencil },
        });
        self
    }

    /// Sets `load_op` to DONT_CARE.
    pub fn load_discard(mut self) -> Self {
        self.load_op = vk::AttachmentLoadOp::DONT_CARE;
        self
    }

    /// Sets `store_op` to DONT_CARE.
    pub fn store_discard(mut self) -> Self {
        self.store_op = vk::AttachmentStoreOp::DONT_CARE;
        self
    }
}

pub trait Attachments {
    /// Returns an iterator over the color attachments in this object.
    fn color_attachments(&self) -> impl Iterator<Item = Attachment<'_>> + '_;

    /// Returns the depth attachment.
    fn depth_attachment(&self) -> Option<Attachment>;

    /// Returns the stencil attachment.
    fn stencil_attachment(&self) -> Option<Attachment>;
}

/// Types that describe a color, depth, or stencil attachment to a rendering operation.
pub trait AsAttachment<'a> {
    /// Returns an object describing the attachment.
    fn as_attachment(&self) -> Attachment<'a>;
}

/// References to images can be used as attachments.
impl<'a> AsAttachment<'a> for &'a ImageView {
    fn as_attachment(&self) -> Attachment<'a> {
        Attachment {
            image_view: self,
            load_op: Default::default(),
            store_op: Default::default(),
            clear_value: None,
        }
    }
}

#[doc(hidden)]
pub struct AttachmentOverride<A>(
    pub A,
    pub Option<vk::AttachmentLoadOp>,
    pub Option<vk::AttachmentStoreOp>,
    pub Option<vk::ClearValue>,
);

impl<'a, A: AsAttachment<'a>> AsAttachment<'a> for AttachmentOverride<A> {
    fn as_attachment(&self) -> Attachment<'a> {
        let mut desc = self.0.as_attachment();
        if let Some(load_op) = self.1 {
            desc.load_op = load_op;
        }
        if let Some(store_op) = self.2 {
            desc.store_op = store_op;
        }
        if let Some(clear_value) = self.3 {
            desc.clear_value = Some(clear_value);
        }
        desc
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone, Debug)]
pub struct VertexBufferDescriptor<'a> {
    pub binding: u32,
    pub buffer_range: BufferRangeAny<'a>,
    pub stride: u32,
}

pub trait VertexInput {
    /// Vertex buffer bindings
    fn buffer_layout(&self) -> Cow<[VertexBufferLayoutDescription]>;

    /// Vertex attributes.
    fn attributes(&self) -> Cow<[VertexInputAttributeDescription]>;

    /// Returns an iterator over the vertex buffers referenced in this object.
    fn vertex_buffers(&self) -> impl Iterator<Item = VertexBufferDescriptor<'_>>;
}

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferView<T: Vertex> {
    pub buffer: vk::Buffer,
    pub offset: vk::DeviceSize,
    pub _phantom: PhantomData<*const T>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies the code of a shader.
#[derive(Debug, Clone, Copy)]
pub enum ShaderCode<'a> {
    /// Compile the shader from the specified source.
    Source(ShaderSource<'a>),
    /// Create the shader from the specified SPIR-V binary.
    Spirv(&'a [u32]),
}

/// Shader code + entry point
#[derive(Debug, Clone, Copy)]
pub struct ShaderEntryPoint<'a> {
    pub code: ShaderCode<'a>,
    pub entry_point: &'a str,
}

/// Specifies the shaders of a graphics pipeline.
pub enum PreRasterizationShaders {
    /// Shaders of the primitive shading pipeline (the classic vertex, tessellation, geometry and fragment shaders).
    PrimitiveShading {
        vertex: ShaderEntryPoint<'static>,
        tess_control: Option<ShaderEntryPoint<'static>>,
        tess_evaluation: Option<ShaderEntryPoint<'static>>,
        geometry: Option<ShaderEntryPoint<'static>>,
    },
    /// Shaders of the mesh shading pipeline (the new mesh and task shaders).
    MeshShading {
        task: Option<ShaderEntryPoint<'static>>,
        mesh: ShaderEntryPoint<'static>,
    },
}

pub struct GraphicsPipelineCreateInfo<'a> {
    pub layout: PipelineLayoutDescriptor<'a>,
    pub vertex_input: VertexInputState<'a>,
    pub pre_rasterization_shaders: PreRasterizationShaders,
    pub rasterization: RasterizationState,
    pub fragment_shader: ShaderEntryPoint<'static>,
    pub depth_stencil: DepthStencilState,
    pub fragment_output: FragmentOutputInterfaceDescriptor<'a>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Specifies the type of a graphics pipeline with the given vertex input, arguments and push constants.
#[macro_export]
macro_rules! graphics_pipeline_interface {
    [
        $(#[$meta:meta])*
        $v:vis struct $name:ident {
            arguments {
                $(
                    $(#[$arg_meta:meta])*
                    $arg_binding:literal => $arg_method:ident($arg_ty:ty)
                ),*
            }

            $(push_constants {
                $(#[$push_cst_meta:meta])*
                $push_constants_method:ident($push_constants_ty:ty)
            })?

            $(output_attachments($attachments_ty:ty))?
        }
    ] => {
        $(#[$meta])*
        $v struct $name<'a> {
            p: crate::encoder::RenderEncoder<'a>,
        }

        impl<'a> $name<'a> {
            $(
                $(#[$arg_meta])*
                pub fn $arg_method(&mut self, arg: &$arg_ty) {
                    unsafe {
                        self.p.bind_arguments($arg_binding, &arg)
                    }
                }
            )*

            $(
                $(#[$push_cst_meta])*
                pub fn $push_constants_method(&mut self, push_constants: &$push_constants_ty) {
                    unsafe {
                        self.p.bind_push_constants(push_constants)
                    }
                }
            )?
        }

        impl<'a> crate::device::StaticPipelineInterface for $name<'a> {
            const ARGUMENTS: &'static [&'static crate::argument::ArgumentsLayout<'static>] = &[
                $(
                    &$arg_ty::LAYOUT
                ),*
            ];

            const PUSH_CONSTANTS: &'static [crate::vk::PushConstantRange] = graphics_pipeline_interface!(@push_constants $($push_constants_ty)?);

            type Attachments = graphics_pipeline_interface!(@attachments $($attachments_ty)?);
        }
    };

    //------------------------------------------------

    // No push constants -> empty constant ranges
    (@push_constants ) => { &[] };
    (@push_constants $t:ty) => { <$t as $crate::argument::StaticPushConstants>::PUSH_CONSTANT_RANGES };

    // Default attachment type if left unspecified
    (@attachments ) => { () };
    (@attachments $t:ty) => { $t };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

fn is_write_access(mask: vk::AccessFlags2) -> bool {
    // TODO: this is not exhaustive
    mask.intersects(
        vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE
            | vk::AccessFlags2::HOST_WRITE
            | vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
            | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV,
    )
}

/// Computes the number of mip levels for a 2D image of the given size.
///
/// # Examples
///
/// ```
/// use graal::mip_level_count;
/// assert_eq!(mip_level_count(512, 512), 9);
/// assert_eq!(mip_level_count(512, 256), 9);
/// assert_eq!(mip_level_count(511, 256), 8);
/// ```
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT
    )
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, vk::Format::S8_UINT)
}

pub fn aspects_for_format(fmt: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_only_format(fmt) {
        vk::ImageAspectFlags::DEPTH
    } else if is_stencil_only_format(fmt) {
        vk::ImageAspectFlags::STENCIL
    } else if is_depth_and_stencil_format(fmt) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

// Implementation detail of the VertexInput macro
#[doc(hidden)]
pub const fn append_attributes<const N: usize>(
    head: &'static [VertexInputAttributeDescription],
    binding: u32,
    base_location: u32,
    tail: &'static [VertexAttributeDescription],
) -> [VertexInputAttributeDescription; N] {
    const NULL_ATTR: VertexInputAttributeDescription = VertexInputAttributeDescription {
        location: 0,
        binding: 0,
        format: vk::Format::UNDEFINED,
        offset: 0,
    };
    let mut result = [NULL_ATTR; N];
    let mut i = 0;
    while i < head.len() {
        result[i] = head[i];
        i += 1;
    }
    while i < N {
        let j = i - head.len();
        result[i] = VertexInputAttributeDescription {
            location: base_location + j as u32,
            binding,
            format: tail[j].format,
            offset: tail[j].offset,
        };
        i += 1;
    }

    result
}
