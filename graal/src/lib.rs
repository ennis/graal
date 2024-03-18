mod command;
mod device;
mod instance;
mod platform;
mod platform_impl;
mod shader;
mod surface;
mod tracker;
mod types;
pub mod util;

use ash::vk::Handle;
use fxhash::FxHashMap;
use std::{
    borrow::Cow,
    convert::TryInto,
    marker::PhantomData,
    mem,
    ops::{Bound, RangeBounds},
    os::raw::c_void,
    path::Path,
    ptr::NonNull,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};
use tracing::debug;

// --- reexports ---

// TODO: make it optional
pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;
pub use shaderc;
// reexports for macro internals
#[doc(hidden)]
pub use memoffset::offset_of as __offset_of;
#[doc(hidden)]
pub use memoffset::offset_of_tuple as __offset_of_tuple;
pub use ordered_float;
pub use shader::{compile_shader, get_shader_compiler};

pub use command::*;
pub use device::*;
pub use instance::*;
pub use surface::*;
pub use types::*;
// proc-macros
pub use graal_macros::{Arguments, Attachments, Vertex};

pub mod prelude {
    pub use crate::{
        util::{CommandStreamExt, DeviceExt},
        vk, Arguments, Attachments, Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState,
        CommandStream, CompareOp, ComputeEncoder, DepthStencilState, Device, Format, FragmentOutputInterfaceDescriptor,
        FrontFace, GraphicsPipeline, GraphicsPipelineCreateInfo, Image, ImageCreateInfo, ImageType, ImageUsage,
        ImageView, IndexType, LineRasterization, LineRasterizationMode, MemoryLocation, PipelineBindPoint,
        PipelineLayoutDescriptor, Point2D, PolygonMode, PreRasterizationShaders, PrimitiveTopology, RasterizationState,
        Rect2D, RenderEncoder, Sampler, SamplerCreateInfo, ShaderCode, ShaderEntryPoint, ShaderSource, Size2D,
        StaticArguments, StaticAttachments, StencilState, Vertex, VertexBufferDescriptor,
        VertexBufferLayoutDescription, VertexInputAttributeDescription, VertexInputRate, VertexInputState,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Represents a swap chain.
#[derive(Debug)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub surface: vk::SurfaceKHR,
    pub format: vk::SurfaceFormatKHR,
    pub width: u32,
    pub height: u32,
    pub images: Vec<vk::Image>,
}

/// Contains information about an image in a swapchain.
#[derive(Debug)]
pub struct SwapchainImage {
    /// Handle of the swapchain that owns this image.
    pub swapchain: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub index: u32,
    pub image: Image,
}

/// Graphics pipelines.
pub struct GraphicsPipeline {
    pub(crate) device: Device,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub(crate) fn new(device: Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
        }
    }

    pub fn set_label(&self, label: &str) {
        self.device
            .set_debug_object_name(vk::ObjectType::PIPELINE, self.pipeline.as_raw(), label, None);
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Compute pipelines.
pub struct ComputePipeline {
    pub(crate) device: Device,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
}

impl ComputePipeline {
    pub(crate) fn new(device: Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
        }
    }

    pub fn set_label(&self, label: &str) {
        self.device
            .set_debug_object_name(vk::ObjectType::PIPELINE, self.pipeline.as_raw(), label, None);
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Samplers
#[derive(Clone, Debug)]
pub struct Sampler {
    // A weak ref is sufficient, the device already owns samplers in its cache
    device: WeakDevice,
    sampler: vk::Sampler,
}

impl Sampler {
    pub(crate) fn new(device: &Device, sampler: vk::Sampler) -> Sampler {
        Sampler {
            device: device.weak(),
            sampler,
        }
    }

    pub fn set_label(&self, label: &str) {
        self.device.upgrade().unwrap().set_debug_object_name(
            vk::ObjectType::SAMPLER,
            self.sampler.as_raw(),
            label,
            None,
        );
    }

    pub fn handle(&self) -> vk::Sampler {
        // FIXME: check if the device is still alive, otherwise the sampler isn't valid anymore
        //assert!(self.device.strong_count() > 0);
        self.sampler
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
struct CommandPool {
    queue_family: u32,
    command_pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    used: Vec<vk::CommandBuffer>,
}

impl CommandPool {
    unsafe fn new(device: &ash::Device, queue_family_index: u32) -> CommandPool {
        // create a new one
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index,
            ..Default::default()
        };
        let command_pool = device
            .create_command_pool(&create_info, None)
            .expect("failed to create a command pool");

        CommandPool {
            queue_family: queue_family_index,
            command_pool,
            free: vec![],
            used: vec![],
        }
    }

    fn alloc(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        let cb = self.free.pop().unwrap_or_else(|| unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = device
                .allocate_command_buffers(&allocate_info)
                .expect("failed to allocate command buffers");
            buffers[0]
        });
        self.used.push(cb);
        cb
    }

    unsafe fn reset(&mut self, device: &ash::Device) {
        device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
        self.free.append(&mut self.used)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ResourceMapsWithAccess {
    pub buffers: FxHashMap<BufferId, (Arc<BufferInner>, BufferAccess)>,
    pub images: FxHashMap<ImageId, (Arc<ImageInner>, ImageAccess)>,
    pub image_views: FxHashMap<ImageViewId, Arc<ImageViewInner>>,
}

impl ResourceMapsWithAccess {
    fn new() -> Self {
        Self {
            buffers: FxHashMap::default(),
            images: FxHashMap::default(),
            image_views: FxHashMap::default(),
        }
    }
}

#[derive(Default)]
struct ResourceMaps {
    pub buffers: FxHashMap<BufferId, Arc<BufferInner>>,
    pub images: FxHashMap<ImageId, Arc<ImageInner>>,
    pub image_views: FxHashMap<ImageViewId, Arc<ImageViewInner>>,
}

impl ResourceMaps {
    fn new() -> Self {
        Self {
            buffers: FxHashMap::default(),
            images: FxHashMap::default(),
            image_views: FxHashMap::default(),
        }
    }
}

impl ResourceMaps {
    fn insert<R: Resource>(&mut self, resource: Arc<R>) {
        R::insert(self, resource)
    }
}

fn make_buffer_barrier(buffer: vk::Buffer, src: BufferAccess, dst: BufferAccess) -> vk::BufferMemoryBarrier2 {
    let (src_stage_mask, src_access_mask) = map_buffer_access_to_barrier(src);
    let (dst_stage_mask, dst_access_mask) = map_buffer_access_to_barrier(dst);
    vk::BufferMemoryBarrier2 {
        src_stage_mask,
        src_access_mask,
        dst_stage_mask,
        dst_access_mask,
        buffer,
        offset: 0,
        size: vk::WHOLE_SIZE,
        ..Default::default()
    }
}

fn make_image_barrier(
    image: vk::Image,
    format: vk::Format,
    src: ImageAccess,
    dst: ImageAccess,
) -> vk::ImageMemoryBarrier2 {
    let (src_stage_mask, src_access_mask) = map_image_access_to_barrier(src);
    let (dst_stage_mask, dst_access_mask) = map_image_access_to_barrier(dst);

    let src_layout = map_image_access_to_layout(src, format);
    let dst_layout = map_image_access_to_layout(dst, format);

    vk::ImageMemoryBarrier2 {
        src_stage_mask,
        src_access_mask,
        dst_stage_mask,
        dst_access_mask,
        old_layout: src_layout,
        new_layout: dst_layout,
        image,
        subresource_range: vk::ImageSubresourceRange {
            aspect_mask: aspects_for_format(format),
            base_mip_level: 0,
            level_count: vk::REMAINING_MIP_LEVELS,
            base_array_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
        },
        ..Default::default()
    }
}

trait Resource {
    type Id;
    fn insert(maps: &mut ResourceMaps, resource: Arc<Self>);
    fn remove(&self, maps: &mut ResourceMaps);
    fn submission_index(&self) -> u64;
}

#[derive(Debug)]
struct BufferInner {
    device: Device,
    id: BufferId,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    group: Option<GroupId>,
    handle: vk::Buffer,
    device_address: vk::DeviceAddress,
}

impl Resource for BufferInner {
    type Id = BufferId;

    fn insert(maps: &mut ResourceMaps, resource: Arc<Self>) {
        maps.buffers.insert(resource.id, resource);
    }

    fn remove(&self, maps: &mut ResourceMaps) {
        maps.buffers.remove(&self.id);
    }

    fn submission_index(&self) -> u64 {
        self.last_submission_index.load(Ordering::Relaxed)
    }
}

impl BufferInner {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.last_submission_index.store(submission_index, Ordering::Release);
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: The device resource tracker holds strong references to resources as long as they are in use by the GPU.
        // This prevents `drop` from being called while the resource is still in use, and thus it's safe to delete the
        // resource here.
        unsafe {
            debug!("dropping buffer {:?} (handle: {:?})", self.id, self.handle);
            self.device.free_memory(&mut self.allocation);
            self.device.destroy_buffer(self.handle, None);
        }
    }
}

/// Wrapper around a Vulkan buffer.
#[derive(Debug)]
pub struct BufferUntyped {
    inner: Arc<BufferInner>,
    handle: vk::Buffer,
    size: u64,
    usage: BufferUsage,
    mapped_ptr: Option<NonNull<c_void>>,
}

impl Drop for BufferUntyped {
    fn drop(&mut self) {
        self.inner.device.drop_buffer(&self.inner)
    }
}

impl BufferUntyped {
    pub fn set_label(&self, label: &str) {
        self.inner
            .device
            .set_debug_object_name(vk::ObjectType::BUFFER, self.handle.as_raw(), label, None);
    }

    pub(crate) fn id(&self) -> BufferId {
        self.inner.id
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.size
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        &self.inner.device
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr.as_ptr() as *mut u8)
    }
}

#[derive(Debug)]
struct ImageInner {
    device: Device,
    id: ImageId,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    group: Option<GroupId>,
    handle: vk::Image,
    format: vk::Format,
    swapchain_image: bool,
}

impl ImageInner {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.last_submission_index.store(submission_index, Ordering::Release);
    }
}

impl Resource for ImageInner {
    type Id = ImageId;

    fn insert(maps: &mut ResourceMaps, resource: Arc<Self>) {
        maps.images.insert(resource.id, resource);
    }

    fn remove(&self, maps: &mut ResourceMaps) {
        maps.images.remove(&self.id);
    }

    fn submission_index(&self) -> u64 {
        self.last_submission_index.load(Ordering::Relaxed)
    }
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if !self.swapchain_image {
            unsafe {
                debug!("dropping image {:?} (handle: {:?})", self.id, self.handle);
                self.device.free_memory(&mut self.allocation);
                self.device.destroy_image(self.handle, None);
            }
        }
    }
}

/// Wrapper around a Vulkan image.
#[derive(Debug)]
pub struct Image {
    inner: Arc<ImageInner>,
    handle: vk::Image,
    usage: ImageUsage,
    type_: ImageType,
    format: Format,
    size: Size3D,
}

impl Drop for Image {
    fn drop(&mut self) {
        self.inner.device.drop_image(&self.inner)
    }
}

impl Image {
    pub fn set_label(&self, label: &str) {
        self.inner
            .device
            .set_debug_object_name(vk::ObjectType::IMAGE, self.handle.as_raw(), label, None);
    }

    /// Returns the `vk::ImageType` of the image.
    pub fn image_type(&self) -> ImageType {
        self.type_
    }

    /// Returns the `vk::Format` of the image.
    pub fn format(&self) -> Format {
        self.format
    }

    /// Returns the `vk::Extent3D` of the image.
    pub fn size(&self) -> Size3D {
        self.size
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn depth(&self) -> u32 {
        self.size.depth
    }

    /// Returns the usage flags of the image.
    pub fn usage(&self) -> ImageUsage {
        self.usage
    }

    pub fn id(&self) -> ImageId {
        self.inner.id
    }

    /// Returns the image handle.
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    /// Creates an image view for the base mip level of this image,
    /// suitable for use as a rendering attachment.
    pub fn create_top_level_view(&self) -> ImageView {
        self.create_view(&ImageViewInfo {
            view_type: match self.image_type() {
                ImageType::Image2D => vk::ImageViewType::TYPE_2D,
                _ => panic!("unsupported image type for attachment"),
            },
            format: self.format(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: aspects_for_format(self.format()),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            component_mapping: [
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
            ],
        })
    }

    /// Creates an `ImageView` object.
    fn create_view(&self, info: &ImageViewInfo) -> ImageView {
        self.inner.device.create_image_view(self, info)
    }
}

#[derive(Debug)]
struct ImageViewInner {
    image: Arc<ImageInner>,
    id: ImageViewId,
    handle: vk::ImageView,
    last_submission_index: AtomicU64,
}

impl ImageViewInner {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.last_submission_index.store(submission_index, Ordering::Release);
    }
}

impl Resource for ImageViewInner {
    type Id = ImageViewId;

    fn insert(maps: &mut ResourceMaps, resource: Arc<Self>) {
        maps.image_views.insert(resource.id, resource);
    }

    fn remove(&self, maps: &mut ResourceMaps) {
        maps.image_views.remove(&self.id);
    }

    fn submission_index(&self) -> u64 {
        self.last_submission_index.load(Ordering::Relaxed)
    }
}

impl Drop for ImageViewInner {
    fn drop(&mut self) {
        unsafe {
            self.image.device.destroy_image_view(self.handle, None);
        }
    }
}

#[derive(Debug)]
pub struct ImageView {
    inner: Arc<ImageViewInner>,
    image: ImageId,
    image_handle: vk::Image,
    handle: vk::ImageView,
    format: Format,
    original_format: Format,
    size: Size3D,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        self.inner.image.device.drop_image_view(&self.inner)
    }
}

impl ImageView {
    /// Returns the format of the image view.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn size(&self) -> Size3D {
        self.size
    }

    pub fn width(&self) -> u32 {
        self.size.width
    }

    pub fn height(&self) -> u32 {
        self.size.height
    }

    pub fn handle(&self) -> vk::ImageView {
        self.handle
    }

    pub fn set_label(&self, label: &str) {
        self.inner
            .image
            .device
            .set_debug_object_name(vk::ObjectType::IMAGE_VIEW, self.handle.as_raw(), label, None);
    }

    fn image_handle(&self) -> vk::Image {
        self.image_handle
    }

    fn original_format(&self) -> vk::Format {
        self.original_format
    }

    fn image_id(&self) -> ImageId {
        self.image.clone()
    }
}

/*
/// Wrapper around a descriptor buffer
pub struct ArgumentBufferUntyped {
    descriptor_buffer: BufferUntyped,
    tracker: ArgumentBufferTracker,
}*/

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
    pub buffer: &'a BufferUntyped,
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
        access: ImageAccess,
    },
    Buffer {
        buffer: &'a BufferUntyped,
        access: BufferAccess,
        offset: u64,
        size: u64,
    },
    Sampler {
        sampler: &'a Sampler,
    },
}

impl<'a> From<(&'a ImageView, ImageAccess)> for ArgumentKind<'a> {
    fn from((image_view, access): (&'a ImageView, ImageAccess)) -> Self {
        ArgumentKind::Image { image_view, access }
    }
}

impl<'a, T: ?Sized> From<(BufferRange<'a, T>, BufferAccess)> for ArgumentKind<'a> {
    fn from((buffer_range, access): (BufferRange<'a, T>, BufferAccess)) -> Self {
        ArgumentKind::Buffer {
            buffer: buffer_range.untyped.buffer,
            access,
            offset: buffer_range.untyped.offset,
            size: buffer_range.untyped.size,
        }
    }
}

impl<'a> From<(BufferRangeUntyped<'a>, BufferAccess)> for ArgumentKind<'a> {
    fn from((buffer_range, access): (BufferRangeUntyped<'a>, BufferAccess)) -> Self {
        ArgumentKind::Buffer {
            buffer: buffer_range.buffer,
            access,
            offset: buffer_range.offset,
            size: buffer_range.size,
        }
    }
}

impl<'a, T: ?Sized> From<(&'a Buffer<T>, BufferAccess)> for ArgumentKind<'a> {
    fn from((buffer, access): (&'a Buffer<T>, BufferAccess)) -> Self {
        ArgumentKind::Buffer {
            buffer: &buffer.untyped,
            access,
            offset: 0,
            size: buffer.untyped.size,
        }
    }
}

impl<'a> From<(&'a BufferUntyped, BufferAccess)> for ArgumentKind<'a> {
    fn from((buffer, access): (&'a BufferUntyped, BufferAccess)) -> Self {
        ArgumentKind::Buffer {
            buffer: &buffer,
            access,
            offset: 0,
            size: buffer.size,
        }
    }
}

impl<'a> From<&'a Sampler> for ArgumentKind<'a> {
    fn from(sampler: &'a Sampler) -> Self {
        ArgumentKind::Sampler { sampler }
    }
}

pub trait Arguments {
    /// The type of inline data for this argument.
    type InlineData: Copy + 'static;

    /// Returns an iterator over all descriptors contained in this object.
    fn arguments(&self) -> impl Iterator<Item = ArgumentDescription> + '_;

    /// Returns the inline data for this argument.
    fn inline_data(&self) -> Cow<Self::InlineData>;
}

pub trait Argument {
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

/*
#[derive(Copy, Clone, Debug)]
pub enum ImageDescriptorType {
    SampledImage,
    ReadOnlyStorageImage,
    ReadWriteStorageImage,
}

impl ImageDescriptorType {
    pub const fn access(self) -> ImageAccess {
        match self {
            ImageDescriptorType::SampledImage => ImageAccess::SAMPLED_READ,
            ImageDescriptorType::ReadOnlyStorageImage => ImageAccess::IMAGE_READ,
            ImageDescriptorType::ReadWriteStorageImage => ImageAccess::IMAGE_READ_WRITE,
        }
    }

    pub const fn to_vk_descriptor_type(self) -> vk::DescriptorType {
        match self {
            ImageDescriptorType::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            ImageDescriptorType::ReadOnlyStorageImage => vk::DescriptorType::STORAGE_IMAGE,
            ImageDescriptorType::ReadWriteStorageImage => vk::DescriptorType::STORAGE_IMAGE,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub enum BufferDescriptorType {
    UniformBuffer,
    ReadOnlyStorageBuffer,
    ReadWriteStorageBuffer,
}

impl BufferDescriptorType {
    pub const fn access(self) -> BufferAccess {
        match self {
            BufferDescriptorType::UniformBuffer => BufferAccess::UNIFORM,
            BufferDescriptorType::ReadOnlyStorageBuffer => BufferAccess::STORAGE_READ,
            BufferDescriptorType::ReadWriteStorageBuffer => BufferAccess::STORAGE_READ_WRITE,
        }
    }

    pub const fn to_vk_descriptor_type(self) -> vk::DescriptorType {
        match self {
            BufferDescriptorType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            BufferDescriptorType::ReadOnlyStorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            BufferDescriptorType::ReadWriteStorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
        }
    }
}*/

#[doc(hidden)]
pub struct ImageArgumentWrapper<'a, const DESCRIPTOR_TYPE: i32, const ACCESS: u32, const SHADER_STAGES: u32>(
    pub &'a ImageView,
);
#[doc(hidden)]
pub struct BufferArgumentWrapper<'a, const DESCRIPTOR_TYPE: i32, const ACCESS: u32, const SHADER_STAGES: u32, T>(
    pub BufferRange<'a, T>,
);

impl<'a, const DESCRIPTOR_TYPE: i32, const ACCESS: u32, const SHADER_STAGES: u32> Argument
    for ImageArgumentWrapper<'a, DESCRIPTOR_TYPE, ACCESS, SHADER_STAGES>
{
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::from_raw(DESCRIPTOR_TYPE);
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(SHADER_STAGES);

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: Self::DESCRIPTOR_TYPE,
            kind: ArgumentKind::Image {
                image_view: self.0,
                access: ImageAccess::from_bits(ACCESS).unwrap(),
            },
        }
    }
}

impl<'a, const DESCRIPTOR_TYPE: i32, const ACCESS: u32, const SHADER_STAGES: u32, T> Argument
    for BufferArgumentWrapper<'a, DESCRIPTOR_TYPE, ACCESS, SHADER_STAGES, T>
{
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::from_raw(DESCRIPTOR_TYPE);
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::from_raw(SHADER_STAGES);

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: Self::DESCRIPTOR_TYPE,
            kind: ArgumentKind::Buffer {
                buffer: self.0.untyped.buffer,
                access: BufferAccess::from_bits(ACCESS).unwrap(),
                offset: self.0.untyped.offset,
                size: self.0.untyped.size,
            },
        }
    }
}

impl<'a> Argument for &'a Sampler {
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

/*
/// Uniform buffer descriptor.
#[derive(Copy, Clone)]
pub struct UniformBuffer<'a, T>(pub BufferRange<'a, T>);

unsafe impl<'a, T> Argument for UniformBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::UNIFORM_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.0.untyped.buffer,
                access: BufferAccess::UNIFORM,
                offset: self.0.untyped.offset,
                size: self.0.untyped.size,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone)]
pub struct ReadOnlyStorageBuffer<'a, T: ?Sized>(pub BufferRange<'a, T>);

unsafe impl<'a, T: ?Sized> Argument for ReadOnlyStorageBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.0.untyped.buffer,
                access: BufferAccess::STORAGE_READ,
                offset: self.0.untyped.offset,
                size: self.0.untyped.size,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone)]
pub struct ReadWriteStorageBuffer<'a, T: ?Sized>(pub BufferRange<'a, T>);

unsafe impl<'a, T: ?Sized> Argument for ReadWriteStorageBuffer<'a, T> {
    const DESCRIPTOR_TYPE: vk::DescriptorType = vk::DescriptorType::STORAGE_BUFFER;
    const DESCRIPTOR_COUNT: u32 = 1;
    const SHADER_STAGES: vk::ShaderStageFlags = vk::ShaderStageFlags::ALL;

    fn argument_description(&self, binding: u32) -> ArgumentDescription {
        ArgumentDescription {
            binding,
            descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
            kind: ArgumentKind::Buffer {
                buffer: self.0.untyped.buffer,
                access: BufferAccess::STORAGE_READ_WRITE,
                offset: self.0.untyped.offset,
                size: self.0.untyped.size,
            },
        }
    }
}

/// Storage buffer descriptor.
#[derive(Copy, Clone, Debug)]
pub struct ReadWriteStorageImage<'a> {
    pub image_view: &'a ImageView,
}

impl<'a> ReadWriteStorageImage<'a> {
    pub fn new(image_view: &'a ImageView) -> Self {
        Self { image_view }
    }
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
                access: ImageAccess::IMAGE_READ_WRITE,
            },
        }
    }
}
*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl BufferUntyped {
    /// Byte range
    pub fn byte_range(&self, range: impl RangeBounds<u64>) -> BufferRangeUntyped {
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
        BufferRangeUntyped {
            buffer: self,
            offset: start,
            size,
        }
    }
}

/// Typed buffers.
pub struct Buffer<T: ?Sized> {
    pub untyped: BufferUntyped,
    _marker: PhantomData<T>,
}

impl<T: ?Sized> Buffer<T> {
    fn new(buffer: BufferUntyped) -> Self {
        Self {
            untyped: buffer,
            _marker: PhantomData,
        }
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.untyped.byte_size()
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.untyped.usage()
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.untyped.handle()
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        self.untyped.device()
    }
}

impl<T> Buffer<[T]> {
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        (self.byte_size() / mem::size_of::<T>() as u64) as usize
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut T> {
        self.untyped.mapped_data().map(|ptr| ptr as *mut T)
    }

    /// Element range.
    pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferRange<[T]> {
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
            untyped: self.untyped.byte_range(start..end),
            _phantom: PhantomData,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BufferRangeUntyped<'a> {
    pub buffer: &'a BufferUntyped,
    pub offset: u64,
    pub size: u64,
}

pub struct BufferRange<'a, T: ?Sized> {
    pub untyped: BufferRangeUntyped<'a>,
    _phantom: PhantomData<T>,
}

// #26925 clone impl
impl<'a, T: ?Sized> Clone for BufferRange<'a, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<'a, T: ?Sized> Copy for BufferRange<'a, T> {}

impl<'a, T> BufferRange<'a, [T]> {
    pub fn len(&self) -> usize {
        (self.untyped.size / mem::size_of::<T>() as u64) as usize
    }

    /*pub fn slice(&self, range: impl RangeBounds<usize>) -> BufferRange<'a, [T]> {
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
            untyped: BufferRangeAny {
                buffer: self.untyped.buffer,
                offset: self.untyped.offset + start,
                size: end - start,
            },
            _phantom: PhantomData,
        }
    }*/
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
    fn depth_stencil_attachment(&self) -> Option<Attachment>;
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
    pub buffer_range: BufferRangeUntyped<'a>,
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

impl<'a> ShaderEntryPoint<'a> {
    pub fn from_source_file(file_path: &'a Path) -> ShaderEntryPoint<'a> {
        Self {
            code: ShaderCode::Source(ShaderSource::File(file_path)),
            entry_point: "main",
        }
    }
}

/// Specifies the shaders of a graphics pipeline.
pub enum PreRasterizationShaders<'a> {
    /// Shaders of the primitive shading pipeline (the classic vertex, tessellation, geometry and fragment shaders).
    PrimitiveShading {
        vertex: ShaderEntryPoint<'a>,
        tess_control: Option<ShaderEntryPoint<'a>>,
        tess_evaluation: Option<ShaderEntryPoint<'a>>,
        geometry: Option<ShaderEntryPoint<'a>>,
    },
    /// Shaders of the mesh shading pipeline (the new mesh and task shaders).
    MeshShading {
        task: Option<ShaderEntryPoint<'a>>,
        mesh: ShaderEntryPoint<'a>,
    },
}

impl<'a> PreRasterizationShaders<'a> {
    /// Creates a new `PreRasterizationShaders` object using mesh shading from the specified source file path.
    ///
    /// The specified source file should contain both task and mesh shaders. The entry point for both shaders is `main`.
    /// Use the `__TASK__` and `__MESH__` macros to distinguish between the two shaders within the source file.
    pub fn mesh_shading_from_source_file(file_path: &'a Path) -> Self {
        let entry_point = "main";
        Self::MeshShading {
            task: Some(ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            }),
            mesh: ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            },
        }
    }

    /// Creates a new `PreRasterizationShaders` object using primitive shading, without tessellation, from the specified source file path.
    pub fn vertex_shader_from_source_file(file_path: &'a Path) -> Self {
        let entry_point = "main";
        Self::PrimitiveShading {
            vertex: ShaderEntryPoint {
                code: ShaderCode::Source(ShaderSource::File(file_path)),
                entry_point,
            },
            tess_control: None,
            tess_evaluation: None,
            geometry: None,
        }
    }
}

pub struct GraphicsPipelineCreateInfo<'a> {
    pub layout: PipelineLayoutDescriptor<'a>,
    pub vertex_input: VertexInputState<'a>,
    pub pre_rasterization_shaders: PreRasterizationShaders<'a>,
    pub rasterization: RasterizationState,
    pub fragment_shader: ShaderEntryPoint<'a>,
    pub depth_stencil: DepthStencilState,
    pub fragment_output: FragmentOutputInterfaceDescriptor<'a>,
}

pub struct ComputePipelineCreateInfo<'a> {
    pub layout: PipelineLayoutDescriptor<'a>,
    pub compute_shader: ShaderEntryPoint<'a>,
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

fn map_buffer_access_to_barrier(state: BufferAccess) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if state.contains(BufferAccess::MAP_READ) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_READ;
    }
    if state.contains(BufferAccess::MAP_WRITE) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_WRITE;
    }
    if state.contains(BufferAccess::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if state.contains(BufferAccess::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if state.contains(BufferAccess::UNIFORM) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::UNIFORM_READ;
    }
    if state.intersects(BufferAccess::STORAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.intersects(BufferAccess::STORAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }
    if state.contains(BufferAccess::INDEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::INDEX_READ;
    }
    if state.contains(BufferAccess::VERTEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }
    if state.contains(BufferAccess::INDIRECT) {
        stages |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        access |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    (stages, access)
}

fn map_image_access_to_barrier(state: ImageAccess) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if state.contains(ImageAccess::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if state.contains(ImageAccess::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if state.contains(ImageAccess::SAMPLED_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.contains(ImageAccess::COLOR_TARGET) {
        stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
    }
    if state.intersects(ImageAccess::DEPTH_STENCIL_READ) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if state.intersects(ImageAccess::DEPTH_STENCIL_WRITE) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if state.contains(ImageAccess::IMAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if state.contains(ImageAccess::IMAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }

    if state == ImageAccess::UNINITIALIZED || state == ImageAccess::PRESENT {
        (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty())
    } else {
        (stages, access)
    }
}

fn map_image_access_to_layout(access: ImageAccess, format: Format) -> vk::ImageLayout {
    let is_color = aspects_for_format(format).contains(vk::ImageAspectFlags::COLOR);
    match access {
        ImageAccess::UNINITIALIZED => vk::ImageLayout::UNDEFINED,
        ImageAccess::COPY_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ImageAccess::COPY_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ImageAccess::SAMPLED_READ if is_color => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ImageAccess::COLOR_TARGET => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ImageAccess::DEPTH_STENCIL_WRITE => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        _ => {
            if access == ImageAccess::PRESENT {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else if is_color {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
        }
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
