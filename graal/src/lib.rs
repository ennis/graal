//mod bindless;
mod command;
mod device;
mod instance;
mod platform;
mod platform_impl;
mod shader;
mod surface;
mod types;
pub mod util;

use std::{
    borrow::Cow,
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
pub use graal_macros::Vertex;

pub mod prelude {
    pub use crate::{
        util::{CommandStreamExt, DeviceExt},
        vk, Buffer, BufferUsage, ClearColorValue, ColorBlendEquation, ColorTargetState, CommandStream, ComputeEncoder,
        DepthStencilState, Device, Format, FragmentState, GraphicsPipeline, GraphicsPipelineCreateInfo, Image,
        ImageCreateInfo, ImageType, ImageUsage, ImageView, MemoryLocation, PipelineBindPoint, PipelineLayoutDescriptor,
        Point2D, PreRasterizationShaders, RasterizationState, Rect2D, RenderEncoder, Sampler, SamplerCreateInfo,
        ShaderCode, ShaderEntryPoint, ShaderSource, Size2D, StencilState, Vertex, VertexBufferDescriptor,
        VertexBufferLayoutDescription, VertexInputAttributeDescription, VertexInputState,
    };
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Standard subgroup size.
pub const SUBGROUP_SIZE: u32 = 32;

/// Device address of a GPU buffer.
#[derive(Copy, Clone, Debug, Ord, PartialOrd, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct DeviceAddressUntyped {
    pub address: vk::DeviceAddress,
}

/// Device address of a GPU buffer containing elements of type `T` its associated type.
///
/// The type should be `T: Copy` for a buffer containing a single element of type T,
/// or `[T] where T: Copy` for slices of elements of type T.
#[repr(transparent)]
pub struct DeviceAddress<T: ?Sized + 'static> {
    pub address: vk::DeviceAddress,
    pub _phantom: PhantomData<T>,
}

impl<T: 'static> DeviceAddress<[T]> {
    pub fn offset(self, offset: usize) -> Self {
        DeviceAddress {
            address: self.address + (offset * mem::size_of::<T>()) as u64,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Clone for DeviceAddress<T> {
    fn clone(&self) -> Self {
        DeviceAddress {
            address: self.address,
            _phantom: PhantomData,
        }
    }
}

impl<T: ?Sized + 'static> Copy for DeviceAddress<T> {}

/// Handle to an image in a shader.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct ImageHandle {
    /// Index of the image in the image descriptor array.
    pub index: u32,
}

/// Handle to an image in a shader.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct Texture2DHandleRange {
    pub index: u32,
    pub count: u32,
}

/// Handle to a sampler in a shader.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub struct SamplerHandle {
    /// Index of the image in the sampler descriptor array.
    pub index: u32,
}

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
///
/// TODO Drop impl
#[derive(Clone)]
pub struct GraphicsPipeline {
    pub(crate) device: Device,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    // Push descriptors require live VkDescriptorSetLayouts (kill me already)
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub(crate) bindless: bool,
}

impl GraphicsPipeline {
    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.pipeline, label);
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}

/// Compute pipelines.
///
/// TODO Drop impl
#[derive(Clone)]
pub struct ComputePipeline {
    pub(crate) device: Device,
    pub(crate) pipeline: vk::Pipeline,
    pub(crate) pipeline_layout: vk::PipelineLayout,
    _descriptor_set_layouts: Vec<DescriptorSetLayout>,
    pub(crate) bindless: bool,
}

impl ComputePipeline {
    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.pipeline, label);
        }
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
    id: SamplerId,
    sampler: vk::Sampler,
}

impl Sampler {
    pub fn set_name(&self, label: &str) {
        unsafe {
            self.device
                .upgrade()
                .expect("the underlying device of this sampler has been destroyed")
                .set_object_name(self.sampler, label);
        }
    }

    pub fn handle(&self) -> vk::Sampler {
        let _device = self
            .device
            .upgrade()
            .expect("the underlying device of this sampler has been destroyed");
        self.sampler
    }

    pub fn descriptor(&self) -> Descriptor {
        Descriptor::Sampler { sampler: self.clone() }
    }

    pub fn device_handle(&self) -> SamplerHandle {
        SamplerHandle { index: self.id.index() }
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

pub trait GpuResource {
    fn set_last_submission_index(&self, submission_index: u64);
}

#[derive(Debug)]
struct BufferInner {
    device: Device,
    id: BufferId,
    memory_location: MemoryLocation,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    handle: vk::Buffer,
    device_address: vk::DeviceAddress,
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        // SAFETY: The device resource tracker holds strong references to resources as long as they are in use by the GPU.
        // This prevents `drop` from being called while the resource is still in use, and thus it's safe to delete the
        // resource here.
        unsafe {
            /*debug!(
                "dropping buffer handle={:?}, last_sub={}",
                self.handle,
                self.last_submission_index.load(Ordering::Relaxed)
            );*/
            // retire the ID
            self.device.inner.buffer_ids.lock().unwrap().remove(self.id);
            self.device.free_memory(&mut self.allocation);
            self.device.destroy_buffer(self.handle, None);
        }
    }
}

/// Wrapper around a Vulkan buffer.
#[derive(Clone, Debug)]
pub struct BufferUntyped {
    inner: Option<Arc<BufferInner>>,
    handle: vk::Buffer,
    size: u64,
    usage: BufferUsage,
    mapped_ptr: Option<NonNull<c_void>>,
}

impl GpuResource for BufferUntyped {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .store(submission_index, Ordering::Release);
    }
}

impl Drop for BufferUntyped {
    fn drop(&mut self) {
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.device.clone().delete_later(last_submission_index, inner);
        }
    }
}

impl BufferUntyped {
    pub fn set_name(&self, name: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.inner.as_ref().unwrap().device.set_object_name(self.handle, name);
        }
    }

    pub fn device_address(&self) -> DeviceAddressUntyped {
        DeviceAddressUntyped {
            address: self.inner.as_ref().unwrap().device_address,
        }
    }

    pub(crate) fn id(&self) -> BufferId {
        self.inner.as_ref().unwrap().id
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.size
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Returns the buffer's memory location.
    pub fn memory_location(&self) -> MemoryLocation {
        self.inner.as_ref().unwrap().memory_location
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        &self.inner.as_ref().unwrap().device
    }

    /// Returns whether the buffer is host-visible, and mapped in host memory.
    pub fn host_visible(&self) -> bool {
        self.mapped_ptr.is_some()
    }

    /// Returns a pointer to the buffer mapped in host memory. Panics if the buffer was not mapped in
    /// host memory.
    pub fn as_mut_ptr(&self) -> *mut u8 {
        self.mapped_ptr
            .expect("buffer was not mapped in host memory (consider using MemoryLocation::CpuToGpu)")
            .as_ptr() as *mut u8
    }
}

impl<T: ?Sized> From<Buffer<T>> for BufferUntyped {
    fn from(buffer: Buffer<T>) -> Self {
        buffer.untyped
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Images

#[derive(Debug)]
struct ImageInner {
    device: Device,
    id: ImageId,
    // Number of user references to this image (via `graal::Image`)
    //user_ref_count: AtomicU32,
    last_submission_index: AtomicU64,
    allocation: ResourceAllocation,
    handle: vk::Image,
    swapchain_image: bool,
}

impl Drop for ImageInner {
    fn drop(&mut self) {
        if !self.swapchain_image {
            unsafe {
                debug!("dropping image {:?} (handle: {:?})", self.id, self.handle);
                self.device.inner.image_ids.lock().unwrap().remove(self.id);
                self.device.free_memory(&mut self.allocation);
                self.device.destroy_image(self.handle, None);
            }
        }
    }
}

/// Wrapper around a Vulkan image.
#[derive(Clone, Debug)]
pub struct Image {
    inner: Option<Arc<ImageInner>>,
    handle: vk::Image,
    usage: ImageUsage,
    type_: ImageType,
    format: Format,
    size: Size3D,
}

impl Drop for Image {
    fn drop(&mut self) {
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.device.clone().delete_later(last_submission_index, inner);
        }
    }
}

impl GpuResource for Image {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .store(submission_index, Ordering::Release);
    }
}

impl Image {
    pub fn set_name(&self, label: &str) {
        unsafe {
            self.inner.as_ref().unwrap().device.set_object_name(self.handle, label);
        }
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
        self.inner.as_ref().unwrap().id
    }

    /// Returns the image handle.
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    pub fn device(&self) -> &Device {
        &self.inner.as_ref().unwrap().device
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
    pub(crate) fn create_view(&self, info: &ImageViewInfo) -> ImageView {
        self.inner.as_ref().unwrap().device.create_image_view(self, info)
    }
}

#[derive(Debug)]
struct ImageViewInner {
    // Don't hold Arc<ImageInner> here
    //
    // 1. create the image view
    // 2. use it in a submission (#1)
    // 3. use the image in a later submission (#2)
    // 4. drop the image ref -> not added to the deferred deletion list because the image view still holds a reference
    // ImageView now holds the last ref
    // 5. drop the ImageView -> image view added to the deferred deletion list
    // 6. ImageView deleted when #1 finishes, along with the image since it holds the last ref,
    //    but the image might still be in use by #2!
    image: Image,
    id: ImageViewId,
    handle: vk::ImageView,
    last_submission_index: AtomicU64,
}

impl Drop for ImageViewInner {
    fn drop(&mut self) {
        unsafe {
            self.image.device().inner.image_view_ids.lock().unwrap().remove(self.id);
            self.image.device().destroy_image_view(self.handle, None);
        }
    }
}

/// A view over an image subresource or subresource range.
#[derive(Clone, Debug)]
pub struct ImageView {
    inner: Option<Arc<ImageViewInner>>,
    handle: vk::ImageView,
    format: Format,
    size: Size3D,
}

impl Drop for ImageView {
    fn drop(&mut self) {
        if let Some(inner) = Arc::into_inner(self.inner.take().unwrap()) {
            let last_submission_index = inner.last_submission_index.load(Ordering::Relaxed);
            inner.image.device().clone().delete_later(last_submission_index, inner);
        }
    }
}

impl GpuResource for ImageView {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.inner
            .as_ref()
            .unwrap()
            .last_submission_index
            .store(submission_index, Ordering::Release);
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

    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.image().device().set_object_name(self.handle, label);
        }
    }

    pub fn image(&self) -> &Image {
        &self.inner.as_ref().unwrap().image
    }

    pub(crate) fn id(&self) -> ImageViewId {
        self.inner.as_ref().unwrap().id
    }

    pub fn texture_descriptor(&self, layout: vk::ImageLayout) -> Descriptor {
        Descriptor::SampledImage {
            image_view: self.clone(),
            layout,
        }
    }

    pub fn storage_image_descriptor(&self, layout: vk::ImageLayout) -> Descriptor {
        Descriptor::StorageImage {
            image_view: self.clone(),
            layout,
        }
    }

    /// Returns the bindless texture handle of this image view.
    pub fn device_image_handle(&self) -> ImageHandle {
        ImageHandle {
            index: self.id().index(),
        }
    }

    //pub fn device_handle(&self) ->
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Clone, Debug)]
pub struct DescriptorSetLayout {
    device: Device,
    last_submission_index: Option<Arc<AtomicU64>>,
    pub handle: vk::DescriptorSetLayout,
}

impl Drop for DescriptorSetLayout {
    fn drop(&mut self) {
        if let Some(last_submission_index) = Arc::into_inner(self.last_submission_index.take().unwrap()) {
            let device = self.device.clone();
            let handle = self.handle;
            self.device
                .call_later(last_submission_index.load(Ordering::Relaxed), move || unsafe {
                    device.destroy_descriptor_set_layout(handle, None);
                });
        }
    }
}

/*
/// Self-contained descriptor set.
pub struct DescriptorSet {
    device: Device,
    last_submission_index: Option<Arc<AtomicU64>>,
    pool: vk::DescriptorPool,
    handle: vk::DescriptorSet,
}

impl DescriptorSet {
    pub fn handle(&self) -> vk::DescriptorSet {
        self.handle
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn set_name(&self, label: &str) {
        // SAFETY: the handle is valid
        unsafe {
            self.device.set_object_name(self.handle, label);
        }
    }
}

impl Drop for DescriptorSet {
    fn drop(&mut self) {
        if let Some(last_submission_index) = Arc::into_inner(self.last_submission_index.take().unwrap()) {
            let device = self.device.clone();
            let pool = self.pool;
            self.device
                .call_later(last_submission_index.load(Ordering::Relaxed), move || unsafe {
                    device.destroy_descriptor_pool(pool, None);
                });
        }
    }
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

/// Description of one argument in an argument block.
pub enum Descriptor {
    SampledImage {
        image_view: ImageView,
        layout: vk::ImageLayout,
    },
    StorageImage {
        image_view: ImageView,
        layout: vk::ImageLayout,
    },
    UniformBuffer {
        buffer: BufferUntyped,
        offset: u64,
        size: u64,
    },
    StorageBuffer {
        buffer: BufferUntyped,
        offset: u64,
        size: u64,
    },
    Sampler {
        sampler: Sampler,
    },
}

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
            buffer: self.clone(),
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

impl<T: ?Sized> Clone for Buffer<T> {
    fn clone(&self) -> Self {
        Self {
            untyped: self.untyped.clone(),
            _marker: PhantomData,
        }
    }
}

impl<T: ?Sized> GpuResource for Buffer<T> {
    fn set_last_submission_index(&self, submission_index: u64) {
        self.untyped.set_last_submission_index(submission_index);
    }
}

impl<T: ?Sized> Buffer<T> {
    fn new(buffer: BufferUntyped) -> Self {
        Self {
            untyped: buffer,
            _marker: PhantomData,
        }
    }

    pub fn set_name(&self, name: &str) {
        self.untyped.set_name(name);
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> u64 {
        self.untyped.byte_size()
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> BufferUsage {
        self.untyped.usage()
    }

    pub fn memory_location(&self) -> MemoryLocation {
        self.untyped.memory_location()
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.untyped.handle()
    }

    pub fn host_visible(&self) -> bool {
        self.untyped.host_visible()
    }

    pub fn device_address(&self) -> DeviceAddress<T> {
        DeviceAddress {
            address: self.untyped.device_address().address,
            _phantom: PhantomData,
        }
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        self.untyped.device()
    }
}

impl<T: Copy + 'static> Buffer<T> {
    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.untyped.as_mut_ptr() as *mut T
    }
}

impl<T> Buffer<[T]> {
    /// Returns the number of elements in the buffer.
    pub fn len(&self) -> usize {
        (self.byte_size() / mem::size_of::<T>() as u64) as usize
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn as_mut_ptr(&self) -> *mut T {
        self.untyped.as_mut_ptr() as *mut T
    }

    /*pub fn device_address(&self) -> DeviceAddress<[T]> {
        DeviceAddress {
            address: self.untyped.device_address().address,
            _phantom: PhantomData,
        }
    }*/

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

#[derive(Clone, Debug)]
pub struct BufferRangeUntyped {
    pub buffer: BufferUntyped,
    pub offset: u64,
    pub size: u64,
}

impl BufferRangeUntyped {
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn handle(&self) -> vk::Buffer {
        self.buffer.handle
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    pub fn storage_descriptor(&self) -> Descriptor {
        Descriptor::StorageBuffer {
            buffer: self.buffer.clone(),
            offset: self.offset,
            size: self.size,
        }
    }

    pub fn uniform_descriptor(&self) -> Descriptor {
        Descriptor::UniformBuffer {
            buffer: self.buffer.clone(),
            offset: self.offset,
            size: self.size,
        }
    }
}

pub struct BufferRange<T: ?Sized> {
    pub untyped: BufferRangeUntyped,
    _phantom: PhantomData<T>,
}

// #26925 clone impl
impl<T: ?Sized> Clone for BufferRange<T> {
    fn clone(&self) -> Self {
        Self {
            untyped: self.untyped.clone(),
            _phantom: PhantomData,
        }
    }
}

impl<T> BufferRange<[T]> {
    pub fn len(&self) -> usize {
        (self.untyped.size / mem::size_of::<T>() as u64) as usize
    }

    pub fn storage_descriptor(&self) -> Descriptor {
        self.untyped.storage_descriptor()
    }

    pub fn uniform_descriptor(&self) -> Descriptor {
        self.untyped.uniform_descriptor()
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
#[derive(Clone)]
pub struct ColorAttachment<'a> {
    pub image_view: &'a ImageView,
    pub clear_value: Option<[f64; 4]>,
    /*pub image_view: ImageView,
    pub load_op: vk::AttachmentLoadOp,
    pub store_op: vk::AttachmentStoreOp,
    pub clear_value: [f64; 4],*/
}

impl ColorAttachment<'_> {
    pub(crate) fn get_vk_clear_color_value(&self) -> vk::ClearColorValue {
        if let Some(clear_value) = self.clear_value {
            match format_numeric_type(self.image_view.format) {
                FormatNumericType::UInt => vk::ClearColorValue {
                    uint32: [
                        clear_value[0] as u32,
                        clear_value[1] as u32,
                        clear_value[2] as u32,
                        clear_value[3] as u32,
                    ],
                },
                FormatNumericType::SInt => vk::ClearColorValue {
                    int32: [
                        clear_value[0] as i32,
                        clear_value[1] as i32,
                        clear_value[2] as i32,
                        clear_value[3] as i32,
                    ],
                },
                FormatNumericType::Float => vk::ClearColorValue {
                    float32: [
                        clear_value[0] as f32,
                        clear_value[1] as f32,
                        clear_value[2] as f32,
                        clear_value[3] as f32,
                    ],
                },
            }
        } else {
            vk::ClearColorValue::default()
        }
    }
}

#[derive(Clone)]
pub struct DepthStencilAttachment<'a> {
    pub image_view: &'a ImageView,
    pub depth_clear_value: Option<f64>,
    pub stencil_clear_value: Option<u32>,
    /*pub depth_load_op: vk::AttachmentLoadOp,
    pub depth_store_op: vk::AttachmentStoreOp,
    pub stencil_load_op: vk::AttachmentLoadOp,
    pub stencil_store_op: vk::AttachmentStoreOp,
    pub depth_clear_value: f64,
    pub stencil_clear_value: u32,*/
}

impl DepthStencilAttachment<'_> {
    pub(crate) fn get_vk_clear_depth_stencil_value(&self) -> vk::ClearDepthStencilValue {
        vk::ClearDepthStencilValue {
            depth: self.depth_clear_value.unwrap_or(0.0) as f32,
            stencil: self.stencil_clear_value.unwrap_or(0),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Clone, Debug)]
pub struct VertexBufferDescriptor {
    pub binding: u32,
    pub buffer_range: BufferRangeUntyped,
    pub stride: u32,
}

pub trait VertexInput {
    /// Vertex buffer bindings
    fn buffer_layout(&self) -> Cow<[VertexBufferLayoutDescription]>;

    /// Vertex attributes.
    fn attributes(&self) -> Cow<[VertexInputAttributeDescription]>;

    /// Returns an iterator over the vertex buffers referenced in this object.
    fn vertex_buffers(&self) -> impl Iterator<Item = VertexBufferDescriptor>;
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

    pub fn from_spirv(spirv: &'a [u32], entry_point: &'a str) -> ShaderEntryPoint<'a> {
        Self {
            code: ShaderCode::Spirv(spirv),
            entry_point,
        }
    }
}

/// Specifies the shaders of a graphics pipeline.
#[derive(Copy, Clone, Debug)]
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

#[derive(Copy, Clone, Debug)]
pub struct GraphicsPipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout],
    // None of the relevant drivers on desktop seem to care about precise push constant ranges,
    // so we just store the total size of push constants.
    pub push_constants_size: usize,
    pub vertex_input: VertexInputState<'a>,
    pub pre_rasterization_shaders: PreRasterizationShaders<'a>,
    pub rasterization: RasterizationState,
    pub depth_stencil: Option<DepthStencilState>,
    pub fragment: FragmentState<'a>,
}

#[derive(Copy, Clone, Debug)]
pub struct ComputePipelineCreateInfo<'a> {
    /// If left empty, use the universal descriptor set layout.
    pub set_layouts: &'a [DescriptorSetLayout],
    pub push_constants_size: usize,
    pub compute_shader: ShaderEntryPoint<'a>,
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
        Format::D16_UNORM_S8_UINT | Format::D24_UNORM_S8_UINT | Format::D32_SFLOAT_S8_UINT
    )
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        Format::D16_UNORM | Format::X8_D24_UNORM_PACK32 | Format::D32_SFLOAT
    )
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, Format::S8_UINT)
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

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum FormatNumericType {
    SInt,
    UInt,
    Float,
}

pub fn format_numeric_type(fmt: vk::Format) -> FormatNumericType {
    match fmt {
        vk::Format::R8_UINT
        | vk::Format::R8G8_UINT
        | vk::Format::R8G8B8_UINT
        | vk::Format::R8G8B8A8_UINT
        | vk::Format::R16_UINT
        | vk::Format::R16G16_UINT
        | vk::Format::R16G16B16_UINT
        | vk::Format::R16G16B16A16_UINT
        | vk::Format::R32_UINT
        | vk::Format::R32G32_UINT
        | vk::Format::R32G32B32_UINT
        | vk::Format::R32G32B32A32_UINT
        | vk::Format::R64_UINT
        | vk::Format::R64G64_UINT
        | vk::Format::R64G64B64_UINT
        | vk::Format::R64G64B64A64_UINT => FormatNumericType::UInt,

        vk::Format::R8_SINT
        | vk::Format::R8G8_SINT
        | vk::Format::R8G8B8_SINT
        | vk::Format::R8G8B8A8_SINT
        | vk::Format::R16_SINT
        | vk::Format::R16G16_SINT
        | vk::Format::R16G16B16_SINT
        | vk::Format::R16G16B16A16_SINT
        | vk::Format::R32_SINT
        | vk::Format::R32G32_SINT
        | vk::Format::R32G32B32_SINT
        | vk::Format::R32G32B32A32_SINT
        | vk::Format::R64_SINT
        | vk::Format::R64G64_SINT
        | vk::Format::R64G64B64_SINT
        | vk::Format::R64G64B64A64_SINT => FormatNumericType::SInt,

        vk::Format::R16_SFLOAT
        | vk::Format::R16G16_SFLOAT
        | vk::Format::R16G16B16_SFLOAT
        | vk::Format::R16G16B16A16_SFLOAT
        | vk::Format::R32_SFLOAT
        | vk::Format::R32G32_SFLOAT
        | vk::Format::R32G32B32_SFLOAT
        | vk::Format::R32G32B32A32_SFLOAT
        | vk::Format::R64_SFLOAT
        | vk::Format::R64G64_SFLOAT
        | vk::Format::R64G64B64_SFLOAT
        | vk::Format::R64G64B64A64_SFLOAT => FormatNumericType::Float,

        // TODO
        _ => FormatNumericType::Float,
    }
}

/*
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
}*/

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
