//! Abstractions over a vulkan device & queues.
use std::{
    borrow::Cow,
    cell::{Cell, RefCell},
    collections::HashMap,
    ffi::{c_char, CString},
    fmt,
    ops::Deref,
    os::raw::c_void,
    ptr,
    ptr::NonNull,
    rc::{Rc, Weak},
    sync::atomic::{AtomicUsize, Ordering},
};

use ash::vk;
use fxhash::FxHashMap;
use shaderc::{EnvVersion, SpirvVersion, TargetEnv};
use slotmap::{Key, SlotMap};

pub use encoder::*;
pub use queue::*;

use crate::{
    aspects_for_format,
    device::resource::ResourceGroup,
    instance::{vk_khr_debug_utils, vk_khr_surface},
    is_write_access, platform_impl, ArgumentsLayout, BufferRangeAny, BufferUsage, CompareOp, Error, Format,
    GraphicsPipelineCreateInfo, ImageSubresourceRange, ImageType, ImageUsage, ImageViewInfo, PreRasterizationShaders,
    ReadWriteStorageImage, ResourceState, SamplerCreateInfo, ShaderCode, ShaderKind, ShaderSource, Size3D,
};

mod command_buffer;
mod encoder;
mod init;
mod queue;
mod resource;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around a vulkan device, associated queues and tracked resources.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Rc<DeviceInner>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Device").finish_non_exhaustive()
    }
}

/// Weak reference to a device
#[derive(Clone)]
pub struct WeakDevice {
    pub(crate) inner: Weak<DeviceInner>,
}

impl fmt::Debug for WeakDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("WeakDevice").finish_non_exhaustive()
    }
}

pub(crate) struct DeviceInner {
    /// Underlying vulkan device
    device: ash::Device,
    /// Platform-specific extension functions
    platform_extensions: platform_impl::PlatformExtensions,
    physical_device: vk::PhysicalDevice,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device_properties: vk::PhysicalDeviceProperties,
    queues: Vec<QueueData>,
    allocator: RefCell<gpu_allocator::vulkan::Allocator>,
    vk_khr_swapchain: ash::extensions::khr::Swapchain,
    vk_ext_shader_object: ash::extensions::ext::ShaderObject,
    vk_khr_push_descriptor: ash::extensions::khr::PushDescriptor,
    vk_ext_mesh_shader: ash::extensions::ext::MeshShader,
    vk_ext_extended_dynamic_state3: ash::extensions::ext::ExtendedDynamicState3,
    resources: RefCell<ResourceMap>,
    resource_groups: RefCell<ResourceGroupMap>,
    deletion_lists: RefCell<Vec<DeferredDeletionList>>,
    sampler_cache: RefCell<HashMap<SamplerCreateInfo, Sampler>>,
    compiler: shaderc::Compiler,
}

#[derive(thiserror::Error, Debug)]
pub enum DeviceCreateError {
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyConfig {
    pub family_index: u32,
    pub count: u32,
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner.device
    }
}

slotmap::new_key_type! {
    /// Identifies a GPU resource (buffer or image).
    pub struct ResourceId;

    /// Identifies a resource group.
    pub struct GroupId;
}

/// Identifier for a buffer resource.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BufferId(pub(crate) ResourceId);

impl BufferId {
    /// Returns the underlying ResourceId.
    pub fn resource_id(&self) -> ResourceId {
        self.0
    }

    /// Produces an invalid BufferId, for testing.
    #[cfg(test)]
    pub fn invalid() -> BufferId {
        BufferId(ResourceId::null())
    }
}

/// Identifier for an image resource.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub(crate) ResourceId);

impl ImageId {
    /// Returns the underlying ResourceId.
    pub fn resource_id(&self) -> ResourceId {
        self.0
    }

    /// Produces an invalid ImageId, for testing.
    pub fn invalid() -> ImageId {
        ImageId(ResourceId::null())
    }
}

impl From<ImageId> for ResourceId {
    fn from(value: ImageId) -> Self {
        value.0
    }
}

impl From<BufferId> for ResourceId {
    fn from(value: BufferId) -> Self {
        value.0
    }
}

/// Command buffers
pub struct CommandBuffer {
    device: Device,
    /// Keeps referenced resources alive.
    refs: Vec<RefCounted<ResourceId>>,
    command_buffer: vk::CommandBuffer,
    initial_uses: FxHashMap<ResourceId, ResourceUse>,
    final_states: FxHashMap<ResourceId, DependencyState>,
    barrier_builder: PipelineBarrierBuilder,
    group_uses: Vec<GroupId>,
}

/// Graphics pipelines.
pub struct GraphicsPipeline {
    device: Device,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub(super) fn new(device: Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
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
    sampler: vk::Sampler,
}

impl Sampler {
    pub(super) fn new(device: &Device, sampler: vk::Sampler) -> Sampler {
        Sampler {
            device: device.weak(),
            sampler,
        }
    }

    pub fn handle(&self) -> vk::Sampler {
        // FIXME: check if the device is still alive, otherwise the sampler isn't valid anymore
        //assert!(self.device.strong_count() > 0);
        self.sampler
    }
}

/// Wrapper around a Vulkan buffer.
#[derive(Debug)]
pub struct Buffer {
    device: Device,
    id: RefCounted<BufferId>,
    handle: vk::Buffer,
    size: u64,
    usage: BufferUsage,
    mapped_ptr: Option<NonNull<c_void>>,
}

impl Buffer {
    /*fn new(
        device: Device,
        id: RefCounted<BufferId>,
        handle: vk::Buffer,
        size: usize,
        usage: vk::BufferUsageFlags,
        mapped_ptr: Option<NonNull<c_void>>,
    ) -> Self {
        Self {
            device,
            id,
            handle,
            size,
            usage,
            mapped_ptr,
        }
    }*/

    pub fn id(&self) -> RefCounted<BufferId> {
        self.id.clone()
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
        &self.device
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr.as_ptr() as *mut u8)
    }
}

/// Wrapper around a Vulkan image.
#[derive(Debug)]
pub struct Image {
    device: Device,
    id: RefCounted<ImageId>,
    handle: vk::Image,
    usage: ImageUsage,
    type_: ImageType,
    format: Format,
    size: Size3D,
}

impl Image {
    /*pub(super) fn new(
        device: Device,
        id: RefCounted<ImageId>,
        handle: vk::Image,
        usage: vk::ImageUsageFlags,
        type_: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
    ) -> Self {
        Image {
            device,
            id,
            handle,
            usage,
            type_,
            format,
            extent,
        }
    }*/

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

    pub fn id(&self) -> RefCounted<ImageId> {
        self.id.clone()
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
        // TODO: check that format is compatible

        // FIXME: support non-zero base mip level
        if info.subresource_range.base_mip_level != 0 {
            unimplemented!("non-zero base mip level");
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: self.handle,
            view_type: info.view_type,
            format: info.format,
            components: vk::ComponentMapping {
                r: info.component_mapping[0],
                g: info.component_mapping[1],
                b: info.component_mapping[2],
                a: info.component_mapping[3],
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: info.subresource_range.aspect_mask,
                base_mip_level: info.subresource_range.base_mip_level,
                level_count: info.subresource_range.level_count,
                base_array_layer: info.subresource_range.base_array_layer,
                layer_count: info.subresource_range.layer_count,
            },
            ..Default::default()
        };

        // SAFETY: the device is valid, the create info is valid
        let handle = unsafe {
            self.device
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        };

        ImageView {
            device: self.device.clone(),
            parent_image: self.id.clone(),
            handle,
            image_handle: self.handle,
            format: info.format,
            original_format: self.format,
            // TODO: size of mip level
            size: self.size,
        }
    }
}

#[derive(Debug)]
pub struct ImageView {
    device: Device,
    parent_image: RefCounted<ImageId>,
    image_handle: vk::Image,
    handle: vk::ImageView,
    format: Format,
    original_format: Format,
    size: Size3D,
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

    fn image_handle(&self) -> vk::Image {
        self.image_handle
    }

    fn original_format(&self) -> vk::Format {
        self.original_format
    }

    fn parent_id(&self) -> RefCounted<ImageId> {
        self.parent_image.clone()
    }

    pub fn as_read_write_storage(&self) -> ReadWriteStorageImage {
        ReadWriteStorageImage { image_view: self }
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.delete_later(self.handle);
        }
    }
}

// TODO: move this to a separate module?
#[derive(Debug)]
pub struct ResourceRegistrationInfo<'a> {
    pub name: &'a str,
    pub allocation: ResourceAllocation,
    pub initial_wait: Option<SemaphoreWait>,
}

#[derive(Debug)]
pub struct ImageRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
    pub handle: vk::Image,
    pub format: vk::Format,
}

#[derive(Debug)]
pub struct BufferRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
    pub handle: vk::Buffer,
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

/// Describes how a resource got its memory.
#[derive(Debug)]
pub enum ResourceAllocation {
    /// We allocated a block of memory exclusively for this resource.
    Allocation {
        allocation: gpu_allocator::vulkan::Allocation,
    },

    /// Memory aliasing: allocate a block of memory for the resource, which can possibly be shared
    /// with other aliasable resources if their lifetimes do not overlap.
    Transient {
        device_memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    },

    /// The memory for this resource was imported or exported from/to an external handle.
    DeviceMemory { device_memory: vk::DeviceMemory },

    /// We don't own the memory for this resource.
    External,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Debug)]
enum DeferredDeletionObject {
    ImageView(vk::ImageView),
    Sampler(vk::Sampler),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    Semaphore(vk::Semaphore),
    Shader(vk::ShaderEXT),
}

impl From<vk::ImageView> for DeferredDeletionObject {
    fn from(view: vk::ImageView) -> Self {
        DeferredDeletionObject::ImageView(view)
    }
}
impl From<vk::Sampler> for DeferredDeletionObject {
    fn from(sampler: vk::Sampler) -> Self {
        DeferredDeletionObject::Sampler(sampler)
    }
}
impl From<vk::PipelineLayout> for DeferredDeletionObject {
    fn from(layout: vk::PipelineLayout) -> Self {
        DeferredDeletionObject::PipelineLayout(layout)
    }
}
impl From<vk::Pipeline> for DeferredDeletionObject {
    fn from(pipeline: vk::Pipeline) -> Self {
        DeferredDeletionObject::Pipeline(pipeline)
    }
}
impl From<vk::Semaphore> for DeferredDeletionObject {
    fn from(semaphore: vk::Semaphore) -> Self {
        DeferredDeletionObject::Semaphore(semaphore)
    }
}
impl From<vk::DescriptorSetLayout> for DeferredDeletionObject {
    fn from(layout: vk::DescriptorSetLayout) -> Self {
        DeferredDeletionObject::DescriptorSetLayout(layout)
    }
}
impl From<vk::ShaderEXT> for DeferredDeletionObject {
    fn from(shader: vk::ShaderEXT) -> Self {
        DeferredDeletionObject::Shader(shader)
    }
}

/// Chooses a swapchain surface format among a list of supported formats.
///
/// TODO there's only one supported format right now...
fn get_preferred_swapchain_surface_format(surface_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    surface_formats
        .iter()
        .find_map(|&fmt| {
            if fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                Some(fmt)
            } else {
                None
            }
        })
        .expect("no suitable surface format available")
}

pub unsafe fn create_device_and_queue(
    present_surface: Option<vk::SurfaceKHR>,
) -> Result<(Device, Queue), DeviceCreateError> {
    let device = Device::new(present_surface)?;
    let queue = device.get_queue_by_global_index(0);
    Ok((device, queue))
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct DeferredDeletionList {
    // Timestamp that we should wait for on each queue before deleting the objects.
    timestamps: Vec<u64>,
    // Objects to delete.
    objects: Vec<DeferredDeletionObject>,
}

struct QueueData {
    /// Family index.
    family_index: u32,
    /// Index within queues of the same family (see vkGetDeviceQueue).
    index: u32,
    queue: vk::Queue,
    timeline: vk::Semaphore,
    next_submission_timestamp: Cell<u64>,
}

#[derive(Debug)]
pub(crate) struct ImageResource {
    handle: vk::Image,
    format: vk::Format,
    all_aspects: vk::ImageAspectFlags,
}

#[derive(Debug)]
pub(crate) struct BufferResource {
    handle: vk::Buffer,
}
/// Describes how the resource is access by different queues.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum OwnerQueue {
    None,
    Exclusive(usize),
    Concurrent(u16),
}

#[derive(Debug)]
pub struct RefCount(NonNull<AtomicUsize>);

unsafe impl Send for RefCount {}
unsafe impl Sync for RefCount {}

impl RefCount {
    const MAX: usize = 1 << 24;

    /// Construct a new `RefCount`, with an initial count of 1.
    fn new() -> RefCount {
        let bx = Box::new(AtomicUsize::new(1));
        Self(unsafe { NonNull::new_unchecked(Box::into_raw(bx)) })
    }

    fn load(&self) -> usize {
        unsafe { self.0.as_ref() }.load(Ordering::Acquire)
    }

    fn is_unique(&self) -> bool {
        self.load() == 1
    }
}

impl Clone for RefCount {
    fn clone(&self) -> Self {
        let old_size = unsafe { self.0.as_ref() }.fetch_add(1, Ordering::AcqRel);
        assert!(old_size < Self::MAX);
        Self(self.0)
    }
}

impl Drop for RefCount {
    fn drop(&mut self) {
        unsafe {
            if self.0.as_ref().fetch_sub(1, Ordering::AcqRel) == 1 {
                drop(Box::from_raw(self.0.as_ptr()));
            }
        }
    }
}

#[derive(Clone, Debug)]
pub struct RefCounted<T> {
    pub ref_count: RefCount,
    pub value: T,
}

impl<T> RefCounted<T> {
    pub fn new(value: T, initial_count: RefCount) -> Self {
        Self {
            ref_count: initial_count,
            value,
        }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> RefCounted<U> {
        RefCounted {
            ref_count: self.ref_count,
            value: f(self.value),
        }
    }
}

#[derive(Debug)]
enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Debug)]
pub(crate) struct Resource {
    name: String,
    // Could be usize, or a pointer to an atomic usize variable. With the pointer,
    // there's no need to hold a reference to the device for Samplers, etc, or to manually call
    // `device.drop_resource(id)`. But that's one more allocation.
    ref_count: RefCount,
    allocation: ResourceAllocation,
    kind: ResourceKind,
    group: Option<GroupId>,
    owner: OwnerQueue,
    wait_semaphore: Option<SignaledSemaphore>,
    dep_state: DependencyState,
    /// Timestamp of the last submission that accessed the resource on its owner queue.
    ///
    /// It's a value of the timeline semaphore of the queue.
    ///
    /// TODO: it's only used for deferred deletion, to ensure that no queue is still using the resource.
    /// We could replace that by a set of semaphore waits in order to support waiting on a resource
    /// used concurrently by multiple queues.
    timestamp: u64,
}

impl Resource {
    pub(crate) fn as_image(&self) -> &ImageResource {
        match &self.kind {
            ResourceKind::Image(r) => r,
            _ => panic!("expected an image resource"),
        }
    }

    pub(crate) fn as_buffer(&self) -> &BufferResource {
        match &self.kind {
            ResourceKind::Buffer(r) => r,
            _ => panic!("expected a buffer resource"),
        }
    }
}

type ResourceMap = SlotMap<ResourceId, Resource>;
type ResourceGroupMap = SlotMap<GroupId, ResourceGroup>;

#[derive(Copy, Clone, Debug, Default)]
struct DependencyState {
    stages: vk::PipelineStageFlags2,
    flush_mask: vk::AccessFlags2,
    visible: vk::AccessFlags2,
    layout: vk::ImageLayout,
}

#[derive(Copy, Clone)]
struct ResourceUse {
    aspect: vk::ImageAspectFlags,
    state: ResourceState,
}

/// Helper to build a pipeline barrier.
#[derive(Default)]
pub(super) struct PipelineBarrierBuilder {
    image_barriers: Vec<vk::ImageMemoryBarrier2>,
    buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
}

impl PipelineBarrierBuilder {
    fn get_or_create_image_barrier(&mut self, image: vk::Image) -> &mut vk::ImageMemoryBarrier2 {
        let index = self.image_barriers.iter().position(|barrier| barrier.image == image);
        if let Some(index) = index {
            &mut self.image_barriers[index]
        } else {
            let barrier = vk::ImageMemoryBarrier2::default();
            self.image_barriers.push(barrier);
            self.image_barriers.last_mut().unwrap()
        }
    }

    fn get_or_create_buffer_barrier(&mut self, buffer: vk::Buffer) -> &mut vk::BufferMemoryBarrier2 {
        let index = self.buffer_barriers.iter().position(|barrier| barrier.buffer == buffer);
        if let Some(index) = index {
            &mut self.buffer_barriers[index]
        } else {
            let barrier = vk::BufferMemoryBarrier2::default();
            self.buffer_barriers.push(barrier);
            self.buffer_barriers.last_mut().unwrap()
        }
    }

    fn clear(&mut self) {
        self.buffer_barriers.clear();
        self.image_barriers.clear();
    }

    fn is_empty(&self) -> bool {
        self.buffer_barriers.is_empty() && self.image_barriers.is_empty()
    }
}

#[derive(Copy, Clone, Debug)]
enum ResourceHandle {
    Buffer(vk::Buffer),
    Image(vk::Image),
}

impl From<vk::Buffer> for ResourceHandle {
    fn from(buffer: vk::Buffer) -> Self {
        Self::Buffer(buffer)
    }
}

impl From<vk::Image> for ResourceHandle {
    fn from(image: vk::Image) -> Self {
        Self::Image(image)
    }
}

fn ensure_memory_dependency(
    barriers: &mut PipelineBarrierBuilder,
    handle: ResourceHandle,
    dep_state: &mut DependencyState,
    use_: &ResourceUse,
) {
    // We need to insert a memory dependency if:
    // - the image needs a layout transition
    // - the resource has writes that are not visible to the target access
    //
    // We need to insert an execution dependency if:
    // - we're writing to the resource (write-after-read hazard, write-after-write hazard)
    if !(dep_state.visible.contains(use_.state.access) || dep_state.visible.contains(vk::AccessFlags2::MEMORY_READ))
        || dep_state.layout != use_.state.layout
        || is_write_access(use_.state.access)
    {
        match handle {
            ResourceHandle::Buffer(buffer) => {
                let barrier = barriers.get_or_create_buffer_barrier(buffer);
                barrier.buffer = buffer;
                barrier.src_stage_mask |= dep_state.stages;
                barrier.dst_stage_mask |= use_.state.stages;
                barrier.src_access_mask |= dep_state.flush_mask;
                barrier.dst_access_mask |= use_.state.access;
                barrier.offset = 0;
                barrier.size = vk::WHOLE_SIZE;
            }
            ResourceHandle::Image(image) => {
                let barrier = barriers.get_or_create_image_barrier(image);
                barrier.image = image;
                barrier.src_stage_mask |= dep_state.stages;
                barrier.dst_stage_mask |= use_.state.stages;
                barrier.src_access_mask |= dep_state.flush_mask;
                barrier.dst_access_mask |= use_.state.access;
                barrier.old_layout = dep_state.layout;
                barrier.new_layout = use_.state.layout;
                barrier.subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: use_.aspect,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                    ..Default::default()
                };
                dep_state.layout = use_.state.layout;
            }
        }
    }

    if is_write_access(use_.state.access) {
        // we're writing to the resource, so reset visibility...
        dep_state.visible = vk::AccessFlags2::empty();
        // ... but signal that there is data to flush.
        dep_state.flush_mask = use_.state.access;
    } else {
        // This memory dependency makes all writes on the resource available, and
        // visible to the types specified in `access.access_mask`.
        // There's no write, so we don't need to flush anything.
        dep_state.flush_mask = vk::AccessFlags2::empty();
        dep_state.visible |= use_.state.access;
    }

    // Update the resource stage mask
    dep_state.stages = use_.state.stages;
}

impl Device {
    pub fn weak(&self) -> WeakDevice {
        WeakDevice {
            inner: Rc::downgrade(&self.inner),
        }
    }

    pub fn raw(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Function pointers for VK_KHR_swapchain.
    pub fn khr_swapchain(&self) -> &ash::extensions::khr::Swapchain {
        &self.inner.vk_khr_swapchain
    }

    /// Function pointers for VK_KHR_push_descriptor.
    pub fn khr_push_descriptor(&self) -> &ash::extensions::khr::PushDescriptor {
        &self.inner.vk_khr_push_descriptor
    }

    pub fn ext_extended_dynamic_state3(&self) -> &ash::extensions::ext::ExtendedDynamicState3 {
        &self.inner.vk_ext_extended_dynamic_state3
    }

    pub fn ext_mesh_shader(&self) -> &ash::extensions::ext::MeshShader {
        &self.inner.vk_ext_mesh_shader
    }

    /// Helper function to associate a debug name to a vulkan handle.
    fn set_debug_object_name(&self, object_type: vk::ObjectType, object_handle: u64, name: &str, serial: Option<u64>) {
        unsafe {
            let name = if let Some(serial) = serial {
                format!("{}@{}", name, serial)
            } else {
                name.to_string()
            };
            let object_name = CString::new(name.as_str()).unwrap();

            vk_khr_debug_utils()
                .set_debug_utils_object_name(
                    self.inner.device.handle(),
                    &vk::DebugUtilsObjectNameInfoEXT {
                        object_type,
                        object_handle,
                        p_object_name: object_name.as_ptr(),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }

    /// Creates a swapchain object.
    pub unsafe fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        width: u32,
        height: u32,
    ) -> Swapchain {
        let mut swapchain = Swapchain {
            handle: Default::default(),
            surface,
            images: vec![],
            format,
            width,
            height,
        };
        self.resize_swapchain(&mut swapchain, width, height);
        swapchain
    }

    /// Returns the list of supported swapchain formats for the given surface.
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<vk::SurfaceFormatKHR> {
        vk_khr_surface()
            .get_physical_device_surface_formats(self.inner.physical_device, surface)
            .unwrap()
    }

    /// Returns one supported surface format. Use if you don't care about the format of your swapchain.
    pub unsafe fn get_preferred_surface_format(&self, surface: vk::SurfaceKHR) -> vk::SurfaceFormatKHR {
        let surface_formats = self.get_surface_formats(surface);
        get_preferred_swapchain_surface_format(&surface_formats)
    }

    /// Resizes a swapchain.
    pub unsafe fn resize_swapchain(&self, swapchain: &mut Swapchain, width: u32, height: u32) {
        let phy = self.inner.physical_device;
        let capabilities = vk_khr_surface()
            .get_physical_device_surface_capabilities(phy, swapchain.surface)
            .unwrap();
        /*let formats = self
        .vk_khr_surface
        .get_physical_device_surface_formats(phy, swapchain.surface)
        .unwrap();*/
        let present_modes = vk_khr_surface()
            .get_physical_device_surface_present_modes(phy, swapchain.surface)
            .unwrap();

        let present_mode = init::get_preferred_present_mode(&present_modes);
        let image_extent = init::get_preferred_swap_extent((width, height), &capabilities);
        let image_count =
            if capabilities.max_image_count > 0 && capabilities.min_image_count + 1 > capabilities.max_image_count {
                capabilities.max_image_count
            } else {
                capabilities.min_image_count + 1
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            flags: Default::default(),
            surface: swapchain.surface,
            min_image_count: image_count,
            image_format: swapchain.format.format,
            image_color_space: swapchain.format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: swapchain.handle,
            ..Default::default()
        };

        let new_handle = self
            .inner
            .vk_khr_swapchain
            .create_swapchain(&create_info, None)
            .expect("failed to create swapchain");
        if swapchain.handle != vk::SwapchainKHR::null() {
            // FIXME what if the images are in use?
            self.inner.vk_khr_swapchain.destroy_swapchain(swapchain.handle, None);
        }

        swapchain.handle = new_handle;
        swapchain.width = width;
        swapchain.height = height;
        swapchain.images = self
            .inner
            .vk_khr_swapchain
            .get_swapchain_images(swapchain.handle)
            .unwrap();
    }

    pub fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.inner.sampler_cache.borrow().get(info) {
            return sampler.clone();
        }

        let create_info = vk::SamplerCreateInfo {
            flags: Default::default(),
            mag_filter: info.mag_filter,
            min_filter: info.min_filter,
            mipmap_mode: info.mipmap_mode,
            address_mode_u: info.address_mode_u,
            address_mode_v: info.address_mode_v,
            address_mode_w: info.address_mode_w,
            mip_lod_bias: info.mip_lod_bias.0,
            anisotropy_enable: info.anisotropy_enable.into(),
            max_anisotropy: info.max_anisotropy.0,
            compare_enable: info.compare_enable.into(),
            compare_op: info.compare_op.into(),
            min_lod: info.min_lod.0,
            max_lod: info.max_lod.0,
            border_color: info.border_color,
            ..Default::default()
        };

        let sampler = unsafe {
            self.inner
                .device
                .create_sampler(&create_info, None)
                .expect("failed to create sampler")
        };
        let sampler = Sampler::new(self, sampler);
        self.inner
            .sampler_cache
            .borrow_mut()
            .insert(info.clone(), sampler.clone());
        sampler
    }

    /// Compiles a shader
    // TODO: this could be moved outside of device?
    fn compile_shader(&self, kind: ShaderKind, source: ShaderSource, entry_point: &str) -> Result<Vec<u32>, Error> {
        let input_file_name = match source {
            ShaderSource::Content(_) => "<embedded shader>",
            ShaderSource::File(path) => path.file_name().unwrap().to_str().unwrap(),
        };

        let source_from_path;
        let source_content = match source {
            ShaderSource::Content(str) => str,
            ShaderSource::File(path) => {
                source_from_path = std::fs::read_to_string(path)?;
                source_from_path.as_str()
            }
        };
        let kind = match kind {
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
            ShaderKind::Geometry => shaderc::ShaderKind::Geometry,
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
            ShaderKind::TessControl => shaderc::ShaderKind::TessControl,
            ShaderKind::TessEvaluation => shaderc::ShaderKind::TessEvaluation,
            ShaderKind::Mesh => shaderc::ShaderKind::Mesh,
            ShaderKind::Task => shaderc::ShaderKind::Task,
        };

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        let mut base_include_path = std::env::current_dir().expect("failed to get current directory");

        match source {
            ShaderSource::File(path) => {
                if let Some(parent) = path.parent() {
                    base_include_path = parent.to_path_buf();
                }
            }
            _ => {}
        }

        compile_options.set_include_callback(move |requested_source, _type, _requesting_source, _include_depth| {
            let mut path = base_include_path.clone();
            path.push(requested_source);
            let content = match std::fs::read_to_string(&path) {
                Ok(content) => content,
                Err(e) => return Err(e.to_string()),
            };
            Ok(shaderc::ResolvedInclude {
                resolved_name: path.display().to_string(),
                content,
            })
        });
        compile_options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_3 as u32);
        compile_options.set_target_spirv(SpirvVersion::V1_5);

        let compilation_artifact = self.inner.compiler.compile_into_spirv(
            source_content,
            kind,
            input_file_name,
            entry_point,
            Some(&compile_options),
        )?;

        Ok(compilation_artifact.as_binary().into())
    }

    /// Creates a shader module.
    fn create_shader_module(
        &self,
        kind: ShaderKind,
        code: &ShaderCode,
        entry_point: &str,
    ) -> Result<vk::ShaderModule, Error> {
        let code = match code {
            ShaderCode::Source(source) => Cow::Owned(self.compile_shader(kind, *source, entry_point)?),
            ShaderCode::Spirv(spirv) => Cow::Borrowed(*spirv),
        };

        let create_info = vk::ShaderModuleCreateInfo {
            flags: Default::default(),
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        let module = unsafe { self.inner.device.create_shader_module(&create_info, None)? };
        Ok(module)
    }

    /// Creates a pipeline layout object.
    fn create_pipeline_layout(
        &self,
        bind_point: vk::PipelineBindPoint,
        arg_layouts: &[ArgumentsLayout],
        push_constants_size: usize,
    ) -> (Vec<vk::DescriptorSetLayout>, vk::PipelineLayout) {
        let mut set_layouts = Vec::with_capacity(arg_layouts.len());
        for layout in arg_layouts.iter() {
            let create_info = vk::DescriptorSetLayoutCreateInfo {
                flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
                binding_count: layout.bindings.len() as u32,
                p_bindings: layout.bindings.as_ptr(),
                ..Default::default()
            };
            let sl = unsafe {
                self.inner
                    .device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("failed to create descriptor set layout")
            };
            set_layouts.push(sl);
        }

        let pc_range = match bind_point {
            vk::PipelineBindPoint::GRAPHICS => vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS
                    | vk::ShaderStageFlags::MESH_EXT
                    | vk::ShaderStageFlags::TASK_EXT,
                offset: 0,
                size: push_constants_size as u32,
            },
            vk::PipelineBindPoint::COMPUTE => vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: push_constants_size as u32,
            },
            _ => unimplemented!(),
        };

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 1,
            p_push_constant_ranges: &pc_range,
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.inner
                .device
                .create_pipeline_layout(&create_info, None)
                .expect("failed to create pipeline layout")
        };

        (set_layouts, pipeline_layout)
    }

    /// Creates a graphics pipeline.
    pub fn create_graphics_pipeline(&self, create_info: GraphicsPipelineCreateInfo) -> Result<GraphicsPipeline, Error> {
        // ------ create pipeline layout from statically known information ------
        let (descriptor_set_layouts, pipeline_layout) = self.create_pipeline_layout(
            vk::PipelineBindPoint::GRAPHICS,
            create_info.layout.arguments.as_ref(),
            create_info.layout.push_constants_size,
        );

        // FIXME: delete descriptor_set_layouts

        // ------ Dynamic states ------

        // TODO: this could be a static property of the pipeline interface
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        if matches!(
            create_info.pre_rasterization_shaders,
            PreRasterizationShaders::PrimitiveShading { .. }
        ) {
            dynamic_states.push(vk::DynamicState::PRIMITIVE_TOPOLOGY);
        }

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        // ------ Vertex state ------

        let vertex_input = create_info.vertex_input;
        let vertex_attribute_count = vertex_input.attributes.len();
        let vertex_buffer_count = vertex_input.buffers.len();

        let mut vertex_attribute_descriptions = Vec::with_capacity(vertex_attribute_count);
        let mut vertex_binding_descriptions = Vec::with_capacity(vertex_buffer_count);

        for attribute in vertex_input.attributes.iter() {
            vertex_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: attribute.format,
                offset: attribute.offset,
            });
        }

        for desc in vertex_input.buffers.iter() {
            vertex_binding_descriptions.push(vk::VertexInputBindingDescription {
                binding: desc.binding,
                stride: desc.stride,
                input_rate: desc.input_rate.into(),
            });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_buffer_count as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_count as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vertex_input.topology.into(),
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        // ------ Shader stages ------
        let mut stages = Vec::new();
        match create_info.pre_rasterization_shaders {
            PreRasterizationShaders::PrimitiveShading {
                vertex,
                tess_control,
                tess_evaluation,
                geometry,
            } => {
                let vertex = self.create_shader_module(ShaderKind::Vertex, &vertex.code, vertex.entry_point)?;
                let tess_control = tess_control
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderKind::TessControl, &t.code, t.entry_point))
                    .transpose()?;
                let tess_evaluation = tess_evaluation
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderKind::TessEvaluation, &t.code, t.entry_point))
                    .transpose()?;
                let geometry = geometry
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderKind::Geometry, &t.code, t.entry_point))
                    .transpose()?;

                stages.push(vk::PipelineShaderStageCreateInfo {
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vertex,
                    p_name: b"main\0".as_ptr() as *const c_char, // TODO
                    p_specialization_info: ptr::null(),
                    ..Default::default()
                });
                if let Some(tess_control) = tess_control {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TESSELLATION_CONTROL,
                        module: tess_control,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
                if let Some(tess_evaluation) = tess_evaluation {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                        module: tess_evaluation,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
                if let Some(geometry) = geometry {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::GEOMETRY,
                        module: geometry,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
            }
            PreRasterizationShaders::MeshShading { mesh, task } => {
                if let Some(task) = task {
                    let task = self.create_shader_module(ShaderKind::Task, &task.code, task.entry_point)?;
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TASK_EXT,
                        module: task,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }

                let mesh = self.create_shader_module(ShaderKind::Mesh, &mesh.code, mesh.entry_point)?;
                stages.push(vk::PipelineShaderStageCreateInfo {
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::MESH_EXT,
                    module: mesh,
                    p_name: b"main\0".as_ptr() as *const c_char, // TODO
                    p_specialization_info: ptr::null(),
                    ..Default::default()
                });
            }
        };

        let fragment = self.create_shader_module(
            ShaderKind::Fragment,
            &create_info.fragment_shader.code,
            create_info.fragment_shader.entry_point,
        )?;
        stages.push(vk::PipelineShaderStageCreateInfo {
            flags: Default::default(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fragment,
            p_name: b"main\0".as_ptr() as *const c_char, // TODO
            p_specialization_info: ptr::null(),
            ..Default::default()
        });

        let attachment_states: Vec<_> = create_info
            .fragment_output
            .color_targets
            .iter()
            .map(|target| match target.blend_equation {
                None => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::FALSE,
                    color_write_mask: target.color_write_mask.into(),
                    ..Default::default()
                },
                Some(blend_equation) => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::TRUE,
                    src_color_blend_factor: blend_equation.src_color_blend_factor.into(),
                    dst_color_blend_factor: blend_equation.dst_color_blend_factor.into(),
                    color_blend_op: blend_equation.color_blend_op.into(),
                    src_alpha_blend_factor: blend_equation.src_alpha_blend_factor.into(),
                    dst_alpha_blend_factor: blend_equation.dst_alpha_blend_factor.into(),
                    alpha_blend_op: blend_equation.alpha_blend_op.into(),
                    color_write_mask: target.color_write_mask.into(),
                },
            })
            .collect();

        // Rasterization state
        let line_rasterization_state = vk::PipelineRasterizationLineStateCreateInfoEXT {
            line_rasterization_mode: create_info.rasterization.line_rasterization.mode.into(),
            stippled_line_enable: vk::FALSE,
            line_stipple_factor: 0,
            line_stipple_pattern: 0,
            ..Default::default()
        };

        let conservative_rasterization_state = vk::PipelineRasterizationConservativeStateCreateInfoEXT {
            p_next: &line_rasterization_state as *const _ as *const _,
            conservative_rasterization_mode: create_info.rasterization.conservative_rasterization_mode.into(),
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            p_next: &conservative_rasterization_state as *const _ as *const _,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: create_info.rasterization.polygon_mode.into(),
            cull_mode: create_info.rasterization.cull_mode.into(),
            front_face: create_info.rasterization.front_face.into(),
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: create_info.fragment_output.multisample.alpha_to_coverage_enabled.into(),
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            flags: Default::default(),
            logic_op_enable: vk::FALSE,
            logic_op: Default::default(),
            attachment_count: attachment_states.len() as u32,
            p_attachments: attachment_states.as_ptr(),
            blend_constants: create_info.fragment_output.blend_constants,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            flags: Default::default(),
            depth_test_enable: (create_info.depth_stencil.depth_compare_op != CompareOp::Always).into(),
            depth_write_enable: create_info.depth_stencil.depth_write_enable.into(),
            depth_compare_op: create_info.depth_stencil.depth_compare_op.into(),
            stencil_test_enable: create_info.depth_stencil.stencil_state.is_enabled().into(),
            front: create_info.depth_stencil.stencil_state.front.into(),
            back: create_info.depth_stencil.stencil_state.back.into(),
            depth_bounds_test_enable: vk::FALSE,
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
            ..Default::default()
        };

        let rendering_info = vk::PipelineRenderingCreateInfo {
            view_mask: 0,
            color_attachment_count: create_info.fragment_output.color_attachment_formats.len() as u32,
            p_color_attachment_formats: create_info.fragment_output.color_attachment_formats.as_ptr(),
            depth_attachment_format: create_info.fragment_output.depth_attachment_format.unwrap_or_default(),
            stencil_attachment_format: create_info
                .fragment_output
                .stencil_attachment_format
                .unwrap_or_default(),
            ..Default::default()
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &rendering_info as *const _ as *const _,
            flags: Default::default(),
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &Default::default(),
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_create_info,
            layout: pipeline_layout,
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .inner
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(GraphicsPipeline::new(self.clone(), pipeline, pipeline_layout))
    }
}
