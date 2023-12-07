//! Abstractions over a vulkan device.
mod init;
mod resource;

use crate::{
    device::resource::ResourceGroup,
    instance::{vk_khr_debug_utils, vk_khr_surface},
    platform_impl,
    queue::{Queue, SemaphoreWait, SignaledSemaphore},
};
use ash::vk;
use slotmap::{Key, SlotMap};
use std::{
    cell::{Cell, RefCell},
    ffi::CString,
    fmt,
    ops::Deref,
    os::raw::c_void,
    ptr,
    ptr::NonNull,
    rc::Rc,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around a vulkan device, associated queues and tracked resources.
pub(crate) struct DeviceInner {
    /// Underlying vulkan device
    pub(crate) device: ash::Device,
    /// Platform-specific extension functions
    pub(crate) platform_extensions: platform_impl::PlatformExtensions,
    pub(crate) physical_device: vk::PhysicalDevice,
    pub(crate) physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    pub(crate) physical_device_properties: vk::PhysicalDeviceProperties,
    //pub(crate) physical_device_features: vk::PhysicalDeviceFeatures,
    queues: Vec<QueueData>,
    pub(crate) allocator: RefCell<gpu_allocator::vulkan::Allocator>,
    pub(crate) vk_khr_swapchain: ash::extensions::khr::Swapchain,
    pub(crate) vk_ext_shader_object: ash::extensions::ext::ShaderObject,

    pub(crate) resources: RefCell<ResourceMap>,
    pub(crate) resource_groups: RefCell<ResourceGroupMap>,
    deletion_lists: RefCell<Vec<DeferredDeletionList>>,
}

#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Rc<DeviceInner>,
}

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

#[derive(thiserror::Error, Debug)]
pub enum DeviceCreateError {
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Feature {}

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

/// Holds information about a buffer resource.
#[derive(Copy, Clone, Debug)]
pub struct BufferHandle {
    /// ID of the buffer resource.
    pub id: BufferId,
    /// Vulkan handle of the buffer.
    pub vk: vk::Buffer,
    /// If the buffer is mapped in client memory, holds a pointer to the mapped range. Null otherwise.
    pub mapped_ptr: Option<NonNull<c_void>>,
}

/// Holds information about an image resource.
#[derive(Copy, Clone, Debug)]
pub struct ImageHandle {
    /// ID of the image resource.
    pub id: ImageId,
    /// Vulkan handle of the image.
    pub vk: vk::Image,
}

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
    pub images: Vec<vk::Image>,
}

/// Contains information about an image in a swapchain.
#[derive(Debug)]
pub struct SwapchainImage {
    /// Handle of the swapchain that owns this image.
    pub swapchain: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub index: u32,
    pub handle: ImageHandle,
}

/// Information passed to `Context::create_image` to describe the image to be created.
#[derive(Copy, Clone, Debug)]
pub struct ImageResourceCreateInfo {
    /// Dimensionality of the image.
    pub type_: vk::ImageType,
    /// Image usage flags. Must include all intended uses of the image.
    pub usage: vk::ImageUsageFlags,
    /// Format of the image.
    pub format: vk::Format,
    /// Size of the image.
    pub extent: vk::Extent3D,
    /// Number of mipmap levels. Note that the mipmaps contents must still be generated manually. Default is 1. 0 is *not* a valid value.
    pub mip_levels: u32,
    /// Number of array layers. Default is `1`. `0` is *not* a valid value.
    pub array_layers: u32,
    /// Number of samples. Default is `1`. `0` is *not* a valid value.
    pub samples: u32,
    /// Tiling.
    pub tiling: vk::ImageTiling,
}

/// Information passed to `Context::create_buffer` to describe the buffer to be created.
#[derive(Copy, Clone, Debug)]
pub struct BufferResourceCreateInfo {
    /// Usage flags. Must include all intended uses of the buffer.
    pub usage: vk::BufferUsageFlags,
    /// Size of the buffer in bytes.
    pub byte_size: u64,
    /// Whether the memory for the resource should be mapped for host access immediately.
    /// If this flag is set, `create_buffer` will also return a pointer to the mapped buffer.
    /// This flag is ignored for resources that can't be mapped.
    pub map_on_create: bool,
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
pub enum DeferredDeletionObject {
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

////////////////////////////////////////////////////////////////////////////////////////////////////

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Device").finish()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SWAPCHAIN STUFF

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

impl Device {
    /// Creates a swapchain object.
    pub unsafe fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        size: (u32, u32),
    ) -> Swapchain {
        let mut swapchain = Swapchain {
            handle: Default::default(),
            surface,
            images: vec![],
            format,
        };
        self.resize_swapchain(&mut swapchain, size);
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
    pub unsafe fn resize_swapchain(&self, swapchain: &mut Swapchain, size: (u32, u32)) {
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
        let image_extent = init::get_preferred_swap_extent(size, &capabilities);
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
        swapchain.images = self
            .inner
            .vk_khr_swapchain
            .get_swapchain_images(swapchain.handle)
            .unwrap();
    }

    /*/// Returns the current values of all queue timelines (i.e. the "progress" value on the device).
    pub fn get_timeline_values(&self) -> Result<TimelineValues, vk::Result> {
        let mut values = [0; MAX_QUEUES];
        for i in 0..self.queues_info.queue_count {
            values[i] = unsafe { self.device.get_semaphore_counter_value(self.queues_info.timelines[i])? };
        }
        Ok(TimelineValues(values))
    }*/

    /*fn update_timeline_values(&self) {
        for (i, &timeline) in self.queues_info.timelines.iter().enumerate() {
            self.completed_values[i] = unsafe { self.device.get_semaphore_counter_value(timeline) }.unwrap();
        }
    }*/
}

pub unsafe fn create_device_and_queue(
    present_surface: Option<vk::SurfaceKHR>,
) -> Result<(Device, Queue), DeviceCreateError> {
    let device = Device::new(present_surface)?;
    let queue = device.get_queue_by_global_index(0);
    Ok((device, queue))
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
pub(crate) struct ImageResource {
    pub(crate) handle: vk::Image,
    pub(crate) format: vk::Format,
    pub(crate) all_aspects: vk::ImageAspectFlags,
}

#[derive(Debug)]
pub(crate) struct BufferResource {
    pub(crate) handle: vk::Buffer,
}

/// Describes how the resource is access by different queues.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub(crate) enum OwnerQueue {
    /// No lock on the resource.
    None,
    /// The specified queue holds an exclusive lock on the resource.
    Exclusive(usize),
    /// The specified queues (queue mask) are accessing the resource for reading concurrently.
    Concurrent(u16),
}

#[derive(Debug)]
pub(crate) enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Debug)]
pub(crate) struct Resource {
    /// Name, for debugging purposes
    pub(crate) name: String,
    /// Whether this pass has been discarded during the last frame.
    pub(crate) discarded: bool,
    /// The allocation for the resource.
    pub(crate) allocation: ResourceAllocation,
    /// Details specific to the kind of resource (buffer or image).
    pub(crate) kind: ResourceKind,
    /// For frozen resources, the group that the resource belongs to. Otherwise `None` if the resource
    /// is not frozen.
    pub(crate) group: Option<GroupId>,

    /// Access types that need flushing (srcAccessMask).
    pub(crate) flush_mask: vk::AccessFlags2,
    /// Which access types can see the last write to the resource.
    pub(crate) visible: vk::AccessFlags2,
    /// Current image layout if the resource is an image. Ignored otherwise.
    pub(crate) layout: vk::ImageLayout,
    pub(crate) owner: OwnerQueue,
    /// The stages that last accessed the resource. Valid only on the writer queue.
    pub(crate) stages: vk::PipelineStageFlags2,
    /// The binary semaphore to wait for before accessing the resource.
    pub(crate) wait_semaphore: Option<SignaledSemaphore>,

    /// Timestamp of the last access to the resource on its owner queue.
    ///
    /// It's a value of the timeline semaphore of the queue.
    ///
    /// TODO: it's only used for deferred deletion, to ensure that no queue is still using the resource.
    /// We could replace that by a set of semaphore waits in order to support waiting on a resource
    /// used concurrently by multiple queues.
    pub(crate) timestamp: u64,
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

pub(crate) type ResourceMap = SlotMap<ResourceId, Resource>;
pub(crate) type ResourceGroupMap = SlotMap<GroupId, ResourceGroup>;

impl Device {
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

    /// Finds the ID of the resource that corresponds to the specified image handle.
    ///
    /// Returns `ResourceId::null()` if `handle` doesn't refer to a resource managed by this context.
    pub(crate) fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
        self.inner
            .resources
            .borrow()
            .iter()
            .find_map(|(id, r)| match &r.kind {
                ResourceKind::Image(img) => {
                    if img.handle == handle {
                        Some(id)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or_else(ResourceId::null)
    }

    /// Finds the ID of the resource that corresponds to the specified buffer handle.
    ///
    /// Returns `ResourceId::null()` if `handle` doesn't refer to a resource managed by this context.
    pub(crate) fn buffer_resource_by_handle(&self, handle: vk::Buffer) -> ResourceId {
        self.inner
            .resources
            .borrow()
            .iter()
            .find_map(|(id, r)| match &r.kind {
                ResourceKind::Buffer(buf) => {
                    if buf.handle == handle {
                        Some(id)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or_else(ResourceId::null)
    }

    pub fn raw(&self) -> &ash::Device {
        &self.inner.device
    }

    pub fn vk_khr_swapchain(&self) -> &ash::extensions::khr::Swapchain {
        &self.inner.vk_khr_swapchain
    }

    /*
    /// Waits for the specified timeline values to be reached on each queue.
    pub fn wait(&self, timeline_values: &TimelineValues, timeout: Duration) -> Result<(), vk::Result> {
        //let _span = trace_span!("Waiting for serials", ?progress);

        let wait_info = vk::SemaphoreWaitInfo {
            semaphore_count: self.queues_info.queue_count as u32,
            p_semaphores: self.queues_info.timelines.as_ptr(),
            p_values: timeline_values.inner.as_ptr(),
            ..Default::default()
        };

        let timeout_ns = timeout.as_nanos();
        assert!(timeout_ns < u64::MAX as u128, "timeout value too large");
        let timeout_ns = timeout_ns as u64;

        unsafe { self.device.wait_semaphores(&wait_info, timeout_ns) }
    }*/
}
