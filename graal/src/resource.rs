use crate::{
    context::{get_vk_sample_count, is_write_access, SemaphoreWait},
    serial::{QueueSerialNumbers, SubmissionNumber},
    vk,
    vk::Handle,
    Device,
};
use gpu_allocator::MemoryLocation;
use slotmap::{Key, SlotMap};
use std::{
    ffi::{c_void, CString},
    mem,
    ptr::NonNull,
};
use tracing::{trace, trace_span};

/// Computes the number of mip levels for a 2D image of the given size.
///
/// # Examples
///
/// ```
/// use graal::get_mip_level_count;
/// assert_eq!(get_mip_level_count(512, 512), 9);
/// assert_eq!(get_mip_level_count(512, 256), 9);
/// assert_eq!(get_mip_level_count(511, 256), 8);
/// ```
pub fn get_mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

#[derive(Copy, Clone, Debug)]
pub struct AllocationRequirements {
    pub(crate) mem_req: vk::MemoryRequirements,
    pub(crate) location: gpu_allocator::MemoryLocation,
    //pub(crate) required_flags: vk::MemoryPropertyFlags,
    //pub(crate) preferred_flags: vk::MemoryPropertyFlags,
}

impl AllocationRequirements {
    /// Returns a copy of these allocation requirements, adjusted so that they fit `other`.
    /// If these requirements cannot be adjusted, returns `None`.
    ///
    /// Adjustment succeeds when the memory location and memory type bits are the same between the two requirements.
    /// In this case, size and alignment are set to the biggest of the two.
    pub(crate) fn adjusted_requirements(
        &self,
        other: &AllocationRequirements,
    ) -> Option<AllocationRequirements> {
        if self.location != other.location {
            return None;
        }
        if self.mem_req.memory_type_bits != other.mem_req.memory_type_bits {
            return None;
        }
        let mut adjusted = *self;
        adjusted.mem_req.alignment = adjusted.mem_req.alignment.max(other.mem_req.alignment);
        adjusted.mem_req.size = adjusted.mem_req.size.max(other.mem_req.size);
        Some(adjusted)
    }
}

/// Information passed to `Context::create_image` to describe the image to be created.
#[derive(Copy, Clone, Debug)]
pub struct ImageResourceCreateInfo {
    /// Dimensionality of the image.
    pub image_type: vk::ImageType,
    /// Image usage flags. Must include all intended uses of the image.
    pub usage: vk::ImageUsageFlags,
    /// Format of the image.
    pub format: vk::Format,
    /// Size of the image.
    pub extent: vk::Extent3D,
    /// Number of mipmap levels. Note that the mipmaps contents must still be generated manually. Default is 1. 0 is *not* a valid value.
    pub mip_levels: u32,
    /// Number of array layers. Default is 1. 0 is *not* a valid value.
    pub array_layers: u32,
    /// Number of samples. Default is 1. 0 is *not* a valid value.
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

slotmap::new_key_type! {
    /// Identifies a GPU resource (buffer or image).
    pub struct ResourceId;

    /// Identifies a resource group.
    pub struct ResourceGroupId;
}

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BufferId(pub(crate) ResourceId);

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub(crate) ResourceId);

#[derive(Debug)]
pub(crate) struct ImageResource {
    pub(crate) handle: vk::Image,
    pub(crate) format: vk::Format,
}

#[derive(Debug)]
pub(crate) struct BufferResource {
    pub(crate) handle: vk::Buffer,
}

#[derive(Debug)]
pub(crate) enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ResourceTrackingInfo {
    pub(crate) first_access: SubmissionNumber,
    pub(crate) owner_queue_family: u32,
    pub(crate) readers: QueueSerialNumbers,
    pub(crate) writer: SubmissionNumber,
    pub(crate) layout: vk::ImageLayout,
    /// Access types for the last write to this resource that have yet to be made available.
    /// This is only relevant for the writer queue, as accesses from concurrent queues are synchronized
    /// with a semaphore that automatically makes all writes visible.
    pub(crate) availability_mask: vk::AccessFlags,
    /// Which access types can see the last write to the resource.
    /// This is only relevant for the writer queue, as accesses from concurrent queues are synchronized
    /// with a semaphore that automatically makes all writes visible.
    pub(crate) visibility_mask: vk::AccessFlags,
    /// The stages that last accessed the resource. Valid only on the writer queue.
    pub(crate) stages: vk::PipelineStageFlags,
    /// The binary semaphore to wait for before accessing the resource.
    pub(crate) wait_binary_semaphore: vk::Semaphore,
}

impl ResourceTrackingInfo {
    pub(crate) fn has_writer(&self) -> bool {
        self.writer.is_valid()
    }

    pub(crate) fn has_readers(&self) -> bool {
        self.readers.iter().any(|&x| x != 0)
    }

    pub(crate) fn clear_readers(&mut self) {
        self.readers = Default::default();
    }
}

impl Default for ResourceTrackingInfo {
    fn default() -> Self {
        ResourceTrackingInfo {
            first_access: Default::default(),
            owner_queue_family: vk::QUEUE_FAMILY_IGNORED,
            readers: Default::default(),
            writer: Default::default(),
            layout: Default::default(),
            availability_mask: Default::default(),
            visibility_mask: Default::default(),
            stages: Default::default(),
            wait_binary_semaphore: Default::default(),
        }
    }
}

/// Describes how a resource got its memory.
#[derive(Clone, Debug)]
pub enum ResourceAllocation {
    /// a block of memory exclusively for this resource.
    Default {
        allocation: gpu_allocator::vulkan::Allocation,
    },

    /// Memory aliasing: allocate a block of memory for the resource, which can possibly be shared
    /// with other aliasable resources if their lifetimes do not overlap.
    Transient {
        device_memory: vk::DeviceMemory,
        offset: vk::DeviceSize,
    },

    /// The memory for this resource was imported or exported from/to an external handle.
    External { device_memory: vk::DeviceMemory },
}

/// Specifies the kind of ownership held on a resource.
#[derive(Clone, Debug)]
pub enum ResourceOwnership {
    /// We own the resource and are responsible for its deletion.
    OwnedResource {
        requirements: AllocationRequirements,
        allocation: Option<ResourceAllocation>,
    },
    /// We are referencing an external resource which we do not own (e.g. a swapchain image).
    External,
}

#[derive(Debug)]
pub(crate) struct Resource {
    /// Name, for debugging purposes
    pub(crate) name: String,
    /// Whether this pass has been discarded during the last frame.
    pub(crate) discarded: bool,
    /// Who owns the resource.
    pub(crate) ownership: ResourceOwnership,
    /// Details specific to the kind of resource (buffer or image).
    pub(crate) kind: ResourceKind,
    /// The group that the resource belongs to.
    pub(crate) group: Option<ResourceGroupId>,
    pub(crate) tracking: ResourceTrackingInfo,
}

impl Resource {
    pub(crate) fn image(&self) -> &ImageResource {
        match &self.kind {
            ResourceKind::Image(r) => r,
            _ => panic!("expected an image resource"),
        }
    }

    pub(crate) fn image_mut(&mut self) -> &mut ImageResource {
        match &mut self.kind {
            ResourceKind::Image(r) => r,
            _ => panic!("expected an image resource"),
        }
    }

    pub(crate) fn buffer(&self) -> &BufferResource {
        match &self.kind {
            ResourceKind::Buffer(r) => r,
            _ => panic!("expected a buffer resource"),
        }
    }

    pub(crate) fn buffer_mut(&mut self) -> &mut BufferResource {
        match &mut self.kind {
            ResourceKind::Buffer(r) => r,
            _ => panic!("expected a buffer resource"),
        }
    }

    /// Sets the resource allocation for resources with delayed allocations.
    pub(crate) fn set_allocation(&mut self, alloc: ResourceAllocation) {
        // set the allocation type on the resource object
        match self.ownership {
            ResourceOwnership::OwnedResource {
                ref mut allocation, ..
            } => {
                assert!(allocation.is_none());
                *allocation = Some(alloc)
            }
            _ => panic!("trying to set an allocation on an unowned object"),
        }
    }
}

pub(crate) type ResourceMap = SlotMap<ResourceId, Resource>;

/// Destroys a resource and frees its device memory if it was allocated for this resource
/// exclusively.
unsafe fn destroy_resource(device: &Device, resource: &mut Resource) {
    // deallocate its memory, if it was allocated for this object exclusively
    match resource.ownership {
        ResourceOwnership::OwnedResource {
            ref mut allocation, ..
        } => {
            // destroy the object, if we're responsible for it (we're not responsible of destroying
            // swapchain images, for example, since they are destroyed with the swapchain).
            match &mut resource.kind {
                ResourceKind::Buffer(buf) => {
                    device
                        .device
                        .destroy_buffer(mem::take(&mut buf.handle), None);
                }
                ResourceKind::Image(img) => {
                    device
                        .device
                        .destroy_image(mem::take(&mut img.handle), None);
                }
            }

            // free the memory associated to the object
            match allocation.take() {
                Some(ResourceAllocation::Default { allocation }) => {
                    device.allocator.borrow_mut().free(allocation).unwrap()
                }
                _ => {
                    // External: the memory is freed elsewhere (?)
                    // Transient: the memory is freed when waiting for a frame to finish
                    // No allocation: nothing to deallocate
                }
            }
        }
        _ => {}
    }
}

/// Holds information about a buffer resource.
#[derive(Copy, Clone, Debug)]
pub struct BufferInfo {
    /// ID of the buffer resource.
    pub id: BufferId,
    /// Vulkan handle of the buffer.
    pub handle: vk::Buffer,
    /// If the buffer is mapped in client memory, holds a pointer to the mapped range. Null otherwise.
    // TODO: Option<NonNull>
    pub mapped_ptr: Option<NonNull<c_void>>,
}

/// Holds information about an image resource.
#[derive(Copy, Clone, Debug)]
pub struct ImageInfo {
    /// ID of the image resource.
    pub id: ImageId,
    /// Vulkan handle of the image.
    pub handle: vk::Image,
}

#[derive(Clone, Debug)]
pub struct ResourceRegistrationInfo<'a> {
    pub name: &'a str,
    pub ownership: ResourceOwnership,
    pub initial_wait: Option<SemaphoreWait>,
}

#[derive(Clone, Debug)]
pub struct ImageRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
    pub handle: vk::Image,
    pub format: vk::Format,
}

#[derive(Clone, Debug)]
pub struct BufferRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
    pub handle: vk::Buffer,
}

pub(crate) struct ResourceGroup {
    pub(crate) wait_serials: QueueSerialNumbers,
    // ignored if waiting on multiple queues
    pub(crate) src_stage_mask: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask: vk::PipelineStageFlags,
    // ignored if waiting on multiple queues
    pub(crate) src_access_mask: vk::AccessFlags,
    pub(crate) dst_access_mask: vk::AccessFlags,
}

/// Helper function to associate a debug name to a vulkan handle.
fn set_debug_object_name(
    device: &Device,
    object_type: vk::ObjectType,
    object_handle: u64,
    name: &str,
    serial: Option<u64>,
) {
    unsafe {
        let name = if let Some(serial) = serial {
            format!("{}@{}", name, serial)
        } else {
            name.to_string()
        };
        let object_name = CString::new(name.as_str()).unwrap();
        device
            .vk_ext_debug_utils
            .debug_utils_set_object_name(
                device.device.handle(),
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

pub(crate) struct DeviceResources {
    pub(crate) resources: ResourceMap,
    pub(crate) resource_groups: slotmap::SlotMap<ResourceGroupId, ResourceGroup>,
}

impl DeviceResources {
    pub(crate) fn new() -> DeviceResources {
        DeviceResources {
            resources: SlotMap::with_key(),
            resource_groups: SlotMap::with_key(),
        }
    }

    unsafe fn register_resource(
        &mut self,
        device: &Device,
        info: ResourceRegistrationInfo,
        kind: ResourceKind,
    ) -> ResourceId {
        let (object_type, object_handle) = match kind {
            ResourceKind::Buffer(ref buf) => (vk::ObjectType::BUFFER, buf.handle.as_raw()),
            ResourceKind::Image(ref img) => (vk::ObjectType::IMAGE, img.handle.as_raw()),
        };

        let id = self.resources.insert(Resource {
            name: info.name.to_string(),
            discarded: false,
            tracking: Default::default(),
            kind,
            ownership: info.ownership,
            group: None,
        });

        set_debug_object_name(device, object_type, object_handle, info.name, None);

        id
    }

    /// Frees or recycles resources used by frames that have completed and that have no user
    /// references.
    pub(crate) fn cleanup_resources(
        &mut self,
        device: &Device,
        completed_serials: QueueSerialNumbers,
    ) {
        let _ = trace_span!("cleanup_resources");
        // we retain only resources that have a non-zero user refcount (the user is still holding a reference to the resource),
        // and resources that have reader or writer passes that have not yet completed
        self.resources.retain(|id, r| {
            // refcount != 0 OR any reader not completed OR writer not completed
            let keep = !r.discarded
                || r.tracking.readers > completed_serials
                || r.tracking.writer.serial() > completed_serials.serial(r.tracking.writer.queue());
            if !keep {
                trace!(?id, name = r.name.as_str(), tracking=?r.tracking, "destroy_resource");
                unsafe {
                    // Safety: we know that all serials <= `self.completed_serials` have finished
                    destroy_resource(device, r);
                }
            }
            keep
        })
    }

    /// Finds the ID of the resource that corresponds to the specified image handle.
    ///
    /// Returns `ResourceId::null()` if `handle` doesn't refer to a resource managed by this context.
    pub(crate) fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
        self.resources
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
            .unwrap_or(ResourceId::null())
    }

    /// Finds the ID of the resource that corresponds to the specified buffer handle.
    ///
    /// Returns `ResourceId::null()` if `handle` doesn't refer to a resource managed by this context.
    pub(crate) fn buffer_resource_by_handle(&self, handle: vk::Buffer) -> ResourceId {
        self.resources
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
            .unwrap_or(ResourceId::null())
    }
}

impl Device {
    pub(crate) fn cleanup_resources(&self, completed_serials: QueueSerialNumbers) {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.cleanup_resources(&self, completed_serials)
    }

    unsafe fn register_resource(
        &self,
        info: ResourceRegistrationInfo,
        kind: ResourceKind,
    ) -> ResourceId {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.register_resource(&self, info, kind)
    }

    /// Registers an existing buffer resource in the context.
    pub unsafe fn register_buffer_resource(&self, info: BufferRegistrationInfo) -> BufferId {
        let id = self.register_resource(
            info.resource,
            ResourceKind::Buffer(BufferResource {
                handle: info.handle,
            }),
        );
        BufferId(id)
    }

    /// Registers an existing image resource in the context.
    pub unsafe fn register_image_resource(&self, info: ImageRegistrationInfo) -> ImageId {
        let id = self.register_resource(
            info.resource,
            ResourceKind::Image(ImageResource {
                handle: info.handle,
                format: info.format,
            }),
        );
        ImageId(id)
    }

    /// Marks the image as ready to be deleted.
    ///
    /// The actual destruction of the resource is delayed until all passes referencing this resource
    /// have finished execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # use graal::{ImageInfo, MemoryLocation, vk, ImageResourceCreateInfo};
    /// # let mut context = graal::Context::new();
    ///
    /// context.start_frame(Default::default());
    ///
    /// // create the image resource.
    /// # let image_resource_create_info : ImageResourceCreateInfo = todo!();
    /// let image_info = context.create_image("target", MemoryLocation::GpuOnly, &image_resource_create_info);
    ///
    /// // reference the resource in pass P1, in the current frame.
    /// context.add_graphics_pass("P1", |pass| {
    ///
    ///     pass.reference_image(image_info.id,
    ///             vk::AccessFlags::TRANSFER_WRITE,    // whatever
    ///             vk::PipelineStageFlags::TRANSFER,
    ///             vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    ///             vk::ImageLayout::TRANSFER_DST_OPTIMAL);
    ///     // ... do stuff in the pass ...
    /// });
    ///
    /// // We won't be using the image in subsequent passes, we can ask to destroy it.
    /// // The resource will be destroyed once P1 has finished executing.
    /// context.destroy_image(image_info.id);
    /// context.end_frame();
    /// ```
    pub fn destroy_image(&self, id: ImageId) {
        // resources are really destroyed during `Context::cleanup_resources`, which checks that
        // all passes referencing this resource have finished executing.
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resources.get_mut(id.0).unwrap().discarded = true;
    }

    /// Marks the buffer as unused and ready to be deleted.
    ///
    /// The resource will be destroyed once all passes referencing this resource
    /// have finished execution.
    ///
    /// # Examples
    ///
    /// ```
    /// # use graal::{BufferInfo, MemoryLocation, vk, BufferResourceCreateInfo};
    /// # let mut context = graal::Context::new();
    ///
    /// context.start_frame(Default::default());
    ///
    /// // create the buffer resource.
    /// # let buffer_resource_create_info : BufferResourceCreateInfo = todo!();
    /// let buffer_info = context.create_buffer("target", MemoryLocation::GpuOnly,
    ///                                         &buffer_resource_create_info);
    ///
    /// // reference the resource in pass P1, in the current frame.
    /// context.add_graphics_pass("P1", |pass| {
    ///     pass.reference_buffer(buffer_info.id,
    ///             vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
    ///             vk::PipelineStageFlags::VERTEX_SHADER);
    ///     // ... do stuff with the buffer in the pass ...
    /// });
    ///
    /// // We won't be using the buffer in subsequent passes, we can ask to destroy it.
    /// // The resource will be destroyed once P1 has finished executing.
    /// context.destroy_buffer(buffer_info.id);
    /// context.end_frame();
    /// ```
    pub fn destroy_buffer(&self, id: BufferId) {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resources.get_mut(id.0).unwrap().discarded = true;
    }

    /// Creates a resource group.
    pub fn create_resource_group(
        &self,
        dst_stage_mask: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
    ) -> ResourceGroupId {
        // resource groups are for read-only resources
        assert!(!is_write_access(dst_access_mask));
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resource_groups.insert(ResourceGroup {
            wait_serials: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }

    /// Destroys a resource group.
    pub fn destroy_resource_group(&self, group_id: ResourceGroupId) {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resource_groups.remove(group_id);
    }

    /// Creates a new image resource.
    ///
    /// Returns an `ImageInfo` struct containing the image resource ID and the vulkan image handle.
    ///
    /// # Notes
    /// The image might not have any device memory attached when this function returns.
    /// This is because graal may delay the allocation and binding of device memory until the end of the
    /// current frame (see `Context::end_frame`).
    ///
    /// # Examples
    ///
    /// ```
    /// # use graal::{ImageInfo, MemoryLocation, vk, ImageResourceCreateInfo};
    /// # let mut context = graal::Context::new();
    ///
    /// // Create a 512x512 RGBA16F image that will serve as both a color attachment and a sampled texture.
    /// let ImageInfo { id, handle } = context.create_image("texture", MemoryLocation::GpuOnly, &ImageResourceCreateInfo {
    ///     image_type: vk::ImageType::TYPE_2D,
    ///     usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::COLOR_ATTACHMENT,
    ///     format: vk::Format::R16G16B16A16_SFLOAT,
    ///     extent: vk::Extent3D {
    ///         width: 512,
    ///         height: 512,
    ///         depth: 1,
    ///     },
    ///     mip_levels: 1,
    ///     array_layers: 1,
    ///     samples: 1,
    ///     tiling: Default::default(),
    /// });
    /// ```
    ///
    /// Whether the resource should live only for the duration of the frame it's used in.
    /// When the frame that uses the resource completes, the resource is automatically deleted.
    /// The resource can only be used in one frame.
    pub fn create_image(
        &self,
        name: &str,
        location: MemoryLocation,
        image_info: &ImageResourceCreateInfo,
    ) -> ImageInfo {
        // for now all resources are CONCURRENT, because that's the only way they can
        // be read across multiple queues.
        // Maybe exclusive ownership will be needed at some point, but then we should prevent
        // them from being used across multiple queues. I know that there's the possibility of doing
        // a "queue ownership transfer", but that shit is incomprehensible.
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.image_type,
            format: image_info.format,
            extent: image_info.extent,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: image_info.tiling,
            usage: image_info.usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: self.queues_info.queue_count as u32,
            p_queue_family_indices: self.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.device.get_image_memory_requirements(handle) };
        // register the resource in the context
        let id = unsafe {
            self.register_image_resource(ImageRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    ownership: ResourceOwnership::OwnedResource {
                        requirements: AllocationRequirements { mem_req, location },
                        allocation: None,
                    },
                    initial_wait: None,
                },
                handle,
                format: image_info.format,
            })
        };

        ImageInfo { id, handle }
    }

    /// Creates a new buffer resource.
    ///
    /// Returns a `BufferInfo` struct containing the buffer resource ID, the vulkan buffer handle,
    /// and a pointer to the buffer mapped in host memory, if `buffer_create_info.map_on_create == true`.
    ///
    /// # Notes
    /// If `map_on_create` is specified in `BufferResourceCreateInfo`, the returned vulkan buffer
    /// is guaranteed to have a block of device memory attached to it after this function returns
    /// (i.e. a call `vkBindBufferMemory` has been made for this buffer).
    ///
    /// Otherwise, graal can opt to delay the allocation and binding of device memory for this buffer until the
    /// end of the current frame for optimization purposes (see `Context::end_frame`).
    /// In this case, the created buffer object might not have any device memory attached when this function returns.
    ///
    /// # Examples
    ///
    /// ```
    /// # use graal::{BufferInfo, BufferResourceCreateInfo, MemoryLocation, vk};
    /// # let mut context = graal::Context::new();
    ///
    /// // Create a staging buffer for uploading data to the GPU
    /// let BufferInfo { id, handle, mapped_ptr } = context.create_buffer("staging", MemoryLocation::CpuToGpu, &BufferResourceCreateInfo {
    ///     usage: vk::BufferUsageFlags::TRANSFER_SRC,
    ///     byte_size: 1024,
    ///     map_on_create: true,    // ensures that mapped_ptr is not empty
    /// });
    /// ```
    pub fn create_buffer(
        &self,
        name: &str,
        location: MemoryLocation,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {
        // create the buffer object first
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: buffer_create_info.byte_size,
            usage: buffer_create_info.usage,
            sharing_mode: if self.queues_info.queue_count == 1 {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            queue_family_index_count: self.queues_info.queue_count as u32,
            p_queue_family_indices: self.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };

        // get its memory requirements
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(handle) };

        let (ownership, mapped_ptr) = if !buffer_create_info.map_on_create {
            // We can delay allocation only if the user requests a transient resource and
            // if the resource does not need to be mapped immediately.
            let ownership = ResourceOwnership::OwnedResource {
                requirements: AllocationRequirements { mem_req, location },
                allocation: None,
            };
            (/* ownership */ ownership, /* mapped_ptr */ None)
        } else {
            // caller requested a mapped pointer, must create and allocate immediately
            let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements: mem_req,
                location,
                linear: true,
            };
            let allocation = self
                .allocator
                .borrow_mut()
                .allocate(&allocation_create_desc)
                .expect("failed to allocate device memory");
            unsafe {
                self.device
                    .bind_buffer_memory(handle, allocation.memory(), allocation.offset() as u64)
                    .unwrap();
            }
            let mapped_ptr = allocation.mapped_ptr();
            let ownership = ResourceOwnership::OwnedResource {
                requirements: AllocationRequirements { mem_req, location },
                allocation: Some(ResourceAllocation::Default { allocation }),
            };
            /*let mapped_ptr = if buffer_create_info.map_on_create {
                let ptr = allocation.mapped_ptr().expect("failed to map buffer");
                //assert!(!ptr.is_null(), "failed to map buffer");
                ptr.as_ptr() as *mut u8
            } else {
                ptr::null_mut()
            };*/

            (ownership, mapped_ptr)
        };

        let id = unsafe {
            self.register_buffer_resource(BufferRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    initial_wait: None,
                    ownership,
                },
                handle,
            })
        };

        BufferInfo {
            id,
            handle,
            mapped_ptr,
        }
    }

    /// Returns the handle of the corresponding image resource.
    /// Panics if `id` does not refer to an image resource.
    pub fn image_handle(&self, id: ImageId) -> vk::Image {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resources.get(id.0).unwrap().image().handle
    }

    /// Returns the handle of the corresponding buffer resource.
    /// Panics if `id` does not refer to a buffer resource.
    pub fn buffer_handle(&self, id: BufferId) -> vk::Buffer {
        let mut resources = self.resources.lock().expect("failed to lock resources");
        resources.resources.get(id.0).unwrap().buffer().handle
    }
}
