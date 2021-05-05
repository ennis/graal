use crate::{
    context::{
        external_memory_handle::ExternalMemoryHandle, get_vk_sample_count, pass::Pass,
        set_debug_object_name, QueueSerialNumbers, SubmissionNumber,
    },
    platform_impl, Context, Device,
};
use ash::{version::DeviceV1_0, vk, vk::Handle};
use fixedbitset::FixedBitSet;
use slotmap::{SecondaryMap, SlotMap};
use std::{
    ffi::{OsStr, OsString},
    mem, ptr,
};
use tracing::{trace, trace_span};

slotmap::new_key_type! {
    pub struct ResourceId;
}

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BufferId(pub(crate) ResourceId);

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub(crate) ResourceId);

pub(crate) type ResourceMap = SlotMap<ResourceId, Resource>;

/// Information about the memory to be allocated for a resource.
//
// NOTE: we don't put the "transient" flag here, because it's not _strictly_ a property of the memory
// we're allocating, but a property of the resource itself (its lifetime).
//
// NOTE: this would be the logical place to put an optional "ExternalMemoryHandle", but then all
// functions using `ResourceMemoryInfo` would have to be unsafe, since it's the responsibility of
// the caller to ensure that the external handle is valid.
// Instead, we decided to split resource creation into two functions: one in which we allocate the
// memory for the resource ourselves, for which we can guarantee its safety, and one in which the
// user provides the memory handle, which is unsafe.
// TODO: considering that we do very minimal checks on the createInfos passed to vulkan, it might
// make sense to just make most resource creation functions unsafe.
//
// TODO: "ResourceMemoryInfo" seems like information about already allocated memory, but it's more
// like a request: maybe something in the lines of "ResourceMemoryAllocationInfo" or
// "ResourceMemoryRequirements"?
#[derive(Copy, Clone, Debug, Default)]
pub struct ResourceMemoryInfo {
    /// Required memory property flags. Panics if those cannot be honored (no memory type with those properties).
    pub required_flags: vk::MemoryPropertyFlags,
    /// Preferred memory property flags. The allocator will honor those flags if a memory type with those properties exist, otherwise it will fallback to the required flags.
    pub preferred_flags: vk::MemoryPropertyFlags,
}

impl ResourceMemoryInfo {
    /// TODO docs
    pub const fn new() -> ResourceMemoryInfo {
        ResourceMemoryInfo {
            required_flags: vk::MemoryPropertyFlags::empty(),
            preferred_flags: vk::MemoryPropertyFlags::empty(),
        }
    }

    /// Requires that the resource be allocated in DEVICE_LOCAL memory.
    pub const fn device_local(self) -> Self {
        ResourceMemoryInfo {
            required_flags: vk::MemoryPropertyFlags::from_raw(
                self.required_flags.as_raw() | vk::MemoryPropertyFlags::DEVICE_LOCAL.as_raw(),
            ),
            ..self
        }
    }

    /// Requires that the resource be allocated in HOST_VISIBLE memory.
    pub const fn host_visible(self) -> Self {
        ResourceMemoryInfo {
            required_flags: vk::MemoryPropertyFlags::from_raw(
                self.required_flags.as_raw() | vk::MemoryPropertyFlags::HOST_VISIBLE.as_raw(),
            ),
            ..self
        }
    }

    /// Requires that the resource be allocated in HOST_COHERENT memory.
    pub const fn host_coherent(self) -> Self {
        ResourceMemoryInfo {
            required_flags: vk::MemoryPropertyFlags::from_raw(
                self.required_flags.as_raw() | vk::MemoryPropertyFlags::HOST_COHERENT.as_raw(),
            ),
            ..self
        }
    }

    /// Device-local resource memory. Shorthand for `ResourceMemoryInfo::new().device_local()`.
    pub const DEVICE_LOCAL: ResourceMemoryInfo = ResourceMemoryInfo::new().device_local();

    /// Host-visible resource memory (upload buffers). Shorthand for `ResourceMemoryInfo::new().host_visible()`.
    pub const HOST_VISIBLE: ResourceMemoryInfo = ResourceMemoryInfo::new().host_visible();

    /// Host-visible and coherent resource memory (upload buffers without need for flushes).
    /// Shorthand for `ResourceMemoryInfo::new().host_visible().host_coherent()`.
    pub const HOST_VISIBLE_COHERENT: ResourceMemoryInfo =
        ResourceMemoryInfo::new().host_visible().host_coherent();

    /// Staging buffers (host-visible, preferably coherent)
    pub const STAGING: ResourceMemoryInfo =
        ResourceMemoryInfo::new().host_visible().host_coherent();
}

/// Parameters of a newly created image resource.
#[derive(Copy, Clone, Debug, Default)]
pub struct ImageResourceCreateInfo {
    /// Image type.
    pub image_type: vk::ImageType,
    /// Usage flags.
    pub usage: vk::ImageUsageFlags,
    /// Format of the image.
    pub format: vk::Format,
    /// Size of the image.
    pub extent: vk::Extent3D,
    /// Number of mipmap levels. Note that the mipmaps contents must still be generated manually.
    pub mip_levels: u32,
    /// Number of array layers.
    pub array_layers: u32,
    /// Number of samples.
    pub samples: u32,
    /// Tiling.
    pub tiling: vk::ImageTiling,
}

/// Parameters of a newly created buffer resource.
#[derive(Copy, Clone, Debug, Default)]
pub struct BufferResourceCreateInfo {
    /// Usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Size of the buffer in bytes.
    pub byte_size: u64,
    /// Whether the memory for the resource should be mapped for host access immediately.
    /// If this flag is set, `create_buffer_resource` will also return a pointer to the mapped buffer.
    /// This flag is ignored for resources that can't be mapped.
    pub map_on_create: bool,
}

/// Computes the number of mip levels for a 2D image of the given size.
pub fn get_mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct AllocationRequirements {
    pub(crate) mem_req: vk::MemoryRequirements,
    pub(crate) required_flags: vk::MemoryPropertyFlags,
    pub(crate) preferred_flags: vk::MemoryPropertyFlags,
}

impl AllocationRequirements {
    pub(crate) fn try_adjust(&mut self, other: &AllocationRequirements) -> bool {
        if self.required_flags != other.required_flags {
            return false;
        }
        if self.mem_req.memory_type_bits != other.mem_req.memory_type_bits {
            return false;
        }
        self.mem_req.alignment = self.mem_req.alignment.max(other.mem_req.alignment);
        self.mem_req.size = self.mem_req.size.max(other.mem_req.size);
        true
    }
}

#[derive(Debug)]
pub(crate) struct ImageResource {
    pub(crate) handle: vk::Image,
    pub(crate) format: vk::Format,
}

#[derive(Debug)]
pub(crate) struct BufferResource {
    pub(crate) handle: vk::Buffer,
}

/// Represents a resource access in a pass.
#[derive(Debug)]
pub(crate) struct ResourceAccessDetails {
    pub(crate) initial_layout: vk::ImageLayout,
    pub(crate) final_layout: vk::ImageLayout,
    pub(crate) access_mask: vk::AccessFlags,
    pub(crate) stage_mask: vk::PipelineStageFlags,
}

#[derive(Debug)]
pub(crate) enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ResourceTrackingInfo {
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

/// Describes the kind of memory that is bound to a resource.
#[derive(Debug)]
pub(crate) enum ResourceMemory {
    /// The resource may share a block of memory allocation with other resources.
    Aliasable(AllocationRequirements),
    /// The resource has a block of memory allocated exclusively to it.
    Exclusive(vk_mem::Allocation),
    /// The memory for the resource is managed externally (e.g. swapchain images)
    External,
    /// The memory for this resource was imported from an external handle.
    Imported { device_memory: vk::DeviceMemory },
}

#[derive(Debug)]
pub(crate) struct Resource {
    /// Name, for debugging purposes
    pub(crate) name: String,
    /// User reference count, for uses by clients outside outside of `Context`.
    pub(crate) user_ref_count: usize,
    /// Usage trackers.
    pub(crate) tracking: ResourceTrackingInfo,
    /// The memory bound to the resource.
    pub(crate) memory: ResourceMemory,
    /// Whether the the context should delete the image once it's not in use.
    pub(crate) should_delete: bool,
    /// Details specific to the kind of resource (buffer or image).
    pub(crate) kind: ResourceKind,
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
}

/// Destroys a resource and frees its device memory if it was allocated for this resource
/// exclusively.
unsafe fn destroy_resource(device: &Device, resource: &mut Resource) {
    // destroy the object, if we're responsible for it (we're not responsible of destroying
    // swapchain images, for example, since they are destroyed with the swapchain).
    if resource.should_delete {
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
    }

    // deallocate its memory, if it was allocated for this object exclusively
    match resource.memory {
        // We have our own exclusive device memory block, free it.
        ResourceMemory::Exclusive(allocation) => device.allocator.free_memory(&allocation).unwrap(),
        // External or aliasable memory: the memory is freed elsewhere.
        // For aliasable memory: the memory block is freed when the frame is completed.
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
    pub mapped_ptr: *mut u8,
}

/// Holds information about a buffer resource containing an array of elements of type T.
#[derive(Copy, Clone, Debug)]
pub struct TypedBufferInfo<T> {
    /// ID of the buffer resource.
    pub id: BufferId,
    /// Vulkan handle of the buffer.
    pub handle: vk::Buffer,
    /// If the buffer is mapped in client memory, holds a pointer to the mapped range. Null otherwise.
    pub mapped_ptr: *mut T,
}

/// `TypedBufferInfo<T>` are convertible into their untyped equivalents.
impl<T> From<TypedBufferInfo<T>> for BufferInfo {
    fn from(buf: TypedBufferInfo<T>) -> Self {
        BufferInfo {
            id: buf.id,
            handle: buf.handle,
            mapped_ptr: buf.mapped_ptr as *mut u8,
        }
    }
}

/// Holds information about an image resource.
#[derive(Copy, Clone, Debug)]
pub struct ImageInfo {
    /// ID of the image resource.
    pub id: ImageId,
    /// Vulkan handle of the image.
    pub handle: vk::Image,
}

// --- Reachability matrix -------------------------------------------------------------------------
struct Reachability {
    m: Vec<FixedBitSet>,
}

impl Reachability {
    fn is_reachable(&self, from: usize, to: usize) -> bool {
        self.m[to][from]
    }
}

fn disjoint_index_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b && a < v.len() && b < v.len());
    unsafe {
        (
            &mut *(v.get_unchecked_mut(a) as *mut _),
            &mut *(v.get_unchecked_mut(b) as *mut _),
        )
    }
}

fn compute_reachability(passes: &[Pass]) -> Reachability {
    let len = passes.len();
    let mut m = Vec::new();
    m.resize_with(passes.len(), || FixedBitSet::with_capacity(len));

    for i in 0..len {
        for &p in passes[i].preds.iter() {
            m[i].set(p, true);
            let (mi, mp) = disjoint_index_mut(&mut m, i, p);
            *mi |= &*mp;
        }
    }

    Reachability { m }
}

impl Context {
    /// Frees or recycles resources used by batches that have completed and that have no user
    /// references.
    pub(crate) fn cleanup_resources(&mut self) {
        let _ = trace_span!("cleanup_resources");
        let device = &self.device;
        let completed_serials = self.completed_serials;
        // we retain only resources that have a non-zero user refcount (the user is still holding a reference to the resource),
        // and resources that have reader or writer passes that have not yet completed
        self.resources.retain(|id, r| {
            // refcount != 0 OR any reader not completed OR writer not completed
            let keep = r.user_ref_count != 0
                || r.tracking.readers > completed_serials
                || r.tracking.writer.serial() > completed_serials.serial(r.tracking.writer.queue());
            if !keep {
                trace!(?id, name = r.name.as_str(), "destroy_resource");
                unsafe {
                    // Safety: we know that all serials <= `self.completed_serials` have finished
                    destroy_resource(device, r);
                }
            }
            keep
        })
    }

    // Common logic for create_image and create_image_with_external_memory_handle. These are two
    // different functions because `create_image_with_external_memory_handle` is unsafe, while
    // create_image isn't.
    // In theory, `external_memory_handle` could be put in `ResourceMemoryInfo`, but that would require
    // the function to be unsafe.
    unsafe fn create_image_internal(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        transient: bool,
        external_memory_handle: Option<ExternalMemoryHandle>,
    ) -> ImageInfo {
        // can't be both transient and have an external memory handle
        assert!(transient ^ external_memory_handle.is_some());

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
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.device.device.get_image_memory_requirements(handle) };
        let memory = if let Some(ref external_memory_handle) = external_memory_handle {
            // import external memory and bind it to the resource
            let device_memory = platform_impl::allocate_external_memory(
                &self.device,
                &mem_req,
                memory_info.required_flags,
                memory_info.preferred_flags,
                external_memory_handle,
            );
            unsafe {
                self.device
                    .device
                    .bind_image_memory(handle, device_memory, 0)
                    .unwrap();
            }
            ResourceMemory::Imported { device_memory }
        } else if transient {
            // transient memory
            ResourceMemory::Aliasable(AllocationRequirements {
                mem_req,
                required_flags: memory_info.required_flags,
                preferred_flags: memory_info.preferred_flags,
            })
        } else {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                preferred_flags: memory_info.preferred_flags,
                required_flags: memory_info.required_flags,
                ..Default::default()
            };
            let (alloc, alloc_info) = self
                .device
                .allocator
                .allocate_memory(&mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            unsafe {
                self.device
                    .device
                    .bind_image_memory(
                        handle,
                        alloc_info.get_device_memory(),
                        alloc_info.get_offset() as u64,
                    )
                    .unwrap();
            }
            ResourceMemory::Exclusive(alloc)
        };
        let id = self.resources.insert(Resource {
            name: name.to_string(),
            user_ref_count: if transient { 0 } else { 1 },
            memory,
            tracking: Default::default(),
            should_delete: true,
            kind: ResourceKind::Image(ImageResource {
                handle,
                format: image_info.format,
            }),
        });
        set_debug_object_name(
            &self.device,
            vk::ObjectType::IMAGE,
            handle.as_raw(),
            name,
            transient.then(|| self.next_serial),
        );

        ImageInfo {
            id: ImageId(id),
            handle,
        }
    }

    /// Creates a new image resource.
    ///
    /// Whether the resource should live only for the duration of the frame it's used in.
    /// When the frame that uses the resource completes, the resource is automatically deleted.
    /// The resource can only be used in one frame.
    pub fn create_image(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        transient: bool,
    ) -> ImageInfo {
        unsafe { self.create_image_internal(name, memory_info, image_info, transient, None) }
    }

    /// Creates an image from an external handle.
    pub unsafe fn create_image_with_external_memory_handle(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        external_memory_handle: ExternalMemoryHandle,
    ) -> ImageInfo {
        self.create_image_internal(
            name,
            memory_info,
            image_info,
            false,
            Some(external_memory_handle),
        )
    }

    unsafe fn create_buffer_internal(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
        transient: bool,
        external_memory_handle: Option<ExternalMemoryHandle>,
    ) -> BufferInfo {
        assert!(transient ^ external_memory_handle.is_some());

        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: buffer_create_info.byte_size,
            usage: buffer_create_info.usage,
            sharing_mode: if self.device.queues_info.queue_count == 1 {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };
        let mem_req = unsafe { self.device.device.get_buffer_memory_requirements(handle) };
        let (memory, mapped_ptr) = if let Some(ref external_memory_handle) = external_memory_handle
        {
            // import external memory and bind it to the resource
            let device_memory = platform_impl::allocate_external_memory(
                &self.device,
                &mem_req,
                memory_info.required_flags,
                memory_info.preferred_flags,
                external_memory_handle,
            );
            unsafe {
                self.device
                    .device
                    .bind_buffer_memory(handle, device_memory, 0)
                    .unwrap();
            }
            let memory = ResourceMemory::Imported { device_memory };
            (
                /* memory */ memory,
                /* mapped_ptr */ ptr::null_mut(),
            )
        } else if transient && !buffer_create_info.map_on_create {
            // We can delay allocation only if the user requests a transient resource and
            // if the resource does not need to be mapped immediately.
            let memory = ResourceMemory::Aliasable(AllocationRequirements {
                mem_req,
                required_flags: memory_info.required_flags,
                preferred_flags: memory_info.preferred_flags,
            });
            (
                /* memory */ memory,
                /* mapped_ptr */ ptr::null_mut(),
            )
        } else {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                flags: if buffer_create_info.map_on_create {
                    vk_mem::AllocationCreateFlags::MAPPED
                } else {
                    vk_mem::AllocationCreateFlags::NONE
                },
                preferred_flags: memory_info.preferred_flags,
                required_flags: memory_info.required_flags,
                ..Default::default()
            };
            let (alloc, alloc_info) = self
                .device
                .allocator
                .allocate_memory(&mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            unsafe {
                self.device
                    .device
                    .bind_buffer_memory(
                        handle,
                        alloc_info.get_device_memory(),
                        alloc_info.get_offset() as u64,
                    )
                    .unwrap();
            }
            let memory = ResourceMemory::Exclusive(alloc);
            let mapped_ptr = if buffer_create_info.map_on_create {
                let ptr = alloc_info.get_mapped_data();
                assert!(!ptr.is_null(), "failed to map buffer");
                ptr
            } else {
                ptr::null_mut()
            };
            (/* memory */ memory, /* memory */ mapped_ptr)
        };
        let id = self.resources.insert(Resource {
            name: name.to_string(),
            user_ref_count: if transient { 0 } else { 1 },
            memory,
            tracking: Default::default(),
            should_delete: true,
            kind: ResourceKind::Buffer(BufferResource { handle }),
        });
        set_debug_object_name(
            &self.device,
            vk::ObjectType::BUFFER,
            handle.as_raw(),
            name,
            transient.then(|| self.next_serial),
        );

        BufferInfo {
            id: BufferId(id),
            handle,
            mapped_ptr,
        }
    }

    /// Creates a buffer resource.
    pub fn create_buffer(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
        transient: bool,
    ) -> BufferInfo {
        unsafe {
            self.create_buffer_internal(name, memory_info, buffer_create_info, transient, None)
        }
    }

    pub unsafe fn create_buffer_with_external_memory_handle(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
        external_memory_handle: ExternalMemoryHandle,
    ) -> BufferInfo {
        self.create_buffer_internal(
            name,
            memory_info,
            buffer_create_info,
            false,
            Some(external_memory_handle),
        )
    }

    pub(crate) fn allocate_memory_for_transients(
        &mut self,
        base_serial: u64,
        temporaries: &[ResourceId],
        passes: &[Pass],
    ) -> Vec<vk_mem::Allocation> {
        let _span = trace_span!("allocate_memory_for_transients").entered();

        #[derive(Copy, Clone, Debug)]
        struct AllocIndex {
            index: usize,
            dead_and_recycled: bool,
        }

        let reachability = compute_reachability(passes);
        // alloc index -> alloc requirements
        let mut requirements: Vec<AllocationRequirements> = Vec::new();
        // resource id -> allocation mapping (index+state)
        let mut alloc_map: SecondaryMap<ResourceId, AllocIndex> = SecondaryMap::new();

        for pass in passes {
            // --- assign memory for all resources accessed in this task
            for access in pass.accesses.iter() {
                let resource_id = access.id;
                let resource = self.resources.get(resource_id).unwrap();

                // already allocated
                if alloc_map.get(resource_id).is_some() {
                    continue;
                }

                let alloc_req = match &resource.memory {
                    ResourceMemory::Aliasable(req) => *req,
                    // not a transient, nothing to allocate
                    _ => continue,
                };

                let mut aliased = false;

                'alias: for &alias_candidate_id in temporaries.iter() {
                    if alias_candidate_id == resource_id {
                        continue;
                    }

                    let alias_candidate = self.resources.get(alias_candidate_id).unwrap();

                    // skip if not aliasable
                    let _alias_alloc_req = match &alias_candidate.memory {
                        ResourceMemory::Aliasable(req) => req,
                        _ => continue,
                    };

                    // skip if the resource has user handles pointing to it that may live beyond the current frame
                    if alias_candidate.user_ref_count > 0 {
                        continue;
                    }

                    let mut alloc_state =
                        if let Some(alloc_state) = alloc_map.get_mut(alias_candidate_id) {
                            // skip if the resource is already dead, and its memory was already reused
                            if alloc_state.dead_and_recycled {
                                continue;
                            }
                            alloc_state
                        } else {
                            // skip if not allocated yet
                            continue;
                        };

                    // if we want to use the resource, the resource must be dead (no more uses in subsequent passes),
                    // and there must be an execution dependency chain between the current task and all passes that last accessed the resource

                    for &read_serial in alias_candidate.tracking.readers.iter() {
                        // Consider the resource to be live if:
                        // 1. the reader is in a previous frame (we don't have info about execution dependencies between passes in different frames)
                        //      TODO is that really the case?
                        // 2. the reader comes after this pass
                        // 3. there's no execution dependency chain from the reader to the current task: this is encoded in the reachability matrix.
                        //      TODO verify that the DAG (and thus the reachability matrix) is properly constructed

                        if read_serial != 0
                            && (read_serial <= base_serial
                                || read_serial >= pass.snn.serial()
                                || !reachability.is_reachable(
                                    (read_serial - base_serial - 1) as usize,
                                    pass.frame_index,
                                ))
                        {
                            continue 'alias;
                        }
                    }

                    let write_serial = alias_candidate.tracking.writer.serial();
                    if write_serial != 0
                        && (write_serial <= base_serial
                            || write_serial >= pass.snn.serial()
                            || !reachability.is_reachable(
                                (write_serial - base_serial - 1) as usize,
                                pass.frame_index,
                            ))
                    {
                        continue;
                    }

                    // the resource is dead, try to reuse
                    let dead_alloc = &mut requirements[alloc_state.index];

                    if !dead_alloc.try_adjust(&alloc_req) {
                        continue;
                    }

                    // the two resources may alias; the requirements have been adjusted
                    // update the allocation map
                    let index = alloc_state.index;
                    alloc_state.dead_and_recycled = true;

                    alloc_map.insert(
                        resource_id,
                        AllocIndex {
                            index,
                            dead_and_recycled: false,
                        },
                    );

                    aliased = true;
                    break;
                }

                if !aliased {
                    // new allocation
                    let index = requirements.len();
                    requirements.push(alloc_req);
                    alloc_map.insert(
                        resource_id,
                        AllocIndex {
                            index,
                            dead_and_recycled: false,
                        },
                    );
                }
            }
        }

        // --- print some debug info
        /*println!("Memory blocks:");
        for (i, req) in requirements.iter().enumerate() {
            println!(" block #{}: {:?}", i, req);
        }
        println!("Memory block assignments:");
        for &tmp in temporaries {
            if let Some(alloc_state) = alloc_map.get(tmp) {
                println!(
                    "`{}` => {:?}",
                    self.resources.get(tmp).unwrap().name,
                    alloc_state
                );
            } else {
                println!("`{}` => N/A", self.resources.get(tmp).unwrap().name);
            }
        }*/

        // now allocate device memory
        let mut allocations = Vec::with_capacity(requirements.len());
        let mut allocation_infos = Vec::with_capacity(requirements.len());

        for alloc_req in requirements.iter() {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                ..Default::default()
            };
            let (allocation, allocation_info) = self
                .device
                .allocator
                .allocate_memory(&alloc_req.mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            allocations.push(allocation);
            allocation_infos.push(allocation_info);
        }

        // and assign them to the resources
        for &tmp in temporaries {
            if let Some(alloc_index) = alloc_map.get(tmp) {
                let resource = self.resources.get_mut(tmp).unwrap();
                let alloc_info = &allocation_infos[alloc_index.index];
                match &resource.kind {
                    ResourceKind::Image(img) => unsafe {
                        self.device
                            .device
                            .bind_image_memory(
                                img.handle,
                                alloc_info.get_device_memory(),
                                alloc_info.get_offset() as u64,
                            )
                            .unwrap();
                    },
                    ResourceKind::Buffer(buf) => unsafe {
                        self.device
                            .device
                            .bind_buffer_memory(
                                buf.handle,
                                alloc_info.get_device_memory(),
                                alloc_info.get_offset() as u64,
                            )
                            .unwrap();
                    },
                }
            }
        }

        allocations
    }
}
