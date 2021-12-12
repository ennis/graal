use crate::{
    context::{get_vk_sample_count, is_write_access, SemaphoreWait},
    serial::{QueueSerialNumbers, SubmissionNumber},
    vk,
    vk::Handle,
    Context, Device, FrameNumber,
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

slotmap::new_key_type! {
    /// Identifies a GPU resource (buffer or image).
    pub struct ResourceId;

    /// Identifies a resource group.
    pub struct ResourceGroupId;
}

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct BufferId(pub(crate) ResourceId);

impl BufferId {
    /// Returns the underlying ResourceId.
    pub fn resource_id(&self) -> ResourceId {
        self.0
    }
}

/// TODO docs
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ImageId(pub(crate) ResourceId);

impl ImageId {
    /// Returns the underlying ResourceId.
    pub fn resource_id(&self) -> ResourceId {
        self.0
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

#[derive(Debug)]
pub(crate) enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum AccessTracker {
    Host,
    Device(SubmissionNumber),
}

#[derive(Copy, Clone, Debug)]
pub(crate) struct ResourceTrackingInfo {
    /// SNN of the first pass accessing the resource.
    pub(crate) first_access: Option<AccessTracker>,
    /// Unused?
    pub(crate) owner_queue_family: u32,
    /// Current readers of the resource.
    pub(crate) readers: QueueSerialNumbers,
    /// Current writer of the resource.
    pub(crate) writer: Option<AccessTracker>,
    /// Current image layout if the resource is an image. Ignored otherwise.
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
    // TODO remove this?
    pub(crate) fn has_writer(&self) -> bool {
        self.writer.is_some()
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
            writer: None,
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
        // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
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
    /// For frozen resources, the group that the resource belongs to. Otherwise `None` if the resource
    /// is not frozen.
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

    pub(crate) fn is_frozen(&self) -> bool {
        self.group.is_some()
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
                    device.allocator.lock().unwrap().free(allocation).unwrap()
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

/// Information about the current state of a frozen image resource.
pub struct ImageResourceState {
    pub group_id: ResourceGroupId,
    // TODO include visibility information? or frozen resources should be visible to all stages?
    pub layout: vk::ImageLayout,
}

/// Information about the current state of a frozen image resource.
pub struct BufferResourceState {
    pub group_id: ResourceGroupId,
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

slotmap::new_key_type! {
    pub struct DescriptorSetLayoutId;
    pub struct SamplerId;
    pub struct PipelineId;
    pub struct PipelineLayoutId;
    pub struct ComputePipelineId;
}

struct Tracked<T> {
    frame: FrameNumber,
    obj: T,
}

/// Vulkan object tracker.
struct ObjectTracker<Id: slotmap::Key, Obj> {
    objects: SlotMap<Id, Tracked<Obj>>,
    pending_deletion: Vec<Tracked<Obj>>,
}

impl<Id: slotmap::Key, Obj> Default for ObjectTracker<Id, Obj> {
    fn default() -> Self {
        ObjectTracker {
            objects: SlotMap::with_key(),
            pending_deletion: vec![],
        }
    }
}

impl<Id: slotmap::Key, Obj> ObjectTracker<Id, Obj> {
    fn insert(&mut self, obj: Obj) -> Id {
        self.objects.insert(Tracked {
            obj,
            frame: Default::default(),
        })
    }

    fn destroy(&mut self, id: Id) {
        if let Some(id) = self.objects.remove(id) {
            self.pending_deletion.push(id)
        } else {
            // TODO more debug
            tracing::warn!("object already removed")
        }
    }

    fn destroy_on_frame_completed(&mut self, frame: FrameNumber, id: Id) {
        if let Some(t) = self.objects.remove(id) {
            self.pending_deletion.push(Tracked { frame, obj: t.obj })
        } else {
            // TODO more debug
            tracing::warn!("object already removed")
        }
    }

    fn cleanup(&mut self, completed_frame: FrameNumber, mut f: impl FnMut(Obj)) {
        let mut i = 0;
        while i <= self.pending_deletion.len() {
            if self.pending_deletion[i].frame <= completed_frame {
                f(self.pending_deletion.swap_remove(i).obj)
            } else {
                i += 1;
            }
        }
    }
}

/// Tracked device objects.
///
/// There are three strategies for destroying objects:
/// 1. defer destruction until the current frame has finished execution. This is needed for:
///     - descriptor sets
///     - framebuffers
///     - samplers
///     - pipelines
///     - image views
/// 2. defer destruction until the current frame has been submitted (and all command buffers have been recorded). Needed for:
///     - pipeline layouts
pub(crate) struct DeviceObjects {
    pub(crate) resources: ResourceMap,
    pub(crate) resource_groups: slotmap::SlotMap<ResourceGroupId, ResourceGroup>,
    descriptor_set_layouts: slotmap::SlotMap<DescriptorSetLayoutId, vk::DescriptorSetLayout>,
    samplers: ObjectTracker<SamplerId, vk::Sampler>,
    pipelines: ObjectTracker<PipelineId, vk::Pipeline>,
    descriptor_allocators: slotmap::SecondaryMap<DescriptorSetLayoutId, DescriptorSetAllocator>,
    /// Pipeline layouts pending deletion after the current frame is submitted.
    dead_pipeline_layouts: Vec<vk::PipelineLayout>,
}

/// Information about a newly created sampler object.
pub struct SamplerInfo {
    /// Vulkan handle of the sampler.
    pub handle: vk::Sampler,
    /// Tracking ID of the sampler object.
    pub id: SamplerId,
}

/// Information about a newly created descriptor set layout object.
pub struct DescriptorSetLayoutInfo {
    /// Vulkan handle of the descriptor set layout.
    pub handle: vk::DescriptorSetLayout,
    /// Tracking ID of the descriptor set layout.
    pub id: DescriptorSetLayoutId,
}
//-----------------------------------------------------------------------------------------
const DESCRIPTOR_POOL_PER_TYPE_COUNT: u32 = 1024;
const DESCRIPTOR_POOL_SET_COUNT: u32 = DESCRIPTOR_POOL_PER_TYPE_COUNT;

/// Allocator for descriptor sets of a specific layout.
#[derive(Debug)]
pub struct DescriptorSetAllocator {
    pub(crate) pool_size_count: u32,
    pub(crate) pool_sizes: [vk::DescriptorPoolSize; 16],
    pub(crate) full_pools: Vec<vk::DescriptorPool>,
    ///
    pub(crate) pool: Option<vk::DescriptorPool>,
    /// Descriptor sets not currently in use.
    pub(crate) free: Vec<vk::DescriptorSet>,
}

impl DescriptorSetAllocator {
    pub fn new(
        descriptor_set_layout_bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetAllocator {
        let mut pool_sizes: [vk::DescriptorPoolSize; 16] = Default::default();
        // count the number of each type of descriptor
        let mut sampler_desc_count = 0;
        let mut combined_image_sampler_desc_count = 0;
        let mut sampled_image_desc_count = 0;
        let mut storage_image_desc_count = 0;
        let mut uniform_texel_buffer_desc_count = 0;
        let mut storage_texel_buffer_desc_count = 0;
        let mut uniform_buffer_desc_count = 0;
        let mut storage_buffer_desc_count = 0;
        let mut uniform_buffer_dynamic_desc_count = 0;
        let mut storage_buffer_dynamic_desc_count = 0;
        let mut input_attachment_desc_count = 0;
        let mut acceleration_structure_desc_count = 0;

        for b in descriptor_set_layout_bindings.iter() {
            match b.descriptor_type {
                vk::DescriptorType::SAMPLER => sampler_desc_count += 1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    combined_image_sampler_desc_count += 1
                }
                vk::DescriptorType::SAMPLED_IMAGE => sampled_image_desc_count += 1,
                vk::DescriptorType::STORAGE_IMAGE => storage_image_desc_count += 1,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER => uniform_texel_buffer_desc_count += 1,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER => storage_texel_buffer_desc_count += 1,
                vk::DescriptorType::UNIFORM_BUFFER => uniform_buffer_desc_count += 1,
                vk::DescriptorType::STORAGE_BUFFER => storage_buffer_desc_count += 1,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    uniform_buffer_dynamic_desc_count += 1
                }
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    storage_buffer_dynamic_desc_count += 1
                }
                vk::DescriptorType::INPUT_ATTACHMENT => input_attachment_desc_count += 1,
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                    acceleration_structure_desc_count += 1
                }
                _ => {}
            }
        }

        let mut pool_size_count = 0;
        if sampler_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::SAMPLER;
            pool_sizes[pool_size_count].descriptor_count =
                sampler_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if combined_image_sampler_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            pool_sizes[pool_size_count].descriptor_count =
                combined_image_sampler_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if sampled_image_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::SAMPLED_IMAGE;
            pool_sizes[pool_size_count].descriptor_count =
                sampled_image_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_image_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_IMAGE;
            pool_sizes[pool_size_count].descriptor_count =
                storage_image_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_texel_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_texel_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_texel_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                storage_texel_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                storage_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_buffer_dynamic_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_buffer_dynamic_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_buffer_dynamic_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
            pool_sizes[pool_size_count].descriptor_count =
                storage_buffer_dynamic_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if input_attachment_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::INPUT_ATTACHMENT;
            pool_sizes[pool_size_count].descriptor_count =
                input_attachment_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if acceleration_structure_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::ACCELERATION_STRUCTURE_KHR;
            pool_sizes[pool_size_count].descriptor_count =
                acceleration_structure_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }

        DescriptorSetAllocator {
            pool_sizes,
            pool_size_count: pool_size_count as u32,
            full_pools: vec![],
            pool: None,
            free: vec![],
        }
    }
}

impl DeviceObjects {
    pub(crate) fn new() -> DeviceObjects {
        DeviceObjects {
            resources: SlotMap::with_key(),
            resource_groups: SlotMap::with_key(),
            descriptor_set_layouts: Default::default(),
            samplers: Default::default(),
            pipelines: Default::default(),
            descriptor_allocators: slotmap::SecondaryMap::default(),
            dead_pipeline_layouts: vec![],
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
                || match r.tracking.writer {
                    None => false,
                    Some(AccessTracker::Device(writer)) => {
                        writer.serial() > completed_serials.serial(writer.queue())
                    }
                    Some(AccessTracker::Host) => {
                        // nothing to wait for
                        false
                    }
                };

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
    /// TODO docs
    pub(crate) fn cleanup_resources(&self, completed_serials: QueueSerialNumbers) {
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.cleanup_resources(&self, completed_serials)
    }

    /// Common helper function to register a buffer or image resource.
    unsafe fn register_resource(
        &self,
        info: ResourceRegistrationInfo,
        kind: ResourceKind,
    ) -> ResourceId {
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.register_resource(&self, info, kind)
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

    /// Creates a sampler object.
    pub fn create_sampler(&self, create_info: &vk::SamplerCreateInfo) -> SamplerInfo {
        let mut objects = self.objects.lock().expect("failed to lock resources");
        unsafe {
            let handle = self
                .device
                .create_sampler(create_info, None)
                .expect("failed to create sampler");
            let id = objects.samplers.insert(handle);
            SamplerInfo { handle, id }
        }
    }

    /// Schedules destruction of the specified sampler.
    pub fn destroy_sampler(&self, id: SamplerId) {
        let mut objects = self.objects.lock().unwrap();
        objects
            .samplers
            .destroy_on_frame_completed(self.context_state.last_started_frame.get(), id);
    }

    /// Creates a descriptor set layout object.
    pub fn create_descriptor_set_layout(
        &self,
        bindings: &[vk::DescriptorSetLayoutBinding],
    ) -> DescriptorSetLayoutInfo {
        // --- create layout ---
        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        let layout = unsafe {
            self.device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("failed to create descriptor set layout")
        };

        // also create an allocator for it
        let allocator = DescriptorSetAllocator::new(bindings);

        let mut objects = self.objects.lock().unwrap();
        let id = objects.descriptor_set_layouts.insert(layout);
        objects.descriptor_allocators.insert(id, allocator);
        DescriptorSetLayoutInfo { handle: layout, id }
    }

    /// Destroys a descriptor set layout object.
    pub fn destroy_descriptor_set_layout(&self, id: DescriptorSetLayoutId) {
        // nothing "in-flight" really needs to keep the descriptor set layout alive, so just destroy it right now.
        let mut objects = self.objects.lock().expect("failed to lock resources");
        if let Some(layout) = objects.descriptor_set_layouts.remove(id) {
            // TODO Safety
            unsafe {
                self.device.destroy_descriptor_set_layout(layout, None);
            }
        } else {
            tracing::warn!("unknown object id {:?}", id);
        }

        // also destroy the associated descriptor set allocator if there is one
        if let Some(allocator) = objects.descriptor_allocators.remove(id) {
            unsafe {
                for pool in allocator.full_pools {
                    self.device.destroy_descriptor_pool(pool, None)
                }
                if let Some(pool) = allocator.pool {
                    self.device.destroy_descriptor_pool(pool, None)
                }
                // no need to destroy the individual descriptor sets
            }
        }
    }

    /// Creates a pipeline layout object.
    pub fn create_pipeline_layout(
        &self,
        create_info: &vk::PipelineLayoutCreateInfo,
    ) -> vk::PipelineLayout {
        unsafe {
            self.device
                .create_pipeline_layout(create_info, None)
                .expect("failed to create pipeline layout")
        }
    }

    pub fn destroy_pipeline_layout(&self, layout: vk::PipelineLayout) {
        // if not recording a frame, then we can destroy it immediately
        // otherwise,
        // destroy when
    }

    /// Allocates a descriptor set.
    pub fn allocate_descriptor_set(&self, layout: DescriptorSetLayoutId) -> vk::DescriptorSet {
        let mut objects = self.objects.lock().unwrap();
        let layout_handle = *objects.descriptor_set_layouts.get(layout).unwrap();
        let allocator = objects.descriptor_allocators.get_mut(layout).unwrap();

        let handle = loop {
            // get or create descriptor pool
            let descriptor_pool = {
                if let Some(pool) = allocator.pool {
                    pool
                } else {
                    let pool = unsafe {
                        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
                            flags: vk::DescriptorPoolCreateFlags::default(),
                            max_sets: DESCRIPTOR_POOL_SET_COUNT,
                            pool_size_count: allocator.pool_size_count,
                            p_pool_sizes: allocator.pool_sizes.as_ptr(),
                            ..Default::default()
                        };
                        self.device
                            .create_descriptor_pool(&descriptor_pool_create_info, None)
                            .unwrap()
                    };
                    allocator.pool = Some(pool);
                    pool
                }
            };

            let result = unsafe {
                let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
                    descriptor_pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &layout_handle,
                    ..Default::default()
                };
                self.device
                    .allocate_descriptor_sets(&descriptor_set_allocate_info)
            };

            match result {
                Ok(d) => break *d.first().unwrap(),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                    // pool is full, retire the current one and loop
                    // it will allocate a new one on the next iteration
                    if let Some(pool) = mem::replace(&mut allocator.pool, None) {
                        allocator.full_pools.push(pool);
                    }
                    continue;
                }
                Err(e) => panic!("error allocating descriptor sets: {}", e),
            }
        };

        handle
    }

    /// Frees the specified descriptor set immediately.
    ///
    /// This assumes that the descriptor set is not in use anymore.
    pub unsafe fn free_descriptor_set(
        &mut self,
        layout: DescriptorSetLayoutId,
        ds: vk::DescriptorSet,
    ) {
        let mut objects = self.objects.lock().unwrap();
        let allocator = objects.descriptor_allocators.get_mut(layout).unwrap();
        allocator.free.push(ds);
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
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.resources.get_mut(id.0).unwrap().discarded = true;
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
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.resources.get_mut(id.0).unwrap().discarded = true;
    }

    /// Creates a resource group.
    pub fn create_resource_group(
        &self,
        dst_stage_mask: vk::PipelineStageFlags,
        dst_access_mask: vk::AccessFlags,
    ) -> ResourceGroupId {
        // resource groups are for read-only resources
        assert!(!is_write_access(dst_access_mask));
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.resource_groups.insert(ResourceGroup {
            wait_serials: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }

    /// Destroys a resource group.
    pub fn destroy_resource_group(&self, group_id: ResourceGroupId) {
        let mut objects = self.objects.lock().expect("failed to lock resources");
        objects.resource_groups.remove(group_id);
    }

    /// Returns information about the current state of a frozen image resource.
    pub fn get_image_state(&self, image_id: ImageId) -> Option<ImageResourceState> {
        let objects = self.objects.lock().expect("failed to lock resources");
        let image = objects.resources.get(image_id.0).expect("invalid resource");
        if let Some(group_id) = image.group {
            Some(ImageResourceState {
                group_id,
                layout: image.tracking.layout,
            })
        } else {
            None
        }
    }

    /// Returns information about the current state of a frozen buffer resource.
    pub fn get_buffer_state(&self, buffer_id: BufferId) -> Option<BufferResourceState> {
        let objects = self.objects.lock().expect("failed to lock resources");
        let buffer = objects
            .resources
            .get(buffer_id.0)
            .expect("invalid resource");
        if let Some(group_id) = buffer.group {
            Some(BufferResourceState { group_id })
        } else {
            None
        }
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

        // allocate immediately
        // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
        let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_req,
            location,
            linear: true,
        };
        let allocation = self
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.device
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();
        }

        // register the resource in the context
        let id = unsafe {
            self.register_image_resource(ImageRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    ownership: ResourceOwnership::OwnedResource {
                        requirements: AllocationRequirements { mem_req, location },
                        allocation: Some(ResourceAllocation::Default { allocation }),
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

        // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
        let (ownership, mapped_ptr) = /*if !buffer_create_info.map_on_create {
            // We can delay allocation only if the user requests a transient resource and
            // if the resource does not need to be mapped immediately.
            let ownership = ResourceOwnership::OwnedResource {
                requirements: AllocationRequirements { mem_req, location },
                allocation: None,
            };
            (/* ownership */ ownership, /* mapped_ptr */ None)
        } else*/ {
            // caller requested a mapped pointer, must create and allocate immediately
            let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
                name,
                requirements: mem_req,
                location,
                linear: true,
            };
            let allocation = self
                .allocator
                .lock()
                .unwrap()
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
        let objects = self.objects.lock().expect("failed to lock resources");
        objects.resources.get(id.0).unwrap().image().handle
    }

    /// Returns the handle of the corresponding buffer resource.
    /// Panics if `id` does not refer to a buffer resource.
    pub fn buffer_handle(&self, id: BufferId) -> vk::Buffer {
        let resources = self.objects.lock().expect("failed to lock resources");
        resources.resources.get(id.0).unwrap().buffer().handle
    }
}
