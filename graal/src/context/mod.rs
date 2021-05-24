use std::{collections::VecDeque, ffi::CString, fmt, os::raw::c_void, mem, ptr};

use ash::{version::DeviceV1_0, vk};
use slotmap::{Key, SlotMap};

use crate::{
    context::{
        submission::CommandAllocator,
    },
    device::Device,
    MAX_QUEUES,
};
use std::cmp::Ordering;
use tracing::{trace, trace_span};

pub(crate) mod frame;
pub(crate) mod pass;
pub(crate) mod submission;
pub(crate) mod transient;

pub use frame::{Frame, PassBuilder};
use std::ops::{Deref, DerefMut};
pub use submission::CommandContext;
use crate::context::pass::SemaphoreWait;
use crate::ash::vk::Handle;

/// Maximum time to wait for batches to finish in `SubmissionState::wait`.
pub(crate) const SEMAPHORE_WAIT_TIMEOUT_NS: u64 = 1_000_000_000;

/// A number that uniquely identifies a frame.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct FrameSerialNumber(pub u64);

/// A number that combines the serial number of a pass and the queue it was submitted on.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct SubmissionNumber(u64);

impl SubmissionNumber {
    /// Creates a new submission number from a queue index and a serial.
    pub fn new(queue_index: usize, serial: u64) -> SubmissionNumber {
        assert!(queue_index < 4);
        assert!(serial < 1u64 << 62);
        SubmissionNumber(((queue_index as u64) << 62) | serial)
    }

    /// The queue that the pass is submitted on.
    pub const fn queue(&self) -> usize {
        (self.0 >> 62) as usize
    }

    /// The serial number of the pass.
    pub const fn serial(&self) -> u64 {
        self.0 & ((1 << 62) - 1)
    }

    /// Whether this submission number is valid.
    pub const fn is_valid(&self) -> bool {
        self.serial() != 0
    }
}

impl fmt::Debug for SubmissionNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.queue(), self.serial())
    }
}

/// A set of serial numbers, one for each queue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct QueueSerialNumbers(pub [u64; MAX_QUEUES]);

impl QueueSerialNumbers {
    //
    pub const fn new() -> QueueSerialNumbers {
        QueueSerialNumbers([0; MAX_QUEUES])
    }

    // TODO better name?
    /*pub const fn has_nonzero_serial(&self) -> bool {
        let mut i = 0;
        while i < MAX_QUEUES {
            if self.0[i] != 0 {
                return true;
            }
            i += 1;
        }
        false
    }*/

    pub fn from_submission_number(snn: SubmissionNumber) -> QueueSerialNumbers {
        Self::from_queue_serial(snn.queue(), snn.serial())
    }

    pub fn from_queue_serial(queue: usize, serial: u64) -> QueueSerialNumbers {
        let mut s = Self::new();
        s[queue] = serial;
        s
    }

    pub fn serial(&self, queue: usize) -> u64 {
        self.0[queue]
    }

    pub fn join(&self, other: QueueSerialNumbers) -> QueueSerialNumbers {
        let mut r = *self;
        r.join_assign(other);
        r
    }

    pub fn join_assign(&mut self, other: QueueSerialNumbers) {
        for i in 0..MAX_QUEUES {
            self[i] = self[i].max(other[i]);
        }
    }

    pub fn join_serial(&self, snn: SubmissionNumber) -> QueueSerialNumbers {
        let mut r = *self;
        r[snn.queue()] = r[snn.queue()].max(snn.serial());
        r
    }

    pub fn iter(&self) -> impl Iterator<Item = &'_ u64> {
        self.0.iter()
    }

    //pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut u64> {
    //    self.0.iter_mut()
    //}
}

impl Deref for QueueSerialNumbers {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for QueueSerialNumbers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PartialOrd for QueueSerialNumbers {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let before = self.0.iter().zip(other.0.iter()).all(|(&a, &b)| a <= b);

        let after = self.0.iter().zip(other.0.iter()).all(|(&a, &b)| a >= b);

        match (before, after) {
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (true, true) => Some(Ordering::Equal),
            (false, false) => None,
        }
    }
}

/// TODO document
fn local_pass_index(serial: u64, frame_base_serial: u64) -> usize {
    assert!(serial > frame_base_serial);
    (serial - frame_base_serial - 1) as usize
}

pub(crate) fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => panic!("unsupported number of samples"),
    }
}

/// Helper function to associate a debug name to a vulkan handle.
/// If `transient`, the next serial is appended to the name.
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

/*pub unsafe fn place_aligned(layout: &Layout, ptr: &mut *mut u8, space: &mut usize) -> *mut u8 {
    let ptr_usize = *ptr as usize;

    let mut off = ptr_usize & (layout.align() - 1);
    if off > 0 {
        off = layout.align() - off;
    }
    if ptr_usize + off + layout.size() > *space {
        ptr::null_mut()
    } else {
        *space -= off + layout.size();
        *ptr = ptr.add(off);
        *ptr
    }
}

struct UploadChunk {
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    base: *mut u8,
    ptr: *mut u8,
    space: usize,
}

impl UploadChunk {
    pub fn new(
        device: &Device,
        memory_usage: vk_mem::MemoryUsage,
        buffer_usage: vk::BufferUsageFlags,
        size: usize,
    ) -> UploadChunk {
        let alloc_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            usage: memory_usage,
            ..Default::default()
        };

        let buffer_create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: size as u64,
            usage: buffer_usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: device.queues_info.queue_count as u32,
            p_queue_family_indices: device.queues_info.families.as_ptr(),
            ..Default::default()
        };

        let (buffer, allocation, alloc_info) = device
            .allocator
            .create_buffer(&buffer_create_info, &alloc_info)
            .expect("failed to allocate buffer");
        UploadChunk {
            allocation,
            buffer,
            base: alloc_info.get_mapped_data(),
            ptr: alloc_info.get_mapped_data(),
            space: size,
        }
    }

    pub unsafe fn allocate(
        &mut self,
        layout: &Layout,
    ) -> Option<(*mut u8, vk::Buffer, vk::DeviceSize)> {
        let ptr = place_aligned(layout, &mut self.ptr, &mut self.space);
        if !ptr.is_null() {
            Some((ptr, self.buffer, ptr.offset_from(self.base) as u64))
        } else {
            None
        }
    }
}

/// A pool of CPU-visible memory used for staging resources or uploading dynamic data.
struct UploadPool {
    usage: vk_mem::MemoryUsage,
    chunk_size: usize,
    chunks: Vec<UploadChunk>,
    dedicated: Vec<vk_mem::Allocation>,
}

fn is_read_access(mask: vk::AccessFlags) -> bool {
    mask.intersects(
        vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::INDEX_READ
            | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
            | vk::AccessFlags::UNIFORM_READ
            | vk::AccessFlags::INPUT_ATTACHMENT_READ
            | vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::HOST_READ
            | vk::AccessFlags::MEMORY_READ
            | vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
            | vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
            | vk::AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
            | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            | vk::AccessFlags::SHADING_RATE_IMAGE_READ_NV
            | vk::AccessFlags::FRAGMENT_DENSITY_MAP_READ_EXT
            | vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
    )
}*/

fn is_write_access(mask: vk::AccessFlags) -> bool {
    mask.intersects(
        vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags::TRANSFER_WRITE
            | vk::AccessFlags::HOST_WRITE
            | vk::AccessFlags::MEMORY_WRITE
            | vk::AccessFlags::TRANSFORM_FEEDBACK_WRITE_EXT
            | vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            | vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV,
    )
}

fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM_S8_UINT => true,
        vk::Format::D24_UNORM_S8_UINT => true,
        vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

fn is_depth_only_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM => true,
        vk::Format::X8_D24_UNORM_PACK32 => true,
        vk::Format::D32_SFLOAT => true,
        _ => false,
    }
}

fn is_stencil_only_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::S8_UINT => true,
        _ => false,
    }
}

pub fn format_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
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
pub struct AllocationRequirements {
    pub(crate) mem_req: vk::MemoryRequirements,
    pub(crate) required_flags: vk::MemoryPropertyFlags,
    pub(crate) preferred_flags: vk::MemoryPropertyFlags,
}

impl AllocationRequirements {
    fn adjusted_requirements(&self, other: &AllocationRequirements) -> Option<AllocationRequirements> {
        if self.required_flags != other.required_flags {
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
    first_access: SubmissionNumber,
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

/*/// Describes the kind of memory that is bound to a resource.
#[derive(Debug)]
pub(crate) enum ResourceMemory {
    /// The resource may share a block of memory allocation with other resources.
    Aliasable(AllocationRequirements),
    /// The resource has a block of memory allocated exclusively to it.
    Exclusive(vk_mem::Allocation),
    /// The memory for the resource is managed externally (e.g. swapchain images)
    Swapchain,
    /// The memory for this resource was imported or exported from/to an external handle.
    External { device_memory: vk::DeviceMemory },
}*/

/// Describes how a resource got its memory.
#[derive(Clone,Debug)]
pub enum ResourceAllocation {

    /// a block of memory exclusively for this resource.
    Default {
        allocation: vk_mem::Allocation
    },

    /// Memory aliasing: allocate a block of memory for the resource, which can possibly be shared
    /// with other aliasable resources if their lifetimes do not overlap.
    Transient {
        device_memory: vk::DeviceMemory,
        offset: vk::DeviceSize
    },

    /// The memory for this resource was imported or exported from/to an external handle.
    External { device_memory: vk::DeviceMemory },

}

#[derive(Clone,Debug)]
pub enum ResourceOwnership {
    Owned {
        requirements: AllocationRequirements,
        allocation: Option<ResourceAllocation>,
    },
    Referenced,
}

#[derive(Debug)]
pub(crate) struct Resource {
    /// Name, for debugging purposes
    pub(crate) name: String,

    /// Whether this pass has been discarded during the last frame.
    pub(crate) discarded: bool,

    /// The memory bound to the resource.
    pub(crate) ownership: ResourceOwnership,

    /// Details specific to the kind of resource (buffer or image).
    pub(crate) kind: ResourceKind,

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
    fn set_allocation(&mut self, alloc: ResourceAllocation) {
        // set the allocation type on the resource object
        match self.ownership {
            ResourceOwnership::Owned {
                ref mut allocation, ..
            } => {
                assert!(allocation.is_none());
                *allocation = Some(alloc)
            },
            _ => panic!("trying to set an allocation on an unowned object")
        }
    }
}

/// Destroys a resource and frees its device memory if it was allocated for this resource
/// exclusively.
unsafe fn destroy_resource(device: &Device, resource: &mut Resource) {
    // deallocate its memory, if it was allocated for this object exclusively
    match resource.ownership {
        ResourceOwnership::Owned { ref allocation, .. } => {
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

            match allocation {
                Some(ResourceAllocation::Default {
                         allocation
                     }) => {
                    device.allocator.free_memory(&allocation).unwrap()
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

impl<T: Copy> TypedBufferInfo<T> {
    pub unsafe fn byte_cast<U>(&self) -> TypedBufferInfo<U> where T: Sized, U: Sized {
        // TODO static assert?
        assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
        TypedBufferInfo {
            id: self.id,
            handle: self.handle,
            mapped_ptr: self.mapped_ptr as *mut U
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

// ---------------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct ResourceRegistrationInfo<'a> {
    pub name: &'a str,
    pub ownership: ResourceOwnership,
    pub initial_wait: Option<SemaphoreWait>
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
            let keep = !r.discarded
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



    unsafe fn register_resource(&mut self,
                                info: ResourceRegistrationInfo,
                                kind: ResourceKind) -> ResourceId
    {
        let (object_type, object_handle) = match kind {
            ResourceKind::Buffer(ref buf) => (vk::ObjectType::BUFFER, buf.handle.as_raw()),
            ResourceKind::Image(ref img) => (vk::ObjectType::IMAGE, img.handle.as_raw()),
        };

        let id = self.resources.insert(Resource {
            name: info.name.to_string(),
            discarded: false,
            tracking: Default::default(),
            kind,
            ownership: info.ownership
        });

        set_debug_object_name(
            &self.device,
            object_type,
            object_handle,
            info.name,
            Some(self.next_serial)
        );

        id
    }

    /// Registers an existing buffer resource in the context.
    pub unsafe fn register_buffer_resource(
        &mut self,
        info: BufferRegistrationInfo,
    ) -> BufferId {
        let id = self.register_resource( info.resource, ResourceKind::Buffer(BufferResource { handle: info.handle }));
        BufferId(id)
    }

    /// Registers an existing image resource in the context.
    pub unsafe fn register_image_resource(
        &mut self,
        info: ImageRegistrationInfo,
    ) -> ImageId {
        let id = self.register_resource(info.resource, ResourceKind::Image(ImageResource { handle: info.handle, format: info.format }));
        ImageId(id)
    }

    /// Destroys the image once it's not in use.
    pub fn destroy_image(&mut self, id: ImageId) {
        self.resources.get_mut(id.0).unwrap().discarded = true;
    }

    /// Destroys the buffer once it's not in use.
    pub fn destroy_buffer(&mut self, id: BufferId) {
        self.resources.get_mut(id.0).unwrap().discarded = true;
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
        // register the resource in the context
        let id = unsafe {
            self.register_image_resource(ImageRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    ownership: ResourceOwnership::Owned {
                        requirements: AllocationRequirements {
                            mem_req,
                            required_flags: memory_info.required_flags,
                            preferred_flags: memory_info.preferred_flags
                        },
                        allocation: None
                    },
                    initial_wait: None
                },
                handle,
                format: image_info.format
            })
        };

        ImageInfo { id, handle }
    }

    /// Creates a new buffer resource.
    pub fn create_buffer(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {

        // create the buffer object first
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

        // get its memory requirements
        let mem_req = unsafe { self.device.device.get_buffer_memory_requirements(handle) };

        let (ownership, mapped_ptr) = if !buffer_create_info.map_on_create {
            // We can delay allocation only if the user requests a transient resource and
            // if the resource does not need to be mapped immediately.
            let ownership = ResourceOwnership::Owned {
                requirements: AllocationRequirements {
                    mem_req,
                    required_flags: memory_info.required_flags,
                    preferred_flags: memory_info.preferred_flags
                },
                allocation: None
            };
            (
                /* ownership */ ownership,
                /* mapped_ptr */ ptr::null_mut(),
            )
        } else {
            // caller requested a mapped pointer, must create and allocate immediately
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                flags: vk_mem::AllocationCreateFlags::MAPPED,
                preferred_flags: memory_info.preferred_flags,
                required_flags: memory_info.required_flags,
                ..Default::default()
            };
            let (allocation, alloc_info) = self
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
            let ownership = ResourceOwnership::Owned {
                requirements: AllocationRequirements {
                    mem_req,
                    required_flags: memory_info.required_flags,
                    preferred_flags: memory_info.preferred_flags
                },
                allocation: Some(ResourceAllocation::Default {
                    allocation
                })
            };
            let mapped_ptr = if buffer_create_info.map_on_create {
                let ptr = alloc_info.get_mapped_data();
                assert!(!ptr.is_null(), "failed to map buffer");
                ptr
            } else {
                ptr::null_mut()
            };
            (/* ownership */ ownership, /* mapped_ptr */ mapped_ptr)
        };

        let id = unsafe { self.register_buffer_resource(BufferRegistrationInfo {
            resource: ResourceRegistrationInfo {
                name,
                initial_wait: None,
                ownership
            },
            handle})
        };

        BufferInfo {
            id,
            handle,
            mapped_ptr,
        }
    }
}


/// Stores the set of resources owned by a currently executing frame.
#[derive(Debug)]
struct FrameInFlight {
    signalled_serials: QueueSerialNumbers,
    transient_allocations: Vec<vk_mem::Allocation>,
    command_pools: Vec<CommandAllocator>,
    semaphores: Vec<vk::Semaphore>,
}

/// Represents a GPU operation that may have not finished yet.
#[derive(Copy, Clone, Debug)]
pub struct GpuFuture {
    pub(crate) serials: QueueSerialNumbers,
}

impl Default for GpuFuture {
    fn default() -> Self {
        GpuFuture::new()
    }
}

impl GpuFuture {
    /// Returns an "empty" GPU future that represents an already completed operation.
    /// Waiting on this future always returns immediately.
    pub const fn new() -> GpuFuture {
        GpuFuture {
            serials: QueueSerialNumbers::new(),
        }
    }

    /// Returns a future representing the moment when the operations represented
    /// by both `self` and `other` have completed.
    pub fn join(&self, other: GpuFuture) -> GpuFuture {
        GpuFuture {
            serials: self.serials.join(other.serials),
        }
    }
}

/// Graphics context
pub struct Context {
    pub(crate) device: Device,

    /// Timeline semaphores for each queue, used for cross-queue and inter-frame synchronization
    timelines: [vk::Semaphore; MAX_QUEUES],

    /// Array containing the last submitted pass serials for each queue
    last_signalled_serials: QueueSerialNumbers,

    /// Pool of recycled command pools.
    available_command_pools: Vec<CommandAllocator>,

    /// Free semaphores guaranteed to be in the unsignalled state.
    semaphore_pool: Vec<vk::Semaphore>,

    /// Resources (images and buffers), mapped by ID
    resources: ResourceMap,

    /// The serial to be used for the next pass (used by `Frame`)
    next_serial: u64,

    /// Array containing the last completed pass serials for each queue
    completed_serials: QueueSerialNumbers,

    /// Number of submitted frames
    submitted_frame_count: u64,

    /// Number of completed frames
    completed_frame_count: u64,

    /// Frames that are currently executing on the GPU.
    in_flight: VecDeque<FrameInFlight>,
}

impl Context {
    /// Creates a new context with a default device.
    pub fn new() -> Context {
        Self::with_device(Device::new(None))
    }

    /// Creates a new context. A vulkan device that can present to the specified surface will be created.
    pub fn with_surface(surface: vk::SurfaceKHR) -> Context {
        let device = Device::new(Some(surface));
        Self::with_device(device)
    }

    /// Creates a new context with the given device.
    pub fn with_device(device: Device) -> Context {
        let mut timelines: [vk::Semaphore; MAX_QUEUES] = Default::default();

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: &mut timeline_create_info as *mut _ as *mut c_void,
            ..Default::default()
        };

        for i in timelines.iter_mut() {
            *i = unsafe {
                device
                    .device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("failed to create semaphore")
            };
        }

        Context {
            device,
            timelines,
            last_signalled_serials: Default::default(),
            available_command_pools: vec![],
            completed_serials: Default::default(),
            next_serial: 0,
            submitted_frame_count: 0,
            completed_frame_count: 0,
            resources: SlotMap::with_key(),
            in_flight: VecDeque::new(),
            semaphore_pool: vec![],
        }
    }

    /// Returns the `graal::Device` owned by this context.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns the `ash::Device` owned by this context.
    /// Shorthand for `self.device().device`.
    pub fn vulkan_device(&self) -> &ash::Device {
        &self.device.device
    }

    /// Returns the handle of the corresponding image resource.
    /// Panics if `id` does not refer to an image resource.
    pub fn image_handle(&self, id: ImageId) -> vk::Image {
        self.resources.get(id.0).unwrap().image().handle
    }

    /// Returns the handle of the corresponding buffer resource.
    /// Panics if `id` does not refer to a buffer resource.
    pub fn buffer_handle(&self, id: BufferId) -> vk::Buffer {
        self.resources.get(id.0).unwrap().buffer().handle
    }

    /// Creates a binary semaphore (or return a previously used semaphore that is unsignalled).
    pub fn create_semaphore(&mut self) -> vk::Semaphore {
        if let Some(semaphore) = self.semaphore_pool.pop() {
            return semaphore;
        }

        unsafe {
            let create_info = vk::SemaphoreCreateInfo {
                ..Default::default()
            };
            self.device
                .device
                .create_semaphore(&create_info, None)
                .unwrap()
        }
    }

    /// Precondition: each semaphore in `semaphores` must be in the unsignalled state, or somehow
    /// be guaranteed to be in the unsignalled state the next time `create_semaphore` is called.
    fn recycle_semaphores(&mut self, mut semaphores: Vec<vk::Semaphore>) {
        self.semaphore_pool.append(&mut semaphores)
    }

    /// Returns whether the given frame, identified by its serial, has completed execution.
    pub fn is_frame_completed(&self, serial: FrameSerialNumber) -> bool {
        self.completed_frame_count >= serial.0
    }

    fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
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

    fn buffer_resource_by_handle(&self, handle: vk::Buffer) -> ResourceId {
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

    fn get_next_serial(&mut self) -> u64 {
        self.next_serial += 1;
        self.next_serial
    }

    /// Waits for all but the last submitted frame to finish and then recycles their resources.
    /// Calls `cleanup_resources` internally.
    fn wait_for_frames_in_flight(&mut self) {
        let _span = trace_span!("wait_for_frames_in_flight").entered();

        // pacing
        while self.in_flight.len() >= 2 {
            // two frames in flight already, must wait for the oldest one
            let f = self.in_flight.pop_front().unwrap();

            let _span = trace_span!("waiting for frame", serials = ?f.signalled_serials, frames_in_flight = self.in_flight.len()).entered();
            self.wait(&f.signalled_serials);

            // update completed serials
            // we just waited on those serials, so we know they are completed
            self.completed_serials = f.signalled_serials;

            // Recycle the command pools allocated for the frame. The allocated command buffers
            // can then be reused for future submissions.
            self.recycle_command_pools(f.command_pools);

            // Recycle the semaphores. They are guaranteed to be unsignalled since the frame must have
            // waited on them.
            self.recycle_semaphores(f.semaphores);

            // free transient allocations
            for alloc in f.transient_allocations.iter() {
                trace!(?alloc, "free_memory");
                self.device.allocator.free_memory(alloc).unwrap();
            }

            // bump completed frame count
            self.completed_frame_count += 1;
        }

        // given the new completed serials, free resources that have expired
        self.cleanup_resources();
    }

    fn dump_state(&self) {
        /*println!("Number of frames in flight: {}", self.in_flight.len());
        println!("Resources:");
        for (id, r) in self.resources.iter() {
            println!("- {:?}: {:?}", id, r);
        }
        println!("Available semaphores:");
        for s in self.available_semaphores.iter() {
            println!("- {:?}", s);
        }
        println!("Available command pools:");
        for cmd_pool in self.available_command_pools.iter() {
            println!(
                "- VkCommandPool {:?}: queue family #{}, {} used, {} free",
                cmd_pool.command_pool,
                cmd_pool.queue_family,
                cmd_pool.used.len(),
                cmd_pool.free.len()
            );
        }*/
        /*println!("VMA stats:");
        if let Ok(stats) = self.device.allocator.calculate_stats() {
            println!(
                "- number of allocations: {} in {} device memory blocks",
                stats.total.allocationCount, stats.total.blockCount
            );
            println!(
                "- memory usage: {} kB ({} kB used + {} kB unused)",
                (stats.total.usedBytes + stats.total.unusedBytes) / 1000,
                stats.total.usedBytes / 1000,
                stats.total.unusedBytes / 1000
            );
        }*/
    }

    pub fn current_frame_index(&self) -> u64 {
        self.submitted_frame_count
    }

    pub fn wait_for(&mut self, future: GpuFuture) {
        self.wait(&future.serials);
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // TODO
    }
}
