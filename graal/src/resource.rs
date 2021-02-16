use crate::pass::{QueueSerialNumbers, SubmissionNumber};
use ash::vk;

/// Information about the memory to be allocated for a resource.
#[derive(Copy, Clone, Debug, Default)]
pub struct ResourceMemoryInfo {
    /// Required memory property flags. Panics if those cannot be honored (no memory type with those properties).
    pub required_flags: vk::MemoryPropertyFlags,
    /// Preferred memory property flags. The allocator will honor those flags if a memory type with those properties exist, otherwise it will fallback to the required flags.
    pub preferred_flags: vk::MemoryPropertyFlags,
}

impl ResourceMemoryInfo {
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
    /// Whether the resource should live only for the duration of the batch it's used in.
    /// When the batch that uses the resource completes, the resource is automatically deleted.
    /// The resource can only be used in one batch.
    pub transient: bool,
}

/// Parameters of a newly created buffer resource.
#[derive(Copy, Clone, Debug, Default)]
pub struct BufferResourceCreateInfo {
    /// Usage flags.
    pub usage: vk::BufferUsageFlags,
    /// Size of the buffer in bytes.
    pub byte_size: u64,
    /// Whether the resource should live only for the duration of the batch it's used in.
    /// When the batch that uses the resource completes, the resource is automatically deleted.
    /// The resource can only be used in one batch.
    pub transient: bool,
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
    pub(crate) layout: vk::ImageLayout,
    pub(crate) access_mask: vk::AccessFlags,
    pub(crate) input_stage: vk::PipelineStageFlags,
    pub(crate) output_stage: vk::PipelineStageFlags,
}

#[derive(Debug)]
pub(crate) enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

#[derive(Debug)]
pub(crate) struct ResourceTrackingInfo {
    pub(crate) owner_queue_family: u32,
    pub(crate) readers: QueueSerialNumbers,
    pub(crate) writer: SubmissionNumber,
    pub(crate) layout: vk::ImageLayout,
    pub(crate) availability_mask: vk::AccessFlags,
    pub(crate) visibility_mask: vk::AccessFlags,
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

    pub(crate) fn buffer(&self) -> &BufferResource {
        match &self.kind {
            ResourceKind::Buffer(r) => r,
            _ => panic!("expected a buffer resource"),
        }
    }
}
