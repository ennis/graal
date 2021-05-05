use std::{
    collections::{HashMap, VecDeque},
    ffi::CString,
    fmt,
    os::raw::c_void,
};

use ash::{version::DeviceV1_0, vk};
use slotmap::{Key, SlotMap};

use crate::{
    context::{
        descriptor::{DescriptorSet, DescriptorSetAllocator},
        resource::{
            ImageResource, Resource, ResourceId, ResourceKind, ResourceMap, ResourceMemory,
            ResourceTrackingInfo,
        },
        submission::CommandAllocator,
    },
    device::Device,
    swapchain::Swapchain,
    DescriptorSetLayoutInfo, FragmentOutputInterface, FragmentOutputInterfaceExt, ImageInfo,
    MAX_QUEUES,
};
use std::cmp::Ordering;
use tracing::{trace, trace_span};

pub(crate) mod descriptor;
pub(crate) mod frame;
pub(crate) mod pass;
pub(crate) mod resource;
pub(crate) mod submission;
mod sync_table;
pub(crate) mod external_memory_handle;

use crate::context::resource::{BufferId, ImageId};
pub use descriptor::DescriptorSetAllocatorId;
pub use frame::{AccessType, AccessTypeInfo, Frame, PassBuilder};
use std::ops::{Deref, DerefMut};
pub use submission::CommandContext;

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

fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
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

/// Information about a swapchain.
#[derive(Copy, Clone, Debug)]
pub struct SwapchainInfo {
    /// ID of the swapchain.
    pub id: SwapchainId,
    /// Handle of the swapchain.
    pub handle: vk::SwapchainKHR,
    /// Format of the images in the swapchain.
    pub format: vk::Format,
}

/// Contains information about an image in a swapchain.
#[derive(Copy, Clone, Debug)]
pub struct SwapchainImage {
    /// ID of the swapchain that owns this image.
    pub swapchain_id: SwapchainId,
    /// Handle of the swapchain that owns this image.
    pub swapchain_handle: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub image_index: u32,
    pub image_info: ImageInfo,
}

slotmap::new_key_type! {
    pub struct SwapchainId;
    pub struct RenderPassId;
    pub struct PipelineLayoutId;
}

type SwapchainMap = SlotMap<SwapchainId, Swapchain>;
type RenderPassMap = SlotMap<RenderPassId, vk::RenderPass>;
//type PipelineLayoutMap = SlotMap<PipelineLayoutId, vk::PipelineLayout>;

/// Stores the set of resources owned by a currently executing frame.
#[derive(Debug)]
struct FrameInFlight {
    signalled_serials: QueueSerialNumbers,
    transient_allocations: Vec<vk_mem::Allocation>,
    command_pools: Vec<CommandAllocator>,
    /// Image views allocated for this frame.
    image_views: Vec<vk::ImageView>,
    /// Framebuffers allocated for this frame.
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_sets: Vec<DescriptorSet>,
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

/// Represents the
pub struct Context {
    device: Device,

    // --- Submission --------------------------------
    /// Timeline semaphores for each queue, used for cross-queue synchronization
    timelines: [vk::Semaphore; MAX_QUEUES],
    /// Array containing the last submitted pass serials for each queue
    last_signalled_serials: QueueSerialNumbers,
    /// Pool of recycled command pools.
    available_command_pools: Vec<CommandAllocator>,

    // --- Descriptors --------------------------------
    set_allocators: SlotMap<DescriptorSetAllocatorId, DescriptorSetAllocator>,
    cache: HashMap<DescriptorSetLayoutInfo, DescriptorSetAllocatorId>,

    // --- Resources --------------------------------
    /// Free semaphores guaranteed to be in the unsignalled state.
    semaphore_pool: Vec<vk::Semaphore>,
    resources: ResourceMap,

    /// The serial to be used for the next pass (used by `Frame`)
    next_serial: u64,
    /// Array containing the last completed pass serials for each queue
    completed_serials: QueueSerialNumbers,
    /// Number of submitted frames
    submitted_frame_count: u64,
    /// Number of completed frames
    completed_frame_count: u64,
    /// Swapchains.
    swapchains: SwapchainMap,
    /// ID-mapped render passes. These are used mostly to store the render passes for
    /// fragment output interface types, but still be able to delete the objects when the context
    /// is dropped.
    render_passes: RenderPassMap,
    /// ID-mapped pipeline layout associated to `PipelineInterface` types.
    //pipeline_layouts: PipelineLayoutMap,
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
            set_allocators: Default::default(),
            completed_serials: Default::default(),
            next_serial: 0,
            submitted_frame_count: 0,
            completed_frame_count: 0,
            resources: SlotMap::with_key(),
            swapchains: SlotMap::with_key(),
            render_passes: SlotMap::with_key(),
            //pipeline_layouts: SlotMap::with_key(),
            in_flight: VecDeque::new(),
            cache: Default::default(),
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
    fn create_semaphore(&mut self) -> vk::Semaphore {
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

            // We can recycle the memory allocated for descriptor sets.
            unsafe {
                self.recycle_descriptor_sets(f.descriptor_sets);
            }

            // Destroy all other frame-bound objects (framebuffers, image views, descriptor sets)
            for fb in f.framebuffers {
                trace!(?fb, "destroy_framebuffer");
                unsafe {
                    self.device.device.destroy_framebuffer(fb, None);
                }
            }
            for img in f.image_views {
                trace!(?img, "destroy_image_view");
                unsafe {
                    self.device.device.destroy_image_view(img, None);
                }
            }

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

    /// Creates a swapchain for the given surface.
    /// This function will automatically choose a "best" format among those supported for the
    /// surface.
    pub unsafe fn create_swapchain(
        &mut self,
        surface: vk::SurfaceKHR,
        initial_size: (u32, u32),
    ) -> SwapchainInfo {
        let swapchain = Swapchain::new(&self.device, surface, initial_size);
        let handle = swapchain.handle;
        let format = swapchain.format;

        let id = self.swapchains.insert(swapchain);

        SwapchainInfo { id, handle, format }
    }

    /// Acquires the next image in the swapchain.
    /// See `vkAcquireNextImageKHR`.
    pub unsafe fn acquire_next_image(&mut self, swapchain_id: SwapchainId) -> SwapchainImage {
        let image_available = self.create_semaphore();
        let swapchain = self.swapchains.get_mut(swapchain_id).unwrap();
        let swapchain_handle = swapchain.handle;
        let (image_index, _suboptimal) =
            swapchain.acquire_next_image(&self.device, image_available);

        // create a resource entry so that we can track usages of the swapchain image
        let image_id = self.resources.insert(Resource {
            name: format!("swapchain {:?} image #{}", swapchain.handle, image_index),
            user_ref_count: 0,
            tracking: ResourceTrackingInfo {
                wait_binary_semaphore: image_available,
                ..Default::default()
            },
            memory: ResourceMemory::External,
            // swapchain images are owned by the swapchain so we shouldn't delete them
            should_delete: false,
            kind: ResourceKind::Image(ImageResource {
                handle: swapchain.images[image_index as usize],
                format: swapchain.format,
            }),
        });

        SwapchainImage {
            swapchain_id,
            swapchain_handle,
            image_info: ImageInfo {
                id: ImageId(image_id),
                handle: swapchain.images[image_index as usize],
            },
            image_index,
        }
    }

    ///
    pub unsafe fn resize_swapchain(&mut self, swapchain: SwapchainId, size: (u32, u32)) {
        let swapchain = self.swapchains.get_mut(swapchain).unwrap();
        swapchain.resize(&self.device, size);
    }

    //
    pub unsafe fn destroy_swapchain(&mut self, _swapchain: SwapchainId) {
        // TODO wait for the device to finish, then destroy swapchain
        unimplemented!()
    }

    /// Gets or creates the associated render pass object for the specified fragment output interface type.
    pub fn get_or_create_render_pass_from_interface<T: FragmentOutputInterface>(
        &mut self,
    ) -> vk::RenderPass {
        let id = T::get_or_init_render_pass(|| {
            let render_pass = T::create_render_pass(self);
            self.render_passes.insert(render_pass)
        });

        *self.render_passes.get(id).unwrap()
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
