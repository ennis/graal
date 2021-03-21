use std::{
    alloc::Layout,
    collections::{HashMap, VecDeque},
    ffi::CString,
    fmt, mem,
    mem::MaybeUninit,
    os::raw::c_void,
    ptr, slice,
};

use ash::{
    version::{DeviceV1_0, DeviceV1_1, DeviceV1_2},
    vk,
    vk::BufferUsageFlags,
};
use fixedbitset::FixedBitSet;
use slotmap::{SecondaryMap, SlotMap};

use crate::{context::{
    descriptor::{DescriptorSet, DescriptorSetAllocator},
    resource::{
        ImageResource, Resource, ResourceId, ResourceKind, ResourceMap, ResourceMemory,
        ResourceTrackingInfo,
    },
    submission::CommandAllocator,
}, device::Device, swapchain::Swapchain, vk::Handle, DescriptorSetInterface, DescriptorSetLayoutBindingInfo, DescriptorSetLayoutInfo, MAX_QUEUES, ImageInfo};
use std::{
    cmp::Ordering,
    ops::{Index, IndexMut},
};

pub(crate) mod batch;
pub(crate) mod descriptor;
pub(crate) mod pass;
pub(crate) mod resource;
pub(crate) mod submission;

pub use batch::{Batch, PassBuilder};
pub use descriptor::DescriptorSetAllocatorId;
use crate::context::resource::{ImageId, BufferId};

/// A number that uniquely identifies a batch.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct BatchSerialNumber(pub u64);

/// A number that combines the serial number of a pass and the queue it was submitted on.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct SubmissionNumber(u64);

impl SubmissionNumber {
    /// Creates a new submission number from a queue index and a serial.
    pub fn new(queue_index: u8, serial: u64) -> SubmissionNumber {
        assert!(serial < 1u64 << 62);
        SubmissionNumber(((queue_index as u64) << 62) | serial)
    }

    /// The queue that the pass is submitted on.
    pub const fn queue(&self) -> u8 {
        (self.0 >> 62) as u8
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
    pub fn new() -> QueueSerialNumbers {
        Default::default()
    }

    pub fn serial(&self, queue: u8) -> u64 {
        self.0[queue as usize]
    }

    pub fn assign_max_serial(&mut self, queue: u8, other: u64) {
        self.0[queue as usize] = self.0[queue as usize].max(other);
    }

    pub fn iter(&self) -> impl Iterator<Item = &'_ u64> {
        self.0.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut u64> {
        self.0.iter_mut()
    }
}

impl Index<u8> for QueueSerialNumbers {
    type Output = u64;

    fn index(&self, index: u8) -> &Self::Output {
        &self.0[index as usize]
    }
}

impl IndexMut<u8> for QueueSerialNumbers {
    fn index_mut(&mut self, index: u8) -> &mut Self::Output {
        &mut self.0[index as usize]
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
            format!("{}.{}", name, serial)
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

pub unsafe fn place_aligned(layout: &Layout, ptr: &mut *mut u8, space: &mut usize) -> *mut u8 {
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
}

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

fn format_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
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

#[derive(Copy, Clone, Debug)]
pub struct SwapchainImage {
    pub swapchain_id: SwapchainId,
    pub swapchain_handle: vk::SwapchainKHR,
    pub image_index: u32,
    pub image_info: ImageInfo
}

slotmap::new_key_type! {
    pub struct SwapchainId;
}

type SwapchainMap = SlotMap<SwapchainId, Swapchain>;

/// Stores the set of resources owned by a currently executing batch.
struct InFlightBatch {
    signalled_serials: QueueSerialNumbers,
    transient_allocations: Vec<vk_mem::Allocation>,
    command_pools: Vec<CommandAllocator>,
    /// Image views allocated for this batch.
    image_views: Vec<vk::ImageView>,
    /// Framebuffers allocated for this batch.
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_sets: Vec<DescriptorSet>,
    semaphores: Vec<vk::Semaphore>,
}

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

    /// The serial to be used for the next pass (used by `Batch`)
    next_serial: u64,
    /// Array containing the last completed pass serials for each queue
    completed_serials: QueueSerialNumbers,
    /// Number of submitted batches
    submitted_batch_count: u64,
    /// Number of completed batches
    completed_batch_count: u64,
    /// Swapchains.
    swapchains: SwapchainMap,
    /// Batches that are currently executing on the GPU.
    in_flight: VecDeque<InFlightBatch>,
}

impl Context {
    /// Creates a new context.
    pub fn new(device: Device) -> Context {
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
            submitted_batch_count: 0,
            completed_batch_count: 0,
            resources: ResourceMap::with_key(),
            swapchains: SlotMap::with_key(),
            in_flight: VecDeque::new(),
            cache: Default::default(),
            semaphore_pool: vec![],
        }
    }

    /// Returns the `ash::Device` associated with this context.
    pub fn device(&self) -> &ash::Device {
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

    /// Returns whether the given batch, identified by its serial, has completed execution.
    pub fn is_batch_completed(&self, serial: BatchSerialNumber) -> bool {
        self.completed_batch_count >= serial.0
    }

    /*fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
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
    }*/

    fn get_next_serial(&mut self) -> u64 {
        self.next_serial += 1;
        self.next_serial
    }

    /// Waits for all but the last submitted batch to finish and then recycles their resources.
    /// Calls `cleanup_resources` internally.
    fn wait_for_batches_in_flight(&mut self) {
        // pacing
        while self.in_flight.len() >= 2 {
            // two batches in flight already, must wait for the oldest one
            let mut b = self.in_flight.pop_front().unwrap();
            self.wait(&b.signalled_serials);

            // update completed serials
            // we just waited on those serials, so we know they are completed
            self.completed_serials = b.signalled_serials;

            // Recycle the command pools allocated for the batch. The allocated command buffers
            // can then be reused for future submissions.
            self.recycle_command_pools(b.command_pools);

            // Recycle the semaphores. They are guaranteed to be unsignalled since the batch must have
            // waited on them.
            self.recycle_semaphores(b.semaphores);

            // We can recycle the memory allocated for descriptor sets.
            unsafe {
                self.recycle_descriptor_sets(b.descriptor_sets);
            }

            // Destroy all other batch-bound objects (framebuffers, image views, descriptor sets)
            for fb in b.framebuffers {
                unsafe {
                    self.device.device.destroy_framebuffer(fb, None);
                }
            }
            for img in b.image_views {
                unsafe {
                    self.device.device.destroy_image_view(img, None);
                }
            }

            // free transient allocations
            for alloc in b.transient_allocations.iter() {
                self.device.allocator.free_memory(alloc).unwrap();
            }

            // bump completed batch count
            self.completed_batch_count += 1;
        }

        // given the new completed serials, free resources that have expired
        self.cleanup_resources();
    }

    fn dump_state(&self) {
        /*println!("Number of batches in flight: {}", self.in_flight.len());
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
        println!("VMA stats:");
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
        }
    }

    pub fn current_batch_index(&self) -> u64 {
        self.submitted_batch_count
    }

    pub unsafe fn create_swapchain(
        &mut self,
        surface: vk::SurfaceKHR,
        initial_size: (u32, u32),
    ) -> SwapchainId {
        self.swapchains
            .insert(Swapchain::new(&self.device, surface, initial_size))
    }

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
                handle: swapchain.images[image_index as usize]
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
    pub fn destroy_swapchain(&mut self, swapchain: SwapchainId) {
        unimplemented!()
    }

    pub fn start_batch(&mut self) -> Batch {
        Batch::new(self)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // TODO
    }
}
