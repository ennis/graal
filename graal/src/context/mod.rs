use crate::{
    context::submission::CommandAllocator,
    device::Device,
    resource::{ResourceKind, ResourceTrackingInfo},
    serial::{FrameNumber, QueueSerialNumbers, SubmissionNumber},
    BufferId, ImageId, ResourceId, MAX_QUEUES,
};
use ash::vk;
use std::{collections::VecDeque, fmt, os::raw::c_void, sync::Arc};
use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::sync::Mutex;
pub use submission::RecordingContext;
use tracing::{trace, trace_span};
use crate::resource::DeviceObjects;

pub(crate) mod frame;
pub(crate) mod submission;
pub(crate) mod transient;

/// Maximum time to wait for batches to finish in `SubmissionState::wait`.
pub(crate) const SEMAPHORE_WAIT_TIMEOUT_NS: u64 = 1_000_000_000;

/// TODO document
fn local_pass_index(serial: u64, frame_base_serial: u64) -> usize {
    assert!(serial > frame_base_serial);
    (serial - frame_base_serial - 1) as usize
}

pub(crate) fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        0 => vk::SampleCountFlags::TYPE_1,
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

pub fn is_write_access(mask: vk::AccessFlags) -> bool {
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

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM_S8_UINT => true,
        vk::Format::D24_UNORM_S8_UINT => true,
        vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM => true,
        vk::Format::X8_D24_UNORM_PACK32 => true,
        vk::Format::D32_SFLOAT => true,
        _ => false,
    }
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
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

/*/// Information about the memory to be allocated for a resource.
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
}*/

/// Represents a resource access in a pass.
#[derive(Debug)]
pub(crate) struct ResourceAccessDetails {
    pub(crate) initial_layout: vk::ImageLayout,
    pub(crate) final_layout: vk::ImageLayout,
    pub(crate) access_mask: vk::AccessFlags,
    pub(crate) stage_mask: vk::PipelineStageFlags,
}

/*/// Holds information about a buffer resource containing an array of elements of type T.
#[derive(Copy, Clone, Debug)]
pub struct TypedBufferInfo<T> {
    /// ID of the buffer resource.
    pub id: BufferId,
    /// Vulkan handle of the buffer.
    pub handle: vk::Buffer,
    /// If the buffer is mapped in client memory, holds a pointer to the mapped range. Null otherwise.
    pub mapped_ptr: Option<NonNull<T>>,
}

/// `TypedBufferInfo<T>` are convertible into their untyped equivalents.
impl<T> From<TypedBufferInfo<T>> for BufferInfo {
    fn from(buf: TypedBufferInfo<T>) -> Self {
        BufferInfo {
            id: buf.id,
            handle: buf.handle,
            mapped_ptr: buf.mapped_ptr.map(|ptr| ptr.cast()),
        }
    }
}

impl<T: Copy> TypedBufferInfo<T> {
    pub unsafe fn byte_cast<U>(&self) -> TypedBufferInfo<U>
    where
        T: Sized,
        U: Sized,
    {
        // TODO static assert?
        assert_eq!(mem::size_of::<T>(), mem::size_of::<U>());
        TypedBufferInfo {
            id: self.id,
            handle: self.handle,
            mapped_ptr: self.mapped_ptr.map(|ptr| ptr.cast()),
        }
    }
}*/

// ---------------------------------------------------------------------------------------------

// ---------------------------------------------------------------------------------------------

/// Stores the set of resources owned by a currently executing frame.
#[derive(Debug)]
struct FrameInFlight {
    signalled_serials: QueueSerialNumbers,
    //transient_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    command_pools: Vec<CommandAllocator>,
    semaphores: Vec<vk::Semaphore>,
    //image_views: Vec<vk::ImageView>,
    //framebuffers: Vec<vk::Framebuffer>,
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

#[derive(Copy, Clone, Debug, Hash, Eq, PartialEq)]
pub(crate) struct ResourceAccess {
    pub(crate) id: ResourceId,
    pub(crate) access_mask: vk::AccessFlags,
}

pub(crate) enum PassEvaluationCallback<'a, UserContext> {
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
    Queue(Box<dyn FnOnce(&mut RecordingContext, &mut UserContext, vk::Queue) + 'a>),
    CommandBuffer(Box<dyn FnOnce(&mut RecordingContext, &mut UserContext, vk::CommandBuffer) + 'a>),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreWaitKind {
    Binary,
    Timeline(u64),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreSignalKind {
    Binary,
    Timeline(u64),
}

/// Represents a semaphore wait operation outside of the queue timelines.
#[derive(Clone, Debug)]
pub struct SemaphoreWait {
    /// The semaphore in question
    pub(crate) semaphore: vk::Semaphore,
    /// Whether the semaphore is internally managed (owned by the context).
    /// If true, the semaphore will be reclaimed by the context after it is consumed (waited on).
    pub(crate) owned: bool,
    /// Destination stage
    pub(crate) dst_stage: vk::PipelineStageFlags,
    /// The kind of wait operation.
    pub(crate) wait_kind: SemaphoreWaitKind,
}

#[derive(Clone, Debug)]
pub(crate) struct SemaphoreSignal {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) signal_kind: SemaphoreSignalKind,
}

/// A pass within a frame.
pub(crate) struct Pass<'a, UserContext> {
    name: String,

    /// Submission number of the pass.
    snn: SubmissionNumber,

    /// Index of the pass in the frame.
    frame_index: usize,

    /// Predecessors of the pass (all passes that must happen before this one).
    preds: Vec<usize>,

    /// Successors of the pass (all passes for which this task is a predecessor).
    //pub(crate) succs: Vec<usize>,

    /// List of accesses made by the pass to resources.
    // FIXME Right now, this is used only for debugging purposes, and when allocating memory for the resources.
    // It probably could be removed.
    pub(crate) accesses: HashSet<ResourceAccess>,

    /// Whether the queue timeline semaphores must be signalled after the pass.
    pub(crate) signal_queue_timelines: bool,

    pub(crate) src_stage_mask: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask: vk::PipelineStageFlags,
    pub(crate) image_memory_barriers: Vec<vk::ImageMemoryBarrier>,
    pub(crate) buffer_memory_barriers: Vec<vk::BufferMemoryBarrier>,
    pub(crate) global_memory_barrier: Option<vk::MemoryBarrier>,

    pub(crate) wait_serials: QueueSerialNumbers,
    pub(crate) wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],

    pub(crate) external_semaphore_waits: Vec<SemaphoreWait>,
    pub(crate) external_semaphore_signals: Vec<SemaphoreSignal>,

    pub(crate) eval_callback: Option<PassEvaluationCallback<'a, UserContext>>,
}

impl<'a, UserContext> Pass<'a, UserContext> {
    pub(crate) fn get_or_create_image_memory_barrier(
        &mut self,
        handle: vk::Image,
        format: vk::Format,
    ) -> &mut vk::ImageMemoryBarrier {
        if let Some(b) = self
            .image_memory_barriers
            .iter_mut()
            .position(|b| b.image == handle)
        {
            &mut self.image_memory_barriers[b]
        } else {
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: format_aspect_mask(format),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            self.image_memory_barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::empty(),
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::UNDEFINED,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: handle,
                subresource_range,
                ..Default::default()
            });
            self.image_memory_barriers.last_mut().unwrap()
        }
    }

    pub(crate) fn get_or_create_buffer_memory_barrier(
        &mut self,
        handle: vk::Buffer,
    ) -> &mut vk::BufferMemoryBarrier {
        if let Some(b) = self
            .buffer_memory_barriers
            .iter_mut()
            .position(|b| b.buffer == handle)
        {
            &mut self.buffer_memory_barriers[b]
        } else {
            self.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::empty(),
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: handle,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            });
            self.buffer_memory_barriers.last_mut().unwrap()
        }
    }

    pub(crate) fn get_or_create_global_memory_barrier(&mut self) -> &mut vk::MemoryBarrier {
        self.global_memory_barrier
            .get_or_insert_with(Default::default)
    }

    pub(crate) fn new(
        name: &str,
        frame_index: usize,
        snn: SubmissionNumber,
    ) -> Pass<'a, UserContext> {
        Pass {
            name: name.to_string(),
            snn,
            preds: vec![],
            //succs: vec![],
            accesses: HashSet::new(),
            signal_queue_timelines: false,
            src_stage_mask: Default::default(),
            dst_stage_mask: Default::default(),
            image_memory_barriers: vec![],
            buffer_memory_barriers: vec![],
            global_memory_barrier: None,
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            external_semaphore_waits: vec![],
            external_semaphore_signals: vec![],
            frame_index,
            eval_callback: None,
        }
    }
}

type TemporarySet = std::collections::BTreeSet<ResourceId>;

/// Collected debugging information about a frame.
struct SyncDebugInfo {
    tracking: slotmap::SecondaryMap<ResourceId, ResourceTrackingInfo>,
    xq_sync_table: [QueueSerialNumbers; MAX_QUEUES],
}

impl SyncDebugInfo {
    fn new() -> SyncDebugInfo {
        SyncDebugInfo {
            tracking: Default::default(),
            xq_sync_table: Default::default(),
        }
    }
}

pub(crate) struct FrameInner<'a, UserContext> {
    frame_number: FrameNumber,
    span: tracing::span::EnteredSpan,
    build_span: tracing::span::EnteredSpan,
    base_sn: u64,
    current_sn: u64,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the frame
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass<'a, UserContext>>,
    /// Serials to wait for before executing the frame.
    wait_init: QueueSerialNumbers,

    /// Cross-queue synchronization table.
    ///
    /// This table tracks, for each queue, the latest passes on every other queue for which we
    /// have inserted an execution dependency in the command stream.
    ///
    /// By construction, we can ensure that all subsequent commands on `dst_queue` will happen after all passes
    /// on `src_queue` with a SN lower than or equal to `xq_sync_table[dst_queue][src_queue]`.
    ///
    ///
    /// # Example
    /// Consider `MAX_QUEUES = 4`. The table starts initialized to zero.
    /// We have 4 passes: with SNNs `0:1`, `1:2`, `2:3`, `0:4`, with dependencies:
    /// 1 -> 2, 1 -> 3 and (2,3) -> 4.
    ///
    /// The sync table starts empty:
    /// ```text
    ///     SRC_Q:  Q0  Q1  Q2  Q3
    ///  DST_Q:
    ///    Q0:   [  0   0   0   0 ]
    ///    Q1:   [  0   0   0   0 ]
    ///    Q2:   [  0   0   0   0 ]
    ///    Q3:   [  0   0   0   0 ]
    /// ```
    ///
    /// When submitting pass `1:2`, we insert a wait on Q1, for SN 1 on Q0.
    /// We can update the table as follows:
    /// ```text
    ///     SRC_Q:  Q0  Q1  Q2  Q3
    ///  DST_Q:
    ///    Q0:   [  0   0   0   0  ]
    ///    Q1:   [  1   0   0   0  ]
    ///    Q2:   [  0   0   0   0  ]
    ///    Q3:   [  0   0   0   0  ]
    /// ```
    ///
    /// Similarly, when submitting pass `2:3`, we insert a wait on Q2, for SN 1 on Q0:
    /// ```text
    ///     SRC_Q:  Q0  Q1  Q2  Q3
    ///  DST_Q:
    ///    Q0:   [  0   0   0   0  ]
    ///    Q1:   [  1   0   0   0  ]
    ///    Q2:   [  1   0   0   0  ]
    ///    Q3:   [  0   0   0   0  ]
    /// ```
    ///
    /// Finally, when submitting pass `0:4`, we insert a wait on Q0, for SN 2 on Q1 and SN 3 on Q2:
    /// The final state of the sync table is:
    /// ```text
    ///     SRC_Q:  Q0  Q1  Q2  Q3
    ///  DST_Q:
    ///    Q0:   [  0   2   3   0  ]
    ///    Q1:   [  1   0   0   0  ]
    ///    Q2:   [  1   0   0   0  ]
    ///    Q3:   [  0   0   0   0  ]
    /// ```
    ///
    /// This tells us that, in the current state of the command stream:
    /// - Q0 has waited for pass SN 2 on Q1, and pass SN 3 on Q2
    /// - Q1 has waited for pass SN 1 on Q0
    /// - Q2 has also waited for pass SN 1 on Q0
    /// - Q3 hasn't synchronized with anything
    xq_sync_table: [QueueSerialNumbers; MAX_QUEUES],

    collect_sync_debug_info: bool,
    sync_debug_info: Vec<SyncDebugInfo>,
    //descriptor_sets: Vec<vk::DescriptorSet>,
    //framebuffers: Vec
}

///
pub struct Frame<'a, UserContext> {
    context: &'a mut Context,
    inner: FrameInner<'a, UserContext>,
    //current_pass: Option<Pass<'a, UserContext>>,
}

pub(crate) struct ContextState {
    /// Whether we are between `start_frame`/`end_frame`.
    pub(crate) is_building_frame: Cell<bool>,
    /// Last started frame
    pub(crate) last_started_frame: Cell<FrameNumber>,
}

pub struct Context {
    pub(crate) device: Arc<Device>,
    /// Free semaphores guaranteed to be in the unsignalled state.
    pub(crate) semaphore_pool: Vec<vk::Semaphore>,
    /// Timeline semaphores for each queue, used for cross-queue and inter-frame synchronization
    pub(crate) timelines: [vk::Semaphore; MAX_QUEUES],
    /// Array containing the last submitted pass serials for each queue
    pub(crate) last_signalled_serials: QueueSerialNumbers,
    /// Pool of recycled command pools.
    pub(crate) available_command_pools: Vec<CommandAllocator>,
    /// Array containing the last completed pass serials for each queue
    pub(crate) completed_serials: QueueSerialNumbers,
    /// The serial to be used for the next pass (used by `Frame`)
    pub(crate) last_sn: u64,
    /// Frames that are currently executing on the GPU.
    in_flight: VecDeque<FrameInFlight>,
    /// Number of submitted frames
    pub(crate) submitted_frame_count: u64,
    /// Number of completed frames
    pub(crate) completed_frame_count: u64,
}


impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO good luck with that
        f.debug_struct("Context").finish()
    }
}

impl Context {

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
            device: Arc::new(device),
            timelines,
            last_signalled_serials: Default::default(),
            available_command_pools: vec![],
            completed_serials: Default::default(),
            semaphore_pool: vec![],
            last_sn: 0,
            submitted_frame_count: 0,
            completed_frame_count: 0,
            in_flight: VecDeque::new(),
        }
    }

    /// Returns the `graal::Device` owned by this context.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the `ash::Device` owned by this context.
    /// Shorthand for `self.device().device`.
    pub fn vulkan_device(&self) -> &ash::Device {
        &self.device.device
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
    pub fn is_frame_completed(&self, serial: FrameNumber) -> bool {
        self.completed_frame_count >= serial.0
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

            // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
            /*// free transient allocations
            for alloc in f.transient_allocations {
                trace!(?alloc, "free_memory");
                self.device.allocator.borrow_mut().free(alloc).unwrap();
            }*/

            // bump completed frame count
            self.completed_frame_count += 1;
        }

        // given the new completed serials, free resources that have expired
        self.device.cleanup_resources(self.completed_serials);
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

    pub fn current_frame_number(&self) -> FrameNumber {
        //assert!(self.is_building_frame, "not building a frame");
        FrameNumber(self.submitted_frame_count + 1)
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
