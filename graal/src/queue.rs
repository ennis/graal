use crate::{
    command_allocator::CommandBufferAllocator,
    device::{
        BufferId, Device, GroupId, ImageId, ImageInfo, ImageRegistrationInfo, QueueOwnership, ResourceAllocation,
        ResourceId, ResourceKind, ResourceMap, ResourceRegistrationInfo, Swapchain, SwapchainImage,
    },
    is_write_access, vk,
};
use ash::{prelude::VkResult, vk::Buffer};
use std::{collections::VecDeque, ffi::c_void, mem, ops::Deref, ptr, rc::Rc, time::Duration};
use tracing::debug;

/// A wrapper around a signaled binary semaphore.
#[derive(Debug)]
pub struct SignaledSemaphore(pub(crate) vk::Semaphore);

/// A wrapper around an unsignaled binary semaphore.
#[derive(Debug)]
pub struct UnsignaledSemaphore(pub(crate) vk::Semaphore);

pub struct ResourceState {
    pub stages: vk::PipelineStageFlags2,
    pub access: vk::AccessFlags2,
    pub layout: vk::ImageLayout,
}

impl ResourceState {
    pub const TRANSFER_SRC: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    };
    pub const TRANSFER_DST: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    };
    pub const SHADER_READ: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::ALL_GRAPHICS,
        access: vk::AccessFlags2::SHADER_READ,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    pub const COLOR_ATTACHMENT_OUTPUT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    pub const PRESENT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
        layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };
}

/// Wrapper around a Vulkan queue that tracks the use of resources.
pub struct Queue {
    device: Rc<Device>,
    queue_index: usize,
    queue_family_index: u32,
    queue: vk::Queue,
    timeline: vk::Semaphore,
    free_cb_allocs: Vec<CommandBufferAllocator>,

    /// Last signalled value on the timeline semaphore of the queue.
    last_signaled: u64,

    //------ Recycled resources ------
    /// Binary semaphores for which we've submitted a wait operation.
    semaphores: Vec<UnsignaledSemaphore>,

    // ------ in flight resources ------
    /// Resources of frames that are currently executing on the GPU.
    current_frame: InFlightResources,
    submitted: VecDeque<InFlightResources>,
}

/// In-flight resources.
#[derive(Debug)]
struct InFlightResources {
    timeline_value: u64,
    /// Command pool
    cb_alloc: CommandBufferAllocator,
    /// Deferred deletion list
    deferred_delete: Vec<DeferredDestroyObject>,
}

impl Queue {
    /// Creates a new queue.
    pub(crate) unsafe fn new(device: Rc<Device>, queue_index: usize) -> Queue {
        let queue = device.queue(queue_index);
        let timeline = device.queue_timeline(queue_index);
        let queue_family_index = device.queue_family_index(queue_index);
        let raw_device = &device.device;
        let cb_alloc = CommandBufferAllocator::new(raw_device, queue_family_index);

        Queue {
            device,
            queue_index,
            queue,
            timeline,
            free_cb_allocs: vec![],
            last_signaled: 0,
            semaphores: vec![],
            queue_family_index,
            current_frame: InFlightResources {
                timeline_value: 1,
                cb_alloc,
                deferred_delete: vec![],
            },
            submitted: Default::default(),
        }
    }

    /// Returns the device associated with this queue.
    pub fn device(&self) -> &Rc<Device> {
        &self.device
    }

    /// Creates a new, or returns an existing, binary semaphore that is in the unsignaled state,
    /// or for which we've submitted a wait operation on this queue and that will eventually be unsignaled.
    pub fn get_or_create_semaphore(&mut self) -> UnsignaledSemaphore {
        // Try to recycle one
        if let Some(semaphore) = self.semaphores.pop() {
            return semaphore;
        }

        // Otherwise create a new one
        unsafe {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            UnsignaledSemaphore(
                self.device
                    .device
                    .create_semaphore(&create_info, None)
                    .expect("vkCreateSemaphore failed"),
            )
        }
    }

    /// Allocates a command buffer.
    pub fn allocate_command_buffer(&mut self) -> vk::CommandBuffer {
        self.current_frame.cb_alloc.alloc(&self.device.device)
    }

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(
        &mut self,
        swapchain: &Swapchain,
        timeout: Duration,
    ) -> Result<SwapchainImage, vk::Result> {
        let image_available = self.get_or_create_semaphore();
        let (image_index, _suboptimal) = match self.device.vk_khr_swapchain.acquire_next_image(
            swapchain.handle,
            timeout.as_nanos() as u64,
            image_available.0,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // recycle the semaphore before returning
                self.semaphores.push(image_available);
                return Err(err);
            }
        };

        let handle = swapchain.images[image_index as usize];
        let name = format!("swapchain {:?} image #{}", handle, image_index);
        let id = self.device.register_image_resource(ImageRegistrationInfo {
            resource: ResourceRegistrationInfo {
                name: &name,
                allocation: ResourceAllocation::External,
                initial_wait: Some(SemaphoreWait {
                    kind: SemaphoreWaitKind::Binary {
                        semaphore: image_available.0,
                        transfer_ownership: true,
                    },
                    dst_stage: Default::default(),
                }),
            },
            handle,
            format: swapchain.format.format,
        });

        Ok(SwapchainImage {
            swapchain: swapchain.handle,
            image_info: ImageInfo { id, handle },
            image_index,
        })
    }

    pub unsafe fn present(
        &mut self,
        render_finished: SignaledSemaphore,
        swapchain_image: &SwapchainImage,
    ) -> VkResult<bool> {
        // build present info that waits on the batch that was just submitted
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &render_finished.0,
            swapchain_count: 1,
            p_swapchains: &swapchain_image.swapchain,
            p_image_indices: &swapchain_image.image_index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };
        let result = self.device.vk_khr_swapchain.queue_present(self.queue, &present_info);
        // we signalled and waited on the semaphore, consider it consumed
        self.semaphores.push(UnsignaledSemaphore(render_finished.0));
        result
    }
}

struct ResourceUse {
    id: ResourceId,
    access_mask: vk::AccessFlags2,
    stage_mask: vk::PipelineStageFlags2,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

/// Describes the kind of semaphore wait operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreWaitKind {
    /// Binary semaphore wait.
    Binary {
        /// The semaphore to wait on.
        semaphore: vk::Semaphore,
        /// Whether to transfer ownership of the semaphore to the queue.
        transfer_ownership: bool,
    },
    /// Timeline semaphore wait.
    Timeline { semaphore: vk::Semaphore, value: u64 },
    /// D3D12 fence wait.
    D3D12Fence {
        semaphore: vk::Semaphore,
        fence: vk::Fence,
        value: u64,
    },
}

/// Describe the kind of semaphore signal operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreSignal {
    /// Binary semaphore signal.
    Binary {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
    },
    /// Timeline semaphore signal.
    Timeline {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The value to signal.
        value: u64,
    },
    /// D3D12 fence signal.
    D3D12Fence {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The fence to signal.
        fence: vk::Fence,
        /// The value to signal.
        value: u64,
    },
}

/// Represents a semaphore wait operation.
#[derive(Clone, Debug)]
pub struct SemaphoreWait {
    /// The kind of wait operation.
    pub kind: SemaphoreWaitKind,
    /// Destination stage
    pub dst_stage: vk::PipelineStageFlags,
}

pub struct Submission<'a> {
    queue: &'a mut Queue,
    name: Option<String>,
    /// Signals of the next submission.
    signals: Vec<SemaphoreSignal>,
    /// Semaphore waits of the next submission.
    waits: Vec<SemaphoreWait>,
    /// Command buffers ready to be submitted.
    cbs: Vec<vk::CommandBuffer>,
    /// Resource uses
    uses: Vec<ResourceUse>,
    /// Resource group uses
    group_uses: Vec<GroupId>,
    /// Ownership transfers
    ownership_transfers: Vec<QueueOwnershipTransfer>,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum QueueOwnershipTransferKind {
    Release,
    AcquireExclusive,
    AcquireConcurrent,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct QueueOwnershipTransfer {
    resource: ResourceId,
    kind: QueueOwnershipTransferKind,
}

/// Helper to build a pipeline barrier.
#[derive(Default)]
struct PipelineBarrierBuilder {
    queue: u32,
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
}

impl<'a> Submission<'a> {
    fn new(queue: &'a mut Queue) -> Submission<'a> {
        Submission {
            queue,
            name: None,
            signals: vec![],
            waits: vec![],
            cbs: vec![],
            uses: vec![],
            group_uses: vec![],
            ownership_transfers: vec![],
        }
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
    }

    /// Makes sure that the given image is in the given layout, and emits the necessary barriers
    /// to transition it if needed.
    ///
    /// # Arguments
    ///
    /// * stage_mask: the pipeline stage at which the image will be used
    /// * access_mask: the access mask for the image (defines both the visiblity requirements and flushes)
    pub fn use_image(&mut self, image: ImageId, state: ResourceState) {
        self.uses.push(ResourceUse {
            id: image.into(),
            access_mask: state.access,
            stage_mask: state.stages,
            initial_layout: state.layout,
            final_layout: state.layout,
        });
    }

    /// Emits the necessary barriers
    pub fn use_buffer(&mut self, buffer: BufferId, state: ResourceState) {
        self.uses.push(ResourceUse {
            id: buffer.into(),
            access_mask: state.access,
            stage_mask: state.stages,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::UNDEFINED,
        });
    }

    pub fn use_group(&mut self, group: GroupId) {
        self.group_uses.push(group);
    }

    /// Adds a semaphore signal operation: when finished, the pass will signal the specified semaphore.
    pub fn signal_raw(&mut self, signal: SemaphoreSignal) {
        self.signals.push(signal);
    }

    /// Waits for an external semaphore to be signalled.
    pub fn wait_raw(&mut self, wait: SemaphoreWait) {
        self.waits.push(wait);
    }

    /// Signals a binary semaphore on this queue.
    pub fn signal(&mut self) -> SignaledSemaphore {
        let semaphore = self.queue.get_or_create_semaphore();
        self.signal_raw(SemaphoreSignal::Binary { semaphore: semaphore.0 });
        SignaledSemaphore(semaphore.0)
    }

    /// Waits for and consumes a signaled binary semaphore.
    pub fn wait(&mut self, semaphore: SignaledSemaphore, dst_stage: vk::PipelineStageFlags) {
        self.wait_raw(SemaphoreWait {
            kind: SemaphoreWaitKind::Binary {
                semaphore: semaphore.0,
                transfer_ownership: true,
            },
            dst_stage,
        });
    }

    pub fn add_to_group(&mut self, resource: impl Into<ResourceId>, group: GroupId) {
        //self.group_uses.push(group);
    }

    /// Exports an image for use in another queue.
    pub fn queue_release(&mut self, resource: impl Into<ResourceId>) {
        self.ownership_transfers.push(QueueOwnershipTransfer {
            resource: resource.into(),
            kind: QueueOwnershipTransferKind::Release,
        });
    }

    /// Acquires an resource for exclusive (read and/or write) use on this queue.
    pub fn queue_acquire_exclusive(&mut self, resource: impl Into<ResourceId>) {
        self.ownership_transfers.push(QueueOwnershipTransfer {
            resource: resource.into(),
            kind: QueueOwnershipTransferKind::AcquireExclusive,
        });
    }

    /// Acquires an resource for concurrent (read-only) use on this queue.
    pub fn queue_acquire_concurrent(&mut self, resource: impl Into<ResourceId>) {
        self.ownership_transfers.push(QueueOwnershipTransfer {
            resource: resource.into(),
            kind: QueueOwnershipTransferKind::AcquireConcurrent,
        });
    }

    pub fn push_command_buffer(&mut self, cb: vk::CommandBuffer) {
        self.cbs.push(cb);
    }

    pub unsafe fn submit(mut self) -> VkResult<()> {
        let mut barrier_builder = PipelineBarrierBuilder::default();

        {
            let mut resources = self.queue.device.resources.borrow_mut();
            let resources = &mut *resources;
            // Build barriers and update resource states
            for use_ in self.uses.iter() {
                add_resource_dependency(self.queue.queue_index, use_, resources, &mut barrier_builder);
            }
        }

        // Push a command buffer with the necessary barriers
        let barrier_cb = self.queue.allocate_command_buffer();
        let device = &self.queue.device.device;
        device.begin_command_buffer(
            barrier_cb,
            &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            },
        )?;
        device.cmd_pipeline_barrier2(
            barrier_cb,
            &vk::DependencyInfo {
                dependency_flags: Default::default(),
                memory_barrier_count: 0,
                p_memory_barriers: ptr::null(),
                buffer_memory_barrier_count: barrier_builder.buffer_barriers.len() as u32,
                p_buffer_memory_barriers: barrier_builder.buffer_barriers.as_ptr(),
                image_memory_barrier_count: barrier_builder.image_barriers.len() as u32,
                p_image_memory_barriers: barrier_builder.image_barriers.as_ptr(),
                ..Default::default()
            },
        );
        device.end_command_buffer(barrier_cb);
        self.cbs.insert(0, barrier_cb);

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        // setup semaphore signal operations
        for signal in self.signals.iter() {
            match signal {
                SemaphoreSignal::Binary { semaphore } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                }
                SemaphoreSignal::Timeline { semaphore, value } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(*value);
                }
                SemaphoreSignal::D3D12Fence {
                    semaphore,
                    fence,
                    value: _,
                } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup semaphore wait operations
        for (_i, w) in self.waits.iter().enumerate() {
            wait_semaphore_dst_stages.push(w.dst_stage);
            match w.kind {
                SemaphoreWaitKind::Binary {
                    semaphore,
                    transfer_ownership,
                } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(0);
                    if transfer_ownership {
                        self.queue.semaphores.push(UnsignaledSemaphore(semaphore));
                    }
                }
                SemaphoreWaitKind::Timeline { semaphore, value } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                }
                SemaphoreWaitKind::D3D12Fence { semaphore, value, .. } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                    d3d12_fence_submit = true;
                }
            }
        }

        // D3D12 fence submission info
        let d3d12_fence_submit_info_ptr;
        let d3d12_fence_submit_info;

        if d3d12_fence_submit {
            d3d12_fence_submit_info = vk::D3D12FenceSubmitInfoKHR {
                wait_semaphore_values_count: wait_semaphore_values.len() as u32,
                p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
                signal_semaphore_values_count: signal_semaphore_values.len() as u32,
                p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
                ..Default::default()
            };
            d3d12_fence_submit_info_ptr = &d3d12_fence_submit_info as *const _ as *const c_void;
        } else {
            d3d12_fence_submit_info_ptr = ptr::null();
        }

        let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
            p_next: d3d12_fence_submit_info_ptr,
            wait_semaphore_value_count: wait_semaphore_values.len() as u32,
            p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
            signal_semaphore_value_count: signal_semaphore_values.len() as u32,
            p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
            ..Default::default()
        };

        let submit_info = vk::SubmitInfo {
            p_next: &timeline_submit_info as *const _ as *const c_void,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_semaphore_dst_stages.as_ptr(),
            command_buffer_count: self.cbs.len() as u32,
            p_command_buffers: self.cbs.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe {
            self.queue
                .device
                .queue_submit(self.queue.queue, &[submit_info], vk::Fence::null())
        }
    }
}

/// Given a resource use, and the current state of the resource, adds the necessary barriers
/// and updates the known state of the resource.
fn add_resource_dependency(
    queue_index: usize,
    use_: &ResourceUse,
    resources: &mut ResourceMap,
    barriers: &mut PipelineBarrierBuilder,
) {
    let resource = resources.get_mut(use_.id).expect("unknown resource");
    assert!(!resource.discarded, "used a discarded resource: {:?}", resource);
    assert!(
        resource.group.is_none(),
        "add_resource_dependency cannot be called on group-synced resources"
    );
    let tracking = &mut resource.tracking;

    let writes_visible = match tracking.queue_ownership {
        QueueOwnership::None => {
            // first use, acquire ownership
            tracking.queue_ownership = QueueOwnership::Exclusive(queue_index as u16);
            // no writes yet
            true
        }
        QueueOwnership::Exclusive(queue) => {
            assert_eq!(
                queue, queue_index as u16,
                "resource is owned by another queue; use Submission::queue_release/Submission::queue_acquire to transfer ownership"
            );
            tracking.visible.contains(use_.access_mask) || tracking.visible.contains(vk::AccessFlags2::MEMORY_READ)
        }
        QueueOwnership::Concurrent(_queues) => {
            todo!("queue concurrent access")
        }
    };

    // We need to insert a memory dependency if:
    // - the image needs a layout transition
    // - the resource has writes that are not visible to the target access
    //
    // We need to insert an execution dependency if:
    // - we're writing to the resource (write-after-read hazard, write-after-write hazard)
    if !writes_visible || tracking.layout != use_.initial_layout || is_write_access(use_.access_mask) {
        match resource.kind {
            ResourceKind::Buffer(ref buffer) => {
                let barrier = barriers.get_or_create_buffer_barrier(buffer.handle);
                barrier.buffer = buffer.handle;
                barrier.src_stage_mask |= tracking.stages;
                barrier.dst_stage_mask |= use_.stage_mask;
                barrier.src_access_mask |= tracking.flush_mask;
                barrier.dst_access_mask |= use_.access_mask;
            }
            ResourceKind::Image(ref image) => {
                let barrier = barriers.get_or_create_image_barrier(image.handle);
                barrier.image = image.handle;
                barrier.src_stage_mask |= tracking.stages;
                barrier.dst_stage_mask |= use_.stage_mask;
                barrier.src_access_mask |= tracking.flush_mask;
                barrier.dst_access_mask |= use_.access_mask;
                barrier.old_layout = tracking.layout;
                assert_eq!(use_.initial_layout, use_.final_layout, "unsupported layout transition");
                barrier.new_layout = use_.initial_layout;
                barrier.subresource_range = vk::ImageSubresourceRange {
                    aspect_mask: image.all_aspects,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                    ..Default::default()
                };
                tracking.layout = use_.final_layout;
            }
        }
    }

    if is_write_access(use_.access_mask) {
        // we're writing to the resource, so reset visibility...
        tracking.visible = vk::AccessFlags2::empty();
        // ... but signal that there is data to flush.
        tracking.flush_mask |= use_.access_mask;
    } else {
        // This memory dependency makes all writes on the resource available, and
        // visible to the types specified in `access.access_mask`.
        // There's no write, so we don't need to flush anything.
        tracking.flush_mask = vk::AccessFlags2::empty();
        tracking.visible |= use_.access_mask;
    }

    // Update the resource stage mask
    tracking.stages = use_.stage_mask;
    tracking.queue_ownership = QueueOwnership::Exclusive(queue_index as u16);
}

impl Queue {
    pub fn build_submission(&mut self) -> Submission {
        Submission::new(self)
    }

    pub fn end_frame(&mut self) -> VkResult<()> {
        let submit_result = unsafe {
            let semaphores = [self.timeline];
            let values = [self.current_frame.timeline_value];
            let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
                p_next: ptr::null(),
                wait_semaphore_value_count: 0,
                p_wait_semaphore_values: ptr::null(),
                signal_semaphore_value_count: 1,
                p_signal_semaphore_values: values.as_ptr(),
                ..Default::default()
            };

            let submit_info = vk::SubmitInfo {
                p_next: &timeline_submit_info as *const _ as *const c_void,
                wait_semaphore_count: 0,
                p_wait_semaphores: ptr::null(),
                p_wait_dst_stage_mask: ptr::null(),
                command_buffer_count: 0,
                p_command_buffers: ptr::null(),
                signal_semaphore_count: 1,
                p_signal_semaphores: semaphores.as_ptr(),
                ..Default::default()
            };

            self.device.queue_submit(self.queue, &[submit_info], vk::Fence::null())
        };

        self.last_signaled = self.current_frame.timeline_value;

        // Do cleanup first so that we can recycle command pools
        self.cleanup();

        let cb_alloc = self.free_cb_allocs.pop().unwrap_or_else(|| {
            debug!("creating new command buffer allocator");
            let queue_family = self.device.queue_family_index(self.queue_index);
            // SAFETY: queue_family is valid
            unsafe { CommandBufferAllocator::new(&self.device.device, queue_family) }
        });

        self.submitted.push_front(mem::replace(
            &mut self.current_frame,
            InFlightResources {
                timeline_value: self.last_signaled + 1,
                cb_alloc,
                deferred_delete: vec![],
            },
        ));
        submit_result
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum DeferredDestroyObject {
    Image(ImageId),
    Buffer(BufferId),
    ImageView(vk::ImageView),
    Sampler(vk::Sampler),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    Semaphore(vk::Semaphore),
}

impl From<ImageId> for DeferredDestroyObject {
    fn from(image: ImageId) -> Self {
        DeferredDestroyObject::Image(image)
    }
}
impl From<BufferId> for DeferredDestroyObject {
    fn from(buffer: BufferId) -> Self {
        DeferredDestroyObject::Buffer(buffer)
    }
}
impl From<vk::ImageView> for DeferredDestroyObject {
    fn from(view: vk::ImageView) -> Self {
        DeferredDestroyObject::ImageView(view)
    }
}
impl From<vk::Sampler> for DeferredDestroyObject {
    fn from(sampler: vk::Sampler) -> Self {
        DeferredDestroyObject::Sampler(sampler)
    }
}
impl From<vk::PipelineLayout> for DeferredDestroyObject {
    fn from(layout: vk::PipelineLayout) -> Self {
        DeferredDestroyObject::PipelineLayout(layout)
    }
}
impl From<vk::Pipeline> for DeferredDestroyObject {
    fn from(pipeline: vk::Pipeline) -> Self {
        DeferredDestroyObject::Pipeline(pipeline)
    }
}
impl From<vk::Semaphore> for DeferredDestroyObject {
    fn from(semaphore: vk::Semaphore) -> Self {
        DeferredDestroyObject::Semaphore(semaphore)
    }
}

/// Deferred object deletion.
impl Queue {
    /// Performs a layout transition on an image.
    ///
    /// Equivalent to building a submission with a single `use_image`.
    pub unsafe fn transition_image(&mut self, image: ImageId, state: ResourceState) -> VkResult<()> {
        let mut s = self.build_submission();
        s.use_image(image, state);
        s.submit()
    }

    pub unsafe fn destroy_later(&mut self, object: impl Into<DeferredDestroyObject>) {
        let obj = object.into();
        let resource_id: Option<ResourceId> = match obj {
            DeferredDestroyObject::Image(image) => Some(image.into()),
            DeferredDestroyObject::Buffer(buffer) => Some(buffer.into()),
            _ => None,
        };
        if let Some(resource_id) = resource_id {
            // For image and buffers, check that they are synchronized on this queue
            let ownership = self
                .device
                .resources
                .borrow()
                .get(resource_id)
                .unwrap()
                .tracking
                .queue_ownership;
            assert_eq!(
                ownership,
                QueueOwnership::Exclusive(self.queue_index as u16),
                "destroy_later called on a resource that is not synchronized on this queue. Use acquire"
            );
        }
        self.current_frame.deferred_delete.push(obj);
    }

    fn cleanup(&mut self) {
        let completed = unsafe {
            self.device
                .device
                .get_semaphore_counter_value(self.timeline)
                .expect("get_semaphore_counter_value failed")
        };

        loop {
            let Some(oldest_frame) = self.submitted.front() else {
                break;
            };
            if oldest_frame.timeline_value > completed {
                break;
            }
            debug!(
                "cleaning up frame {} on queue {}",
                oldest_frame.timeline_value, self.queue_index
            );
            // reclaim resources of the oldest frame that has completed
            let frame = self.submitted.pop_front().unwrap();
            // deferred deletion
            unsafe {
                for obj in frame.deferred_delete {
                    match obj {
                        DeferredDestroyObject::ImageView(view) => {
                            self.device.device.destroy_image_view(view, None);
                        }
                        DeferredDestroyObject::Sampler(sampler) => {
                            self.device.device.destroy_sampler(sampler, None);
                        }
                        DeferredDestroyObject::PipelineLayout(layout) => {
                            self.device.device.destroy_pipeline_layout(layout, None);
                        }
                        DeferredDestroyObject::Pipeline(pipeline) => {
                            self.device.device.destroy_pipeline(pipeline, None);
                        }
                        DeferredDestroyObject::Semaphore(semaphore) => {
                            self.device.device.destroy_semaphore(semaphore, None);
                        }
                        DeferredDestroyObject::Image(image) => self.device.destroy_image(image),
                        DeferredDestroyObject::Buffer(buffer) => {
                            self.device.destroy_buffer(buffer);
                        }
                        _ => {
                            unimplemented!()
                        }
                    }
                }
            }

            // recycle command pool
            let mut cb_alloc = frame.cb_alloc;
            unsafe {
                cb_alloc.reset(&self.device.device);
            }
            self.free_cb_allocs.push(cb_alloc);
        }
    }
}
