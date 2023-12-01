use crate::{
    command_allocator::CommandBufferAllocator,
    device::{
        BufferId, Device, GroupId, ImageHandle, ImageId, ImageRegistrationInfo, OwnerQueue, ResourceAllocation,
        ResourceId, ResourceKind, ResourceRegistrationInfo, Swapchain, SwapchainImage,
    },
    is_write_access, vk,
};
use ash::prelude::VkResult;
use std::{collections::VecDeque, ffi::c_void, mem, ptr, time::Duration};
use tracing::debug;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around a Vulkan queue that tracks the use of resources.
pub struct Queue {
    device: Device,
    global_index: usize,
    family_index: u32,
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
    current: InFlightResources,
    submitted: VecDeque<InFlightResources>,
}

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

/// In-flight resources.
#[derive(Debug)]
struct InFlightResources {
    timestamp: u64,
    /// Command pool
    cb_alloc: CommandBufferAllocator,
    // Deferred deletion list
    //deferred_delete: Vec<DeferredDestroyObject>,
}

impl Queue {
    pub(crate) unsafe fn new(
        device: Device,
        global_index: usize,
        queue: vk::Queue,
        timeline: vk::Semaphore,
        family_index: u32,
        initial_timestamp: u64,
    ) -> Queue {
        let raw_device = &device.raw();
        let cb_alloc = CommandBufferAllocator::new(raw_device, family_index);

        Queue {
            device,
            global_index,
            queue,
            timeline,
            free_cb_allocs: vec![],
            last_signaled: 0,
            semaphores: vec![],
            family_index,
            current: InFlightResources {
                timestamp: initial_timestamp,
                cb_alloc,
            },
            submitted: Default::default(),
        }
    }

    /// Returns the device associated with this queue.
    pub fn device(&self) -> &Device {
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
                    .raw()
                    .create_semaphore(&create_info, None)
                    .expect("vkCreateSemaphore failed"),
            )
        }
    }

    /// Allocates a command buffer.
    pub fn allocate_command_buffer(&mut self) -> vk::CommandBuffer {
        self.current.cb_alloc.alloc(&self.device.raw())
    }

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(
        &mut self,
        swapchain: &Swapchain,
        timeout: Duration,
    ) -> Result<SwapchainImage, vk::Result> {
        let image_available = self.get_or_create_semaphore();
        let (image_index, _suboptimal) = match self.device.vk_khr_swapchain().acquire_next_image(
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
            handle: ImageHandle { id, vk: handle },
            index: image_index,
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
            p_image_indices: &swapchain_image.index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };
        let result = self.device.vk_khr_swapchain().queue_present(self.queue, &present_info);
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

pub struct Submission {
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

impl Submission {
    pub fn new() -> Submission {
        Submission {
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

    pub fn signal(&mut self, semaphore: UnsignaledSemaphore) -> SignaledSemaphore {
        self.signal_raw(SemaphoreSignal::Binary { semaphore: semaphore.0 });
        SignaledSemaphore(semaphore.0)
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
}

impl Queue {
    /// Given a resource use, and the current state of the resource, adds the necessary barriers
    /// and updates the known state of the resource.
    fn record_resource_use(&self, use_: &ResourceUse, barriers: &mut PipelineBarrierBuilder) {
        let queue_index = self.global_index;
        let timestamp = self.current.timestamp;
        let mut resources = self.device.inner.resources.borrow_mut();
        let resource = resources.get_mut(use_.id).expect("unknown resource");

        assert!(!resource.discarded, "used a discarded resource: {:?}", resource);
        assert!(
            resource.group.is_none(),
            "add_resource_dependency cannot be called on group-synced resources"
        );

        let writes_visible = match resource.owner {
            OwnerQueue::None => {
                // first use, acquire ownership
                resource.owner = OwnerQueue::Exclusive(queue_index);
                // no writes yet
                true
            }
            OwnerQueue::Exclusive(queue) => {
                assert_eq!(
                    queue, queue_index,
                    "resource is owned by another queue; use Submission::queue_release/Submission::queue_acquire to transfer ownership"
                );
                resource.visible.contains(use_.access_mask) || resource.visible.contains(vk::AccessFlags2::MEMORY_READ)
            }
            OwnerQueue::Concurrent(_queues) => {
                todo!("queue concurrent access")
            }
        };

        // We need to insert a memory dependency if:
        // - the image needs a layout transition
        // - the resource has writes that are not visible to the target access
        //
        // We need to insert an execution dependency if:
        // - we're writing to the resource (write-after-read hazard, write-after-write hazard)
        if !writes_visible || resource.layout != use_.initial_layout || is_write_access(use_.access_mask) {
            match resource.kind {
                ResourceKind::Buffer(ref buffer) => {
                    let barrier = barriers.get_or_create_buffer_barrier(buffer.handle);
                    barrier.buffer = buffer.handle;
                    barrier.src_stage_mask |= resource.stages;
                    barrier.dst_stage_mask |= use_.stage_mask;
                    barrier.src_access_mask |= resource.flush_mask;
                    barrier.dst_access_mask |= use_.access_mask;
                }
                ResourceKind::Image(ref image) => {
                    let barrier = barriers.get_or_create_image_barrier(image.handle);
                    barrier.image = image.handle;
                    barrier.src_stage_mask |= resource.stages;
                    barrier.dst_stage_mask |= use_.stage_mask;
                    barrier.src_access_mask |= resource.flush_mask;
                    barrier.dst_access_mask |= use_.access_mask;
                    barrier.old_layout = resource.layout;
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
                    resource.layout = use_.final_layout;
                }
            }
        }

        if is_write_access(use_.access_mask) {
            // we're writing to the resource, so reset visibility...
            resource.visible = vk::AccessFlags2::empty();
            // ... but signal that there is data to flush.
            resource.flush_mask |= use_.access_mask;
        } else {
            // This memory dependency makes all writes on the resource available, and
            // visible to the types specified in `access.access_mask`.
            // There's no write, so we don't need to flush anything.
            resource.flush_mask = vk::AccessFlags2::empty();
            resource.visible |= use_.access_mask;
        }

        // Update the resource stage mask
        resource.stages = use_.stage_mask;
        resource.timestamp = timestamp;
        resource.owner = OwnerQueue::Exclusive(queue_index);
    }

    pub unsafe fn submit(&mut self, mut submission: Submission) -> VkResult<()> {
        let mut barrier_builder = PipelineBarrierBuilder::default();

        // Build barriers and update resource states
        for use_ in submission.uses.iter() {
            self.record_resource_use(use_, &mut barrier_builder);
        }

        // Push a command buffer with the necessary barriers
        let barrier_cb = self.allocate_command_buffer();
        let device = self.device.raw();
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
        submission.cbs.insert(0, barrier_cb);

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        // setup semaphore signal operations
        for signal in submission.signals.iter() {
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
        for (_i, w) in submission.waits.iter().enumerate() {
            wait_semaphore_dst_stages.push(w.dst_stage);
            match w.kind {
                SemaphoreWaitKind::Binary {
                    semaphore,
                    transfer_ownership,
                } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(0);
                    if transfer_ownership {
                        self.semaphores.push(UnsignaledSemaphore(semaphore));
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
            command_buffer_count: submission.cbs.len() as u32,
            p_command_buffers: submission.cbs.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        unsafe { self.device.queue_submit(self.queue, &[submit_info], vk::Fence::null()) }
    }

    pub fn end_frame(&mut self) -> VkResult<()> {
        let submit_result = unsafe {
            let semaphores = [self.timeline];
            let values = [self.current.timestamp];
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

        self.last_signaled = self.current.timestamp;

        // Do cleanup first so that we can recycle command pools
        let completed_timestamp = self.cleanup();

        // Scan the list of resources that are owned by the queue and that have been discarded
        unsafe {
            // SAFETY: we waited for the queue to be finished with the resources
            self.device
                .cleanup_queue_resources(self.global_index, completed_timestamp);
        }

        // Setup resources for the next frame
        let cb_alloc = self.free_cb_allocs.pop().unwrap_or_else(|| {
            debug!("creating new command buffer allocator");
            // SAFETY: queue_family is valid
            unsafe { CommandBufferAllocator::new(self.device.raw(), self.family_index) }
        });

        self.submitted.push_front(mem::replace(
            &mut self.current,
            InFlightResources {
                timestamp: self.last_signaled + 1,
                cb_alloc,
            },
        ));

        self.device
            .set_current_queue_timestamp(self.global_index, self.current.timestamp);

        submit_result
    }
}

/// Deferred object deletion.
impl Queue {
    /// Performs a layout transition on an image.
    ///
    /// Equivalent to building a submission with a single `use_image`.
    pub unsafe fn transition_image(&mut self, image: ImageId, state: ResourceState) -> VkResult<()> {
        let mut s = Submission::new();
        s.use_image(image, state);
        self.submit(s)
    }

    fn cleanup(&mut self) -> u64 {
        let completed = unsafe {
            self.device
                .raw()
                .get_semaphore_counter_value(self.timeline)
                .expect("get_semaphore_counter_value failed")
        };

        loop {
            let Some(oldest_frame) = self.submitted.front() else {
                break;
            };
            if oldest_frame.timestamp > completed {
                break;
            }
            debug!(
                "cleaning up frame {} on queue {}",
                oldest_frame.timestamp, self.global_index
            );
            // reclaim resources of the oldest frame that has completed
            let frame = self.submitted.pop_front().unwrap();

            // recycle command pool
            let mut cb_alloc = frame.cb_alloc;
            unsafe {
                cb_alloc.reset(&self.device.raw());
            }
            self.free_cb_allocs.push(cb_alloc);
        }

        completed
    }
}
