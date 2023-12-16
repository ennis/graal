//! Queues & submissions
use std::{collections::VecDeque, ffi::c_void, mem, ptr, time::Duration};

use ash::prelude::VkResult;
use tracing::debug;

use crate::{
    device::{
        ensure_memory_dependency, Device, ImageRegistrationInfo, OwnerQueue, PipelineBarrierBuilder,
        ResourceAllocation, ResourceHandle, ResourceKind, ResourceRegistrationInfo, Swapchain, SwapchainImage,
    },
    vk, CommandBuffer, Image, ImageType, ImageUsage, ResourceState, Size3D,
};

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

/// In-flight resources.
#[derive(Debug)]
struct InFlightResources {
    timestamp: u64,
    /// Command pool
    cb_alloc: CommandBufferAllocator,
}

/// A wrapper around a signaled binary semaphore.
#[derive(Debug)]
pub struct SignaledSemaphore(pub(crate) vk::Semaphore);

/// A wrapper around an unsignaled binary semaphore.
#[derive(Debug)]
pub struct UnsignaledSemaphore(pub(crate) vk::Semaphore);

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

impl Queue {
    pub(super) unsafe fn new(
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
    pub fn create_command_buffer(&mut self) -> CommandBuffer {
        let cb = self.current.cb_alloc.alloc(&self.device.raw());
        unsafe {
            self.device
                .raw()
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
        }
        CommandBuffer::new(self.device(), cb)
    }

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(
        &mut self,
        swapchain: &Swapchain,
        timeout: Duration,
    ) -> Result<SwapchainImage, vk::Result> {
        let image_available = self.get_or_create_semaphore();
        let (image_index, _suboptimal) = match self.device.khr_swapchain().acquire_next_image(
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
            image: Image {
                device: self.device.clone(),
                id,
                handle,
                usage: ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
                type_: ImageType::Image2D,
                format: swapchain.format.format,
                size: Size3D {
                    width: swapchain.width,
                    height: swapchain.height,
                    depth: 1,
                },
            },
            index: image_index,
        })
    }

    pub unsafe fn present(&mut self, swapchain_image: &SwapchainImage) -> VkResult<bool> {
        let mut cb = self.create_command_buffer();
        cb.use_image(&swapchain_image.image, ResourceState::PRESENT);

        // FIXME: this pushes two command buffers: one for the PipelineBarrier, and another, empty one
        // There should be another mechanism for transitioning resources outside of command buffers
        self.submit([cb])?;
        let render_finished = self.signal();

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
        let result = self.device.khr_swapchain().queue_present(self.queue, &present_info);
        // we signalled and waited on the semaphore, consider it consumed
        self.semaphores.push(UnsignaledSemaphore(render_finished.0));
        result
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

    /// Waits for and consumes a signaled binary semaphore.
    pub fn wait(&mut self, semaphore: SignaledSemaphore, dst_stage: vk::PipelineStageFlags) {
        unsafe {
            self.submit_raw(
                [],
                &[SemaphoreWait {
                    kind: SemaphoreWaitKind::Binary {
                        semaphore: semaphore.0,
                        transfer_ownership: true,
                    },
                    dst_stage,
                }],
                &[],
            )
            .expect("queue wait failed");
        }
    }

    /// Signals a binary semaphore.
    pub fn signal(&mut self) -> SignaledSemaphore {
        let semaphore = self.get_or_create_semaphore();
        unsafe {
            self.submit_raw([], &[], &[SemaphoreSignal::Binary { semaphore: semaphore.0 }])
                .expect("queue signal failed");
        }
        SignaledSemaphore(semaphore.0)
    }

    /// Submits command buffers.
    pub fn submit(&mut self, cmd_bufs: impl IntoIterator<Item = CommandBuffer>) -> VkResult<()> {
        unsafe { self.submit_raw(cmd_bufs, &[], &[]) }
    }

    unsafe fn submit_raw(
        &mut self,
        cmd_bufs: impl IntoIterator<Item = CommandBuffer>,
        waits: &[SemaphoreWait],
        signals: &[SemaphoreSignal],
    ) -> VkResult<()> {
        // Insert pipeline barriers between command buffers
        let mut resources = self.device.inner.resources.borrow_mut();
        let mut command_buffers = vec![];

        for cb in cmd_bufs.into_iter() {
            let cb: CommandBuffer = cb;
            // finish the command buffer
            self.device.raw().end_command_buffer(cb.command_buffer)?;
            let mut barrier_builder = PipelineBarrierBuilder::default();
            for (id, use_) in cb.initial_uses.iter() {
                let resource = &mut resources[*id];

                // ensure that it can be used on this queue
                match resource.owner {
                    OwnerQueue::None => {
                        resource.owner = OwnerQueue::Exclusive(self.global_index);
                    }
                    OwnerQueue::Exclusive(q) => {
                        assert_eq!(q, self.global_index, "resource is already owned by another queue");
                    }
                    _ => {
                        panic!("concurrent ownership is not supported")
                    }
                }

                let handle = match resource.kind {
                    ResourceKind::Image(ref image) => ResourceHandle::Image(image.handle),
                    ResourceKind::Buffer(ref buffer) => ResourceHandle::Buffer(buffer.handle),
                };
                ensure_memory_dependency(&mut barrier_builder, handle, &mut resource.dep_state, use_);
            }
            for (id, final_state) in cb.final_states.iter() {
                resources[*id].dep_state = *final_state;
                resources[*id].timestamp = self.current.timestamp;
            }
            if !barrier_builder.is_empty() {
                let barrier_cb = self.current.cb_alloc.alloc(&self.device.raw());
                record_barrier_command_buffer(&self.device, barrier_cb, barrier_builder);
                command_buffers.push(barrier_cb);
            }
            command_buffers.push(cb.command_buffer);

            // we can drop refs now, the referenced resources lifetime is extended by setting the timestamp.
            // NOTE: the call does nothing, but it's clearer to call it explicitly
            drop(cb.refs);
        }

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        // setup semaphore signal operations
        for signal in signals.iter() {
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
                    fence: _,
                    value: _,
                } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup semaphore wait operations
        for (_i, w) in waits.iter().enumerate() {
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
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
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

/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
struct CommandBufferAllocator {
    queue_family: u32,
    command_pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    used: Vec<vk::CommandBuffer>,
}

impl CommandBufferAllocator {
    unsafe fn new(device: &ash::Device, queue_family_index: u32) -> CommandBufferAllocator {
        // create a new one
        let create_info = vk::CommandPoolCreateInfo {
            flags: vk::CommandPoolCreateFlags::TRANSIENT,
            queue_family_index,
            ..Default::default()
        };
        let command_pool = device
            .create_command_pool(&create_info, None)
            .expect("failed to create a command pool");

        CommandBufferAllocator {
            queue_family: queue_family_index,
            command_pool,
            free: vec![],
            used: vec![],
        }
    }

    fn alloc(&mut self, device: &ash::Device) -> vk::CommandBuffer {
        let cb = self.free.pop().unwrap_or_else(|| unsafe {
            let allocate_info = vk::CommandBufferAllocateInfo {
                command_pool: self.command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let buffers = device
                .allocate_command_buffers(&allocate_info)
                .expect("failed to allocate command buffers");
            buffers[0]
        });
        self.used.push(cb);
        cb
    }

    unsafe fn reset(&mut self, device: &ash::Device) {
        device
            .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
            .unwrap();
        self.free.append(&mut self.used)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
impl Submission {
    pub fn new() -> Submission {
        Submission {
            name: None,
            signals: vec![],
            waits: vec![],
            cbs: vec![],
        }
    }

    pub fn set_name(&mut self, name: impl Into<String>) {
        self.name = Some(name.into());
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

    /*
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
    }*/

    pub fn push_command_buffer(&mut self, cb: CommandBuffer) {
        self.cbs.push(cb);
    }
}
*/

fn record_barrier_command_buffer(device: &Device, command_buffer: vk::CommandBuffer, barriers: PipelineBarrierBuilder) {
    let device = device.raw();
    unsafe {
        device
            .begin_command_buffer(
                command_buffer,
                &vk::CommandBufferBeginInfo {
                    flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                    ..Default::default()
                },
            )
            .unwrap();
        device.cmd_pipeline_barrier2(
            command_buffer,
            &vk::DependencyInfo {
                dependency_flags: Default::default(),
                memory_barrier_count: 0,
                p_memory_barriers: ptr::null(),
                buffer_memory_barrier_count: barriers.buffer_barriers.len() as u32,
                p_buffer_memory_barriers: barriers.buffer_barriers.as_ptr(),
                image_memory_barrier_count: barriers.image_barriers.len() as u32,
                p_image_memory_barriers: barriers.image_barriers.as_ptr(),
                ..Default::default()
            },
        );
        device.end_command_buffer(command_buffer).unwrap();
    }
}
