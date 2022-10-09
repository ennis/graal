//! Code related to the recording of command buffers submitted to GPU queues (`vkQueueSubmit`, presentation).

use crate::{
    context::{
        FrameInFlight, PassEvaluationCallback, PresentOperationResult, SemaphoreSignal, SemaphoreSignalKind,
        SemaphoreWait, SemaphoreWaitKind, SubmitResult,
    },
    serial::{DeviceProgress, SubmissionNumber},
    vk, Context, Frame, FrameNumber, MAX_QUEUES,
};
use std::{
    collections::VecDeque,
    ffi::{c_void, CString},
    ops::Deref,
    ptr,
    time::Duration,
};
use tracing::trace_span;

////////////////////////////////////////////////////////////////////////////////////////////////////
//
////////////////////////////////////////////////////////////////////////////////////////////////////

/// Context passed to the command callbacks.
/// FIXME: not sure we need the context? maybe just directly pass a reference to the device
pub struct RecordingContext<'a> {
    pub context: &'a Context,
}

impl<'a> Deref for RecordingContext<'a> {
    type Target = Context;

    fn deref(&self) -> &Context {
        self.context
    }
}

/// Allocates command buffers in a `vk::CommandPool` and allows re-use of freed command buffers.
#[derive(Debug)]
pub(crate) struct CommandAllocator {
    queue_family: u32,
    command_pool: vk::CommandPool,
    free: Vec<vk::CommandBuffer>,
    used: Vec<vk::CommandBuffer>,
}

impl CommandAllocator {
    fn allocate_command_buffer(&mut self, device: &ash::Device) -> vk::CommandBuffer {
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

    fn reset(&mut self, device: &ash::Device) {
        unsafe {
            device
                .reset_command_pool(self.command_pool, vk::CommandPoolResetFlags::empty())
                .unwrap();
        }
        self.free.append(&mut self.used)
    }
}

/// Represents a queue submission (a call to vkQueueSubmit or vkQueuePresent)
struct CommandBatch {
    wait_serials: DeviceProgress,
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    signal_snn: SubmissionNumber,
    external_semaphore_waits: Vec<SemaphoreWait>,     // TODO arrayvec
    external_semaphore_signals: Vec<SemaphoreSignal>, // TODO arrayvec
    command_buffers: Vec<vk::CommandBuffer>,
}

impl CommandBatch {
    fn new() -> CommandBatch {
        CommandBatch {
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            signal_snn: Default::default(),
            external_semaphore_waits: vec![],
            external_semaphore_signals: vec![],
            command_buffers: Vec::new(),
        }
    }

    /// A submission batch is considered empty if there are no command buffers to submit and
    /// nothing to signal.
    /// Even if there are no command buffers, a batch may still submitted if the batch defines
    /// a wait and a signal operation, as a way of sequencing a timeline semaphore wait and a binary semaphore signal, for instance.
    fn is_empty(&self) -> bool {
        self.command_buffers.is_empty() && !self.signal_snn.is_valid() && self.external_semaphore_signals.is_empty()
    }

    fn reset(&mut self) {
        self.wait_serials = Default::default();
        self.wait_dst_stages = Default::default();
        self.wait_serials = Default::default();
        self.signal_snn = Default::default();
        self.external_semaphore_waits.clear();
        self.external_semaphore_signals.clear();
        self.command_buffers.clear();
    }
}

impl Default for CommandBatch {
    fn default() -> Self {
        CommandBatch::new()
    }
}

/// States related to queue operations and the tracking of frames in flight.
pub(super) struct SubmitState {
    /// Array containing the last submitted pass serials for each queue
    last_signalled_serials: DeviceProgress,

    /// Pool of recycled command pools.
    command_pools: Vec<CommandAllocator>,

    /// Frames that are currently executing on the GPU.
    in_flight: VecDeque<FrameInFlight>,

    /// The last completed pass serials for each queue
    completed_serials: DeviceProgress,

    /// Number of completed frames
    completed_frame_count: u64,
}

impl SubmitState {
    pub(super) fn new() -> SubmitState {
        SubmitState {
            last_signalled_serials: Default::default(),
            command_pools: vec![],
            in_flight: Default::default(),
            completed_serials: Default::default(),
            completed_frame_count: 0,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// Context
////////////////////////////////////////////////////////////////////////////////////////////////////

impl Context {
    fn submit_command_batch(&mut self, queue: usize, batch: &CommandBatch, used_semaphores: &mut Vec<vk::Semaphore>) {
        if batch.is_empty() {
            return;
        }

        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();

        // end command buffers
        for &cb in batch.command_buffers.iter() {
            unsafe { self.device.device.end_command_buffer(cb).unwrap() }
        }

        // setup queue timeline signals if necessary
        if batch.signal_snn.serial() > 0 {
            signal_semaphores.push(self.device.queue_timeline(queue));
            signal_semaphore_values.push(batch.signal_snn.serial());
            self.submit_state.last_signalled_serials[queue] = batch.signal_snn.serial();
        }

        // external semaphore signals
        for signal in batch.external_semaphore_signals.iter() {
            signal_semaphores.push(signal.semaphore);
            match signal.signal_kind {
                SemaphoreSignalKind::Binary => {
                    signal_semaphore_values.push(0);
                }
                SemaphoreSignalKind::Timeline(value) => {
                    signal_semaphore_values.push(value);
                }
            }
        }

        // setup queue timeline wait values & stage masks (see VkTimelineSemaphoreSubmitInfo below).
        for (i, &w) in batch.wait_serials.iter().enumerate() {
            if w != 0 {
                wait_semaphores.push(self.device.queue_timeline(i));
                wait_semaphore_values.push(w);
                wait_semaphore_dst_stages.push(batch.wait_dst_stages[i]);
            }
        }

        // external semaphore waits
        for wait in batch.external_semaphore_waits.iter() {
            wait_semaphores.push(wait.semaphore);
            wait_semaphore_dst_stages.push(wait.dst_stage);
            match wait.wait_kind {
                SemaphoreWaitKind::Binary => {
                    wait_semaphore_values.push(0);
                }
                SemaphoreWaitKind::Timeline(value) => {
                    wait_semaphore_values.push(value);
                }
            }

            // Every semaphore that is waited on (except queue timelines) is put in `used_semaphores`.
            // We don't immediately allow re-use of the semaphore, since there's
            // no guarantee that the next signal of the semaphore will happen after the wait that
            // we just queued. For instance, it could be signalled on another queue.
            used_semaphores.push(wait.semaphore);
        }

        let mut timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
            wait_semaphore_value_count: wait_semaphore_values.len() as u32,
            p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
            signal_semaphore_value_count: signal_semaphore_values.len() as u32,
            p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
            ..Default::default()
        };

        let submit_info = vk::SubmitInfo {
            p_next: &mut timeline_submit_info as *mut _ as *mut c_void,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_semaphore_dst_stages.as_ptr(),
            command_buffer_count: batch.command_buffers.len() as u32,
            p_command_buffers: batch.command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        let vk_queue = self.device.queues_info.queues[queue as usize];
        unsafe {
            self.device
                .device
                .queue_submit(vk_queue, &[submit_info], vk::Fence::null())
                .expect("queue submission failed");
        }
    }

    /// Creates a command pool for the given queue and wraps it in a `CommandAllocator`.
    fn create_command_pool(&mut self, queue_index: usize) -> CommandAllocator {
        // Command pools are tied to a queue family
        let queue_family = self.device.queues_info.families[queue_index];

        // Try to find a free pool with of the correct queue family in the list of recycled command pools.
        // If we find one, remove it from the list and return it. Otherwise create a new one.
        if let Some(pos) = self
            .submit_state
            .command_pools
            .iter()
            .position(|cmd_pool| cmd_pool.queue_family == queue_family)
        {
            // found one, remove it and return it
            self.submit_state.command_pools.swap_remove(pos)
        } else {
            // create a new one
            let create_info = vk::CommandPoolCreateInfo {
                flags: vk::CommandPoolCreateFlags::TRANSIENT,
                queue_family_index: queue_family,
                ..Default::default()
            };
            let command_pool = unsafe {
                self.device
                    .device
                    .create_command_pool(&create_info, None)
                    .expect("failed to create a command pool")
            };
            CommandAllocator {
                queue_family,
                command_pool,
                free: vec![],
                used: vec![],
            }
        }
    }

    /// Builds the command buffers for a frame and submits them to the queues.
    pub(super) fn submit_to_device_queues<UserContext>(
        &mut self,
        user_context: &mut UserContext,
        mut frame: Frame<UserContext>,
    ) -> SubmitResult {
        let _span = trace_span!("Submit frame").entered();

        let mut present_results = vec![];
        // current submission batches per queue
        let mut cmd_batches: [CommandBatch; MAX_QUEUES] = Default::default();
        // one command pool per queue (might not be necessary if the queues belong to the same family,
        // but they usually don't)
        let mut cmd_pools: [Option<CommandAllocator>; MAX_QUEUES] = Default::default();
        // all binary semaphores waited on
        let mut used_semaphores = Vec::new();

        let mut first_pass_of_queue = [true; MAX_QUEUES];

        for p in frame.passes.iter_mut() {
            // queue index
            let q = p.snn.queue();

            let wait_serials = if first_pass_of_queue[q] && frame.initial_wait > self.submit_state.completed_serials {
                p.wait_serials.join(frame.initial_wait)
            } else {
                p.wait_serials
            };

            first_pass_of_queue[q] = false;

            // we need to wait if we have a binary semaphore, or if it's the first pass in this queue
            // and the user specified an initial wait before starting the frame.
            let needs_semaphore_wait =
                wait_serials > self.submit_state.completed_serials || !p.external_semaphore_waits.is_empty();

            if needs_semaphore_wait {
                // the pass needs a semaphore wait, so it needs a separate batch
                // close the batches on all queues that the pass waits on
                for i in 0..MAX_QUEUES {
                    if !cmd_batches[i].is_empty() && (i == q || p.wait_serials[i] != 0) {
                        self.submit_command_batch(i, &cmd_batches[i], &mut used_semaphores);
                        cmd_batches[i].reset();
                    }
                }
            }

            let batch: &mut CommandBatch = &mut cmd_batches[q as usize];

            if needs_semaphore_wait {
                batch.wait_serials = wait_serials;
                batch.wait_dst_stages = p.wait_dst_stages; // FIXME are those OK?
                                                           // the current batch shouldn't have any pending waits because we just flushed them
                batch.external_semaphore_waits = p.external_semaphore_waits.clone();
            }

            // ensure that a command pool has been allocated for the queue
            let command_pool: &mut CommandAllocator =
                cmd_pools[q as usize].get_or_insert_with(|| self.create_command_pool(p.snn.queue()));
            // append to the last command buffer of the batch, otherwise create another one

            if batch.command_buffers.is_empty() {
                let cb = command_pool.allocate_command_buffer(&self.device.device);
                let begin_info = vk::CommandBufferBeginInfo { ..Default::default() };
                unsafe {
                    // TODO safety
                    self.device.device.begin_command_buffer(cb, &begin_info).unwrap();
                }
                batch.command_buffers.push(cb);
            };

            let cb = *batch.command_buffers.last().unwrap();

            // cb is a command buffer in the recording state
            let marker_name = CString::new(p.name.as_str()).unwrap();
            unsafe {
                self.device.vk_ext_debug_utils.cmd_begin_debug_utils_label(
                    cb,
                    &vk::DebugUtilsLabelEXT {
                        p_label_name: marker_name.as_ptr(),
                        color: [0.0; 4],
                        ..Default::default()
                    },
                );
            }

            // emit barriers if needed
            if p.src_stage_mask != vk::PipelineStageFlags::TOP_OF_PIPE
                || p.dst_stage_mask != vk::PipelineStageFlags::BOTTOM_OF_PIPE
                || !p.buffer_memory_barriers.is_empty()
                || !p.image_memory_barriers.is_empty()
            {
                let src_stage_mask = if p.src_stage_mask.is_empty() {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    p.src_stage_mask
                };
                let dst_stage_mask = if p.dst_stage_mask.is_empty() {
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE
                } else {
                    p.dst_stage_mask
                };
                unsafe {
                    let global_memory_barrier = if let Some(mb) = p.global_memory_barrier.as_ref() {
                        std::slice::from_ref(mb)
                    } else {
                        &[]
                    };
                    // TODO safety
                    self.device.device.cmd_pipeline_barrier(
                        cb,
                        src_stage_mask,
                        dst_stage_mask,
                        Default::default(),
                        global_memory_barrier,
                        &p.buffer_memory_barriers,
                        &p.image_memory_barriers,
                    )
                }
            }

            match p.eval.take() {
                Some(PassEvaluationCallback::CommandBuffer(record_fn)) => {
                    // perform a command-buffer level operation
                    let mut cctx = RecordingContext { context: self };
                    record_fn(&mut cctx, user_context, cb);
                    // update signalled serial for the batch (pass serials are guaranteed to be increasing)
                    //eprintln!("PassEvaluationCallback::CommandBuffer, signal_snn={:?}", p.snn);
                    batch.signal_snn = p.snn;
                }
                Some(PassEvaluationCallback::Queue(submit_fn)) => {
                    // perform a queue-level operation:
                    // this terminates the current batch
                    self.submit_command_batch(q, batch, &mut used_semaphores);
                    batch.reset();
                    // call the handler
                    let queue = self.device.queues_info.queues[q as usize];
                    let mut cctx = RecordingContext { context: self };
                    submit_fn(&mut cctx, user_context, queue);
                    //eprintln!("PassEvaluationCallback::Queue, signal_snn={:?}", p.snn);
                    batch.signal_snn = p.snn;
                }
                Some(PassEvaluationCallback::Present { swapchain, image_index }) => {
                    // present operation:
                    // modify the current batch to signal a binary semaphore and close it
                    let render_finished_semaphore = self.create_semaphore();
                    // FIXME if the swapchain image is last modified by another queue,
                    // then this batch contains no commands, only one timeline wait
                    // and one binary semaphore signal.
                    // This could be optimized by signalling a binary semaphore on the pass
                    // that modifies the swapchain image, but at the cost of code complexity
                    // and maintainability.
                    // Eventually, the presentation engine might support timeline semaphores
                    // directly, which will make this entire problem vanish.
                    batch.external_semaphore_signals.push(SemaphoreSignal {
                        semaphore: render_finished_semaphore,
                        signal_kind: SemaphoreSignalKind::Binary,
                    });
                    self.submit_command_batch(q, batch, &mut used_semaphores);
                    batch.reset();
                    // build present info that waits on the batch that was just submitted
                    let present_info = vk::PresentInfoKHR {
                        wait_semaphore_count: 1,
                        p_wait_semaphores: &render_finished_semaphore,
                        swapchain_count: 1,
                        p_swapchains: &swapchain,
                        p_image_indices: &image_index,
                        p_results: ptr::null_mut(),
                        ..Default::default()
                    };
                    unsafe {
                        // TODO safety
                        let queue = self.device.queues_info.queues[q as usize];
                        let result = self.device.vk_khr_swapchain.queue_present(queue, &present_info);

                        match result {
                            Ok(suboptimal) => {
                                // nothing
                                if suboptimal {
                                    present_results.push(PresentOperationResult {
                                        swapchain,
                                        result: vk::Result::SUBOPTIMAL_KHR,
                                    })
                                }
                            }
                            Err(err @ vk::Result::ERROR_OUT_OF_DATE_KHR) => {
                                // The docs say:
                                //
                                //      However, if the presentation request is rejected by the presentation engine with an error VK_ERROR_OUT_OF_DATE_KHR, VK_ERROR_FULL_SCREEN_EXCLUSIVE_MODE_LOST_EXT,
                                //      or VK_ERROR_SURFACE_LOST_KHR, the set of queue operations are still considered to be enqueued and thus any semaphore wait operation specified in VkPresentInfoKHR
                                //      will execute when the corresponding queue operation is complete.
                                //
                                // So we can just report the error and continue since the semaphores
                                // will be left in a consistent state.
                                present_results.push(PresentOperationResult { swapchain, result: err });
                            }
                            Err(err) => {
                                // TODO handle more complicated errors
                                panic!("vkQueuePresent failed: {}", err)
                            }
                        }
                    }
                    // we signalled and waited on the semaphore, consider it consumed
                    used_semaphores.push(render_finished_semaphore);
                }
                None => {}
            }

            unsafe {
                // FIXME this can end up in a different command buffer
                self.device.vk_ext_debug_utils.cmd_end_debug_utils_label(cb);
            }

            // the pass needs a semaphore signal: this terminates the batch on the queue
            // FIXME what does this do if the pass is a queue-level operation?
            if p.signal_queue_timelines || !p.external_semaphore_signals.is_empty() {
                /*eprintln!(
                    "submit_command_batch (queue = {}, reason: signal_queue_timelines={})",
                    q, p.signal_queue_timelines
                );*/
                self.submit_command_batch(q, batch, &mut used_semaphores);
                batch.reset();
            }
        }

        // close unfinished batches
        for batch in cmd_batches.iter() {
            //eprintln!("submit_command_batch (finishing)");
            self.submit_command_batch(batch.signal_snn.queue(), batch, &mut used_semaphores)
        }

        let command_pools = cmd_pools.iter_mut().filter_map(|cmd_pool| cmd_pool.take()).collect();

        // Add this frame to the list of "frames in flight": frames that might be executing on the device.
        // When this frame is completed, all resources of the frame will be automatically recycled.
        // This includes:
        // - device memory blocks for transient allocations
        // - command buffers (in command pools)
        // - image views
        // - framebuffers
        // - descriptor sets
        self.submit_state.in_flight.push_back(FrameInFlight {
            signalled_serials: self.submit_state.last_signalled_serials,
            //transient_allocations,
            command_pools,
            semaphores: used_semaphores,
        });

        SubmitResult {
            progress: self.submit_state.last_signalled_serials,
            present_results,
        }
    }

    /// Recycles command pools returned by `submit_frame`.
    pub(super) fn recycle_command_pools(&mut self, mut allocators: Vec<CommandAllocator>) {
        for a in allocators.iter_mut() {
            a.reset(&self.device.device)
        }
        self.submit_state.command_pools.append(&mut allocators);
    }

    /// Returns whether the given frame, identified by its serial, has completed execution.
    pub fn is_frame_completed(&self, serial: FrameNumber) -> bool {
        self.submit_state.completed_frame_count >= serial.0
    }

    /// Reclaims the resources of the frames that have finished, and optionally waits for the specified progress.
    ///
    /// # Arguments
    /// * wait_target progress to wait for before reclaiming finished frames
    /// * wait_target_timeout timeout for the wait, ignored if `wait_target` is `None`.
    ///
    pub(super) fn retire_completed_frames(
        &mut self,
        wait_target: Option<DeviceProgress>,
        wait_target_timeout: Duration,
    ) -> Result<(), vk::Result> {
        let _span = trace_span!("Frame pacing").entered();

        if let Some(ref wait_target) = wait_target {
            self.device.wait(wait_target, wait_target_timeout)?;
        }

        // retire all finished frames and reclaim their resources, starting from the oldest
        let current_progress = self.device.current_progress()?;
        let mut should_cleanup_resources = false;
        loop {
            if self.submit_state.in_flight.is_empty() {
                break;
            }

            let oldest_frame = self.submit_state.in_flight.front().unwrap();

            // break if the frame isn't finished yet (the device hasn't yet reached its progress value)
            if oldest_frame.signalled_serials > current_progress {
                break;
            }

            // the frame has finished, do cleanup
            should_cleanup_resources = true;
            let oldest_frame = self.submit_state.in_flight.pop_front().unwrap();

            // update completed serials
            // we just waited on those serials, so we know they are completed
            self.submit_state.completed_serials = oldest_frame.signalled_serials;

            // Recycle the command pools allocated for the frame. The allocated command buffers
            // can then be reused for future submissions.
            self.recycle_command_pools(oldest_frame.command_pools);

            // Recycle the semaphores. They are guaranteed to be unsignalled since the frame must have
            // waited on them.
            unsafe {
                self.recycle_semaphores(oldest_frame.semaphores);
            }

            // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
            /*// free transient allocations
            for alloc in f.transient_allocations {
                trace!(?alloc, "free_memory");
                self.device.allocator.borrow_mut().free(alloc).unwrap();
            }*/

            // bump completed frame count
            self.submit_state.completed_frame_count += 1;
        }

        // if at least one frame has been retired, reclaim/free the resources that may have expired as a result
        if should_cleanup_resources {
            unsafe {
                // SAFETY: we just waited for the passes to finish
                self.device.cleanup_resources(
                    self.submit_state.completed_serials,
                    FrameNumber(self.submit_state.completed_frame_count),
                )
            }
        }

        Ok(())
    }
}
