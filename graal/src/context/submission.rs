use crate::{
    context::{
        pass::{Pass, PassCommands},
        QueueSerialNumbers, SubmissionNumber,
    },
    vk, Context, DescriptorSetInterface, MAX_QUEUES,
};

use crate::context::{descriptor::DescriptorSet, SEMAPHORE_WAIT_TIMEOUT_NS};
use ash::version::{DeviceV1_0, DeviceV1_2};
use std::{
    ffi::{c_void, CString},
    ops::{Deref, DerefMut},
    ptr,
};

use tracing::trace_span;

/// Resources created at submission time.
struct SubmissionTimeResources {
    /// Image views created in this batch
    image_views: Vec<vk::ImageView>,
    /// Framebuffers created in this batch
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl SubmissionTimeResources {
    fn new() -> SubmissionTimeResources {
        SubmissionTimeResources {
            image_views: vec![],
            framebuffers: vec![],
            descriptor_sets: vec![],
        }
    }
}

/// Context passed to the command callbacks.
pub struct CommandContext<'a> {
    context: &'a mut Context,
    resources: &'a mut SubmissionTimeResources,
}

impl<'a> Deref for CommandContext<'a> {
    type Target = Context;

    fn deref(&self) -> &Context {
        self.context
    }
}

impl<'a> DerefMut for CommandContext<'a> {
    fn deref_mut(&mut self) -> &mut Context {
        self.context
    }
}

impl<'a> CommandContext<'a> {
    /// Creates a descriptor set from an instance of a DescriptorSetInterface type.
    ///
    /// The returned descriptor set lives only for the duration of this pass.
    pub unsafe fn create_descriptor_set<T: DescriptorSetInterface>(
        &mut self,
        descriptors: &T,
    ) -> vk::DescriptorSet {
        let set = self.context.create_descriptor_set(descriptors);
        self.resources.descriptor_sets.push(set);
        set.set
    }

    /// Creates a transient image view.
    ///
    /// Preconditions:
    /// - the provided `vk::ImageViewCreateInfo` must be valid.
    ///
    /// The returned `vk::ImageView` can only be used in this current batch and is automatically
    /// reclaimed after that.
    pub unsafe fn create_image_view(
        &mut self,
        create_info: &vk::ImageViewCreateInfo,
    ) -> vk::ImageView {
        let handle = self
            .context
            .device
            .device
            .create_image_view(&create_info, None)
            .unwrap();
        self.resources.image_views.push(handle);
        handle
    }

    /// Creates a transient framebuffer.
    ///
    /// Preconditions:
    /// - render_pass must be a valid render pass object
    /// - attachment must contain only valid image views
    ///
    /// The returned framebuffer lives only for the duration of this batch.
    pub unsafe fn create_framebuffer(
        &mut self,
        width: u32,
        height: u32,
        layers: u32,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
    ) -> vk::Framebuffer {
        let framebuffer_create_info = vk::FramebufferCreateInfo {
            flags: Default::default(),
            render_pass,
            attachment_count: attachments.len() as u32,
            p_attachments: attachments.as_ptr(),
            width,
            height,
            layers,
            ..Default::default()
        };

        let handle = self
            .context
            .device
            .device
            .create_framebuffer(&framebuffer_create_info, None)
            .unwrap();
        self.resources.framebuffers.push(handle);
        handle
    }
}

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
    wait_serials: QueueSerialNumbers,
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    signal_snn: SubmissionNumber,
    wait_binary_semaphores: Vec<vk::Semaphore>, // TODO arrayvec
    signal_binary_semaphores: Vec<vk::Semaphore>, // TODO arrayvec
    command_buffers: Vec<vk::CommandBuffer>,
}

impl CommandBatch {
    fn new() -> CommandBatch {
        CommandBatch {
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            signal_snn: Default::default(),
            wait_binary_semaphores: vec![],
            signal_binary_semaphores: vec![],
            command_buffers: Vec::new(),
        }
    }

    /// A submission batch is considered empty if there are no command buffers to submit and
    /// nothing to signal.
    /// Even if there are no command buffers, a batch may still submitted if the batch defines
    /// a wait and a signal operation, as a way of sequencing a timeline semaphore wait and a binary semaphore signal, for instance.
    fn is_empty(&self) -> bool {
        self.command_buffers.is_empty()
            && !self.signal_snn.is_valid()
            && self.signal_binary_semaphores.is_empty()
    }

    fn reset(&mut self) {
        self.wait_serials = Default::default();
        self.wait_dst_stages = Default::default();
        self.wait_serials = Default::default();
        self.signal_snn = Default::default();
        self.wait_binary_semaphores.clear();
        self.signal_binary_semaphores.clear();
        self.command_buffers.clear();
    }
}

impl Default for CommandBatch {
    fn default() -> Self {
        CommandBatch::new()
    }
}

/// Represents the result of a successful frame submission operation.
pub(crate) struct FrameSubmissionResult {
    /// The command pools used in the batch
    pub(crate) command_pools: Vec<CommandAllocator>,
    /// The last signalled serial numbers for each queue. They correspond to the last submitted passes
    /// in each queue.
    pub(crate) signalled_serials: QueueSerialNumbers,
    /// All semaphores used in this batch. When the batch is complete, those
    /// semaphores are guaranteed to be in the unsignalled state.
    pub(crate) semaphores: Vec<vk::Semaphore>,
    /// Image views created during command buffer creation.
    pub(crate) image_views: Vec<vk::ImageView>,
    /// Framebuffers created during command buffer creation.
    pub(crate) framebuffers: Vec<vk::Framebuffer>,
    /// Descriptor sets created during command buffer creation.
    pub(crate) descriptor_sets: Vec<DescriptorSet>,
}

impl Context {
    pub(crate) fn wait(&mut self, serials: &QueueSerialNumbers) {
        let _span = trace_span!("wait", ?serials);

        let wait_info = vk::SemaphoreWaitInfo {
            semaphore_count: self.timelines.len() as u32,
            p_semaphores: self.timelines.as_ptr(),
            p_values: serials.0.as_ptr(),
            ..Default::default()
        };
        unsafe {
            self.device
                .device
                .wait_semaphores(&wait_info, SEMAPHORE_WAIT_TIMEOUT_NS)
                .expect("error waiting for batch");
        }
    }

    fn submit_command_batch(
        &mut self,
        q: usize,
        batch: &CommandBatch,
        used_semaphores: &mut Vec<vk::Semaphore>,
    ) {
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

        // setup timeline signal if necessary
        if batch.signal_snn.serial() > 0 {
            signal_semaphores.push(self.timelines[q as usize]);
            signal_semaphore_values.push(batch.signal_snn.serial());
            self.last_signalled_serials[q] = batch.signal_snn.serial();
        }

        // binary semaphore signals
        for &s in batch.signal_binary_semaphores.iter() {
            signal_semaphores.push(s);
            signal_semaphore_values.push(0);
        }

        // setup timeline waits
        for (i, &w) in batch.wait_serials.iter().enumerate() {
            if w != 0 {
                wait_semaphores.push(self.timelines[i]);
                wait_semaphore_values.push(w);
                wait_semaphore_dst_stages.push(batch.wait_dst_stages[i]);
            }
        }

        // setup binary semaphore waits
        for &s in batch.wait_binary_semaphores.iter() {
            wait_semaphores.push(s);
            wait_semaphore_values.push(0);
            // TODO
            wait_semaphore_dst_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            // FIXME: this is incorrect. We immediately allow re-use of the semaphore, but there's
            // no guarantee that the next signal of the semaphore will happen after the wait that
            // we just queued. For instance, it could be signalled on another queue.
            //self.available_semaphores.push(s);

            // Every semaphore that is waited on is put in `used_semaphores`
            used_semaphores.push(s);
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

        let queue = self.device.queues_info.queues[q as usize];
        unsafe {
            self.device
                .device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .expect("queue submission failed");
        }
    }

    /// Creates a command pool and wraps it in a `CommandAllocator`.
    fn create_command_pool(&mut self, queue_index: usize) -> CommandAllocator {
        let queue_family = self.device.queues_info.families[queue_index];
        if let Some(pos) = self
            .available_command_pools
            .iter()
            .position(|cmd_pool| cmd_pool.queue_family == queue_family)
        {
            self.available_command_pools.swap_remove(pos)
        } else {
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

    ///
    pub(crate) fn submit_frame(
        &mut self,
        passes: &mut [Pass],
        wait_init: QueueSerialNumbers,
    ) -> FrameSubmissionResult {
        let _ = trace_span!("submit_frame").entered();

        // current submission batches per queue
        let mut cmd_batches: [CommandBatch; MAX_QUEUES] = Default::default();
        // one command pool per queue (might not be necessary if the queues belong to the same family,
        // but they usually don't)
        let mut cmd_pools: [Option<CommandAllocator>; MAX_QUEUES] = Default::default();
        // all binary semaphores waited on
        let mut used_semaphores = Vec::new();
        let mut submission_time_resources = SubmissionTimeResources::new();

        let mut first_pass_of_queue = [true; MAX_QUEUES];

        for p in passes.iter_mut() {
            // queue index
            let q = p.snn.queue();

            let wait_serials = if first_pass_of_queue[q] && wait_init > self.completed_serials {
                p.wait_serials.join(wait_init)
            } else {
                p.wait_serials
            };

            first_pass_of_queue[q] = false;

            // we need to wait if we have a binary semaphore, or if it's the first pass in this queue
            // and the user specified an initial wait before starting the frame.
            let needs_semaphore_wait =
                wait_serials > self.completed_serials || !p.wait_binary_semaphores.is_empty();

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
                batch.wait_dst_stages = p.wait_dst_stages; // TODO are those OK?
                batch.wait_binary_semaphores = p.wait_binary_semaphores.clone();
            }

            // ensure that a command pool has been allocated for the queue
            let command_pool: &mut CommandAllocator = cmd_pools[q as usize]
                .get_or_insert_with(|| self.create_command_pool(p.snn.queue()));
            // append to the last command buffer of the batch, otherwise create another one

            if batch.command_buffers.is_empty() {
                let cb = command_pool.allocate_command_buffer(&self.device.device);
                let begin_info = vk::CommandBufferBeginInfo {
                    ..Default::default()
                };
                unsafe {
                    // TODO safety
                    self.device
                        .device
                        .begin_command_buffer(cb, &begin_info)
                        .unwrap();
                }
                batch.command_buffers.push(cb);
            };

            let cb = batch.command_buffers.last().unwrap().clone();

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
                    // TODO safety
                    self.device.device.cmd_pipeline_barrier(
                        cb,
                        src_stage_mask,
                        dst_stage_mask,
                        Default::default(),
                        &[],
                        &p.buffer_memory_barriers,
                        &p.image_memory_barriers,
                    )
                }
            }

            match p.commands.take() {
                Some(PassCommands::CommandBuffer(handler)) => {
                    // perform a command-buffer level operation
                    let mut cctx = CommandContext {
                        context: self,
                        resources: &mut submission_time_resources,
                    };
                    handler(&mut cctx, cb);

                    // update signalled serial for the batch (pass serials are guaranteed to be increasing)
                    batch.signal_snn = p.snn;
                }
                Some(PassCommands::Queue(handler)) => {
                    // perform a queue-level operation:
                    // this terminates the current batch
                    self.submit_command_batch(q, batch, &mut used_semaphores);
                    batch.reset();
                    // call the handler
                    let queue = self.device.queues_info.queues[q as usize];
                    let mut cctx = CommandContext {
                        context: self,
                        resources: &mut submission_time_resources,
                    };
                    handler(&mut cctx, queue);
                }
                Some(PassCommands::Present {
                    swapchain,
                    image_index,
                }) => {
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
                    batch
                        .signal_binary_semaphores
                        .push(render_finished_semaphore);
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
                        self.device
                            .vk_khr_swapchain
                            .queue_present(queue, &present_info)
                            .expect("present failed");
                    }
                    // we signalled and waited on the semaphore, it can be reused
                    used_semaphores.push(render_finished_semaphore);
                }
                None => {}
            }

            unsafe {
                // FIXME this can end up in a different command buffer
                self.device.vk_ext_debug_utils.cmd_end_debug_utils_label(cb);
            }

            if p.signal_after {
                // the pass needs a semaphore signal: this terminates the batch on the queue
                // FIXME what does this do if the pass is a queue-level operation?
                self.submit_command_batch(q, batch, &mut used_semaphores);
                batch.reset();
            }
        }

        // close unfinished batches
        for batch in cmd_batches.iter() {
            self.submit_command_batch(batch.signal_snn.queue(), batch, &mut used_semaphores)
        }

        let command_pools = cmd_pools
            .iter_mut()
            .filter_map(|cmd_pool| cmd_pool.take())
            .collect();

        FrameSubmissionResult {
            command_pools,
            signalled_serials: self.last_signalled_serials,
            semaphores: used_semaphores,
            image_views: submission_time_resources.image_views,
            framebuffers: submission_time_resources.framebuffers,
            descriptor_sets: submission_time_resources.descriptor_sets,
        }
    }

    /// Recycles command pools returned by `submit_frame`.
    pub(crate) fn recycle_command_pools(&mut self, mut allocators: Vec<CommandAllocator>) {
        for a in allocators.iter_mut() {
            a.reset(&self.device.device)
        }
        self.available_command_pools.append(&mut allocators);
    }
}
