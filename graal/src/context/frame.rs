//! Contains code related to the construction of frames and passes.
use crate::{
    context::{
        is_write_access, local_pass_index, BufferId, Frame, FrameInner, GpuFuture, ImageId, Pass,
        PassEvaluationCallback, RecordingContext, ResourceAccess, ResourceAccessDetails,
        ResourceId, ResourceKind, SemaphoreSignal, SemaphoreSignalKind, SemaphoreWait,
        SemaphoreWaitKind, SyncDebugInfo, TemporarySet,
    },
    device::{AccessTracker, BufferResource, ImageResource, ResourceAllocation},
    serial::{FrameNumber, QueueSerialNumbers, SubmissionNumber},
    vk,
    vk::Handle,
    Context, Device, ResourceGroupId, ResourceOwnership, SwapchainImage,
};
use slotmap::Key;
use std::{fmt, mem, mem::ManuallyDrop};
use tracing::trace_span;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AccessTypeInfo {
    pub access_mask: vk::AccessFlags,
    pub stage_mask: vk::PipelineStageFlags,
    pub layout: vk::ImageLayout,
}

impl<'a, UserContext> Frame<'a, UserContext> {
    /// Returns this frame's number.
    pub fn frame_number(&self) -> FrameNumber {
        self.inner.frame_number
    }
}

pub struct PassBuilder<'a, 'b, UserContext> {
    frame: &'a mut Frame<'b, UserContext>,
    pass: ManuallyDrop<Pass<'b, UserContext>>,
}

impl<'a, 'b, UserContext> Drop for PassBuilder<'a, 'b, UserContext> {
    fn drop(&mut self) {
        panic!("PassBuilder object dropped. Use `PassBuilder::finished` instead")
    }
}

enum MemoryBarrierKind<'a> {
    Buffer {
        resource: &'a BufferResource,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    },
    Image {
        resource: &'a ImageResource,
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
        old_layout: vk::ImageLayout,
        new_layout: vk::ImageLayout,
    },
    Global {
        src_access_mask: vk::AccessFlags,
        dst_access_mask: vk::AccessFlags,
    },
}

struct PipelineBarrierDesc<'a> {
    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    memory_barrier: Option<MemoryBarrierKind<'a>>,
}

/// Helper function to add a memory dependency between source passes and a destination pass.
/// Updates the sync tracking information in `frame`.
///
/// # Arguments
///
/// * `frame`: the current frame
/// * `pass`: the pass being built
/// * `sources`: SNNs of the passes to synchronize with
fn add_memory_dependency<'a, UserContext>(
    frame: &mut FrameInner<'a, UserContext>,
    dst_pass: &mut Pass<'a, UserContext>,
    sources: QueueSerialNumbers,
    barrier: PipelineBarrierDesc,
) {
    let q = dst_pass.snn.queue();

    // from here, we have two possible methods of synchronization:
    // 1. semaphore signal/wait: this is straightforward, there's not a lot we can do
    // 2. pipeline barrier: we can choose **where** to put the barrier,
    //    we can put it anywhere between the source and the destination pass
    // -> if the source is just before, then use a pipeline barrier
    // -> if there's a gap, use an event?

    if !sources.is_single_source_same_queue_and_frame(q, frame.base_sn) {
        // Either:
        // - there are multiple sources across several queues
        // - the source is in a different queue
        // - the source is in an older frame
        // In those cases, a semaphore wait is necessary to synchronize.

        // go through each non-zero source
        for (iq, &sn) in sources.iter().enumerate() {
            if sn == 0 {
                continue;
            }

            // look in the cross-queue sync table to see if there's already an execution dependency
            // between the source (sn) and us.
            if frame.xq_sync_table[q].0[iq] >= sn {
                // already synced
                continue;
            }

            // we're adding a semaphore wait: update sync table
            frame.xq_sync_table[q].0[iq] = sn;

            dst_pass.wait_serials.0[iq] = sn;
            dst_pass.wait_dst_stages[iq] |= barrier.dst_stage_mask;

            if sn > frame.base_sn {
                let src_pass_index = local_pass_index(sn, frame.base_sn);
                let src_pass = &mut frame.passes[src_pass_index];
                src_pass.signal_queue_timelines = true;
                // update in-frame predecessor list for this pass; used for transient allocation
                dst_pass.preds.push(src_pass_index);
            }
        }
    } else {
        // There's only one source pass, which furthermore is on the same queue, and in the
        // same frame as the destination. In this case, we can use a pipeline barrier for
        // synchronization.

        let src_sn = sources[q];

        // sync dst=q, src=q
        if frame.xq_sync_table[q][q] >= src_sn {
            // if we're already synchronized with the source via a cross-queue (xq) wait
            // (a.k.a. semaphore), we don't need to add a memory barrier.
            // Note that layout transitions are handled separately, outside this condition.
        } else {
            // not synced with a semaphore, see if there's already a pipeline barrier
            // that ensures the execution dependency between the source (src_sn) and us

            let local_src_index = local_pass_index(src_sn, frame.base_sn);

            // The question we ask ourselves now is: is there already an execution dependency,
            // from the source pass, for the stages in `src_stage_mask`,
            // to us (dst_pass), for the stages in `dst_stage_mask`,
            // created by barriers in passes between the source and us?
            //
            // This is not easy to determine: to be perfectly accurate, we need to consider:
            // - transitive dependencies: e.g. COMPUTE -> FRAGMENT and then FRAGMENT -> TRANSFER also creates a COMPUTE -> TRANSFER dependency
            // - logically later and earlier stages: e.g. COMPUTE -> VERTEX also implies a COMPUTE -> FRAGMENT dependency
            //
            // For now, we just look for a pipeline barrier that directly contains the relevant stages
            // (i.e. `barrier.src_stage_mask` contains `src_stage_mask`, and `barrier.dst_stage_mask` contains `dst_stage_mask`,
            // ignoring transitive dependencies and any logical ordering between stages.
            //
            // The impact of this approximation is currently unknown.

            // find a pipeline barrier that already takes care of our execution dependency
            let barrier_pass = frame.passes[local_src_index..]
                .iter_mut()
                .skip(1)
                .find_map(|p| {
                    if p.snn.queue() == q
                        && p.src_stage_mask.contains(barrier.src_stage_mask)
                        && p.dst_stage_mask.contains(barrier.dst_stage_mask)
                    {
                        Some(p)
                    } else {
                        None
                    }
                })
                // otherwise, just add a pipeline barrier on the current pass
                .unwrap_or(dst_pass);

            // add our stages to the execution dependency
            barrier_pass.src_stage_mask |= barrier.src_stage_mask;
            barrier_pass.dst_stage_mask |= barrier.dst_stage_mask;

            // now deal with the memory dependency

            match barrier.memory_barrier {
                Some(MemoryBarrierKind::Image {
                    resource,
                    src_access_mask,
                    dst_access_mask,
                    old_layout,
                    new_layout,
                }) => {
                    let mb = barrier_pass
                        .get_or_create_image_memory_barrier(resource.handle, resource.format);
                    mb.src_access_mask |= src_access_mask;
                    mb.dst_access_mask |= dst_access_mask;
                    // Also specify the layout transition here.
                    // This is redundant with the code after that handles the layout transition,
                    // but we might not always go through here when a layout transition is necessary.
                    // With Sync2, just set these to UNDEFINED.
                    // TODO check for consistency: there should be at more one layout transition
                    // for the image in the pass
                    mb.old_layout = old_layout;
                    mb.new_layout = new_layout;
                }
                Some(MemoryBarrierKind::Buffer {
                    resource,
                    src_access_mask,
                    dst_access_mask,
                }) => {
                    let mb = barrier_pass.get_or_create_buffer_memory_barrier(resource.handle);
                    mb.src_access_mask |= src_access_mask;
                    mb.dst_access_mask |= dst_access_mask;
                }
                Some(MemoryBarrierKind::Global {
                    src_access_mask,
                    dst_access_mask,
                }) => {
                    let mb = barrier_pass.get_or_create_global_memory_barrier();
                    mb.src_access_mask |= src_access_mask;
                    mb.dst_access_mask |= dst_access_mask;
                }
                _ => {}
            }
        }
    }
}

impl<'a, 'b, UserContext> PassBuilder<'a, 'b, UserContext> {
    /// Adds a semaphore wait operation: the pass will first wait for the specified semaphore to be signalled
    /// before starting.
    pub fn add_external_semaphore_wait(
        &mut self,
        semaphore: vk::Semaphore,
        dst_stage: vk::PipelineStageFlags,
        wait_kind: SemaphoreWaitKind,
    ) {
        self.pass.external_semaphore_waits.push(SemaphoreWait {
            semaphore,
            owned: false,
            dst_stage,
            wait_kind,
        })
    }

    /// Adds a semaphore signal operation: when finished, the pass will signal the specified semaphore.
    pub fn add_external_semaphore_signal(
        &mut self,
        semaphore: vk::Semaphore,
        signal_kind: SemaphoreSignalKind,
    ) {
        self.pass.external_semaphore_signals.push(SemaphoreSignal {
            semaphore,
            signal_kind,
        })
    }

    /// Registers an image access made by this pass.
    pub fn add_image_dependency(
        &mut self,
        id: ImageId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) {
        self.reference_resource(
            id.0,
            &ResourceAccessDetails {
                initial_layout,
                final_layout,
                access_mask,
                stage_mask,
            },
        )
    }

    pub fn add_buffer_dependency(
        &mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    ) {
        self.reference_resource(
            id.0,
            &ResourceAccessDetails {
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::UNDEFINED,
                access_mask,
                stage_mask,
            },
        )
    }

    /// Sets the command buffer recording function for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_record_callback(
        &mut self,
        record_cb: impl FnOnce(&mut RecordingContext, &mut UserContext, vk::CommandBuffer) + 'b,
    ) {
        self.pass.eval_callback = Some(PassEvaluationCallback::CommandBuffer(Box::new(record_cb)));
    }

    pub fn set_submit_callback(
        &mut self,
        submit_cb: impl FnOnce(&mut RecordingContext, &mut UserContext, vk::Queue) + 'b,
    ) {
        self.pass.eval_callback = Some(PassEvaluationCallback::Queue(Box::new(submit_cb)));
    }

    /// Registers an access to a resource within the specified pass and updates the dependency graph.
    ///
    /// This is the meat of the automatic synchronization system: given the known state of the resources,
    /// this function infers the necessary execution barriers, memory barriers, and layout transitions,
    /// and updates the state of the resources.
    fn reference_resource(&mut self, id: ResourceId, access: &ResourceAccessDetails) {
        let mut objects = self.frame.context.device.objects.lock().unwrap();
        let resources = &mut objects.resources;
        let resource = resources.get_mut(id).unwrap();

        let frame = &mut self.frame.inner;
        let dst_pass = &mut self.pass;

        assert!(!resource.discarded, "referenced a discarded resource");
        // we can't synchronize on a resource that belongs to a group: we synchronize on the group instead
        assert!(resource.group.is_none(), "cannot synchronize on a resource belonging to a group; synchronize on the group instead");

        //------------------------
        // handle the special case of upload buffers: when the last access is a write in the host domain,
        // and only reading from the resource.
        // In this case, we don't need to add an execution dependency, as the spec guarantees that host writes are made visible.

        //------------------------
        // first, add the resource into the set of temporaries used within this frame
        if frame.temporary_set.insert(id) {
            frame.temporaries.push(id);
        }

        // set first access
        if resource.tracking.first_access.is_none() {
            resource.tracking.first_access = Some(AccessTracker::Device(dst_pass.snn));
        }

        // First, some definitions:
        // - current pass         : the pass which is accessing the resource, and for which we are registering a dependency
        // - current SN           : the SN of the current pass
        // - writer pass          : the pass that last wrote to the resource
        // - writer SN            : the SN of the writer pass
        // - input stage mask     : the pipeline stages that will access the resource in the current pass
        // - writer stage mask    : the pipeline stages that the writer
        // - availability barrier : the memory barrier that is in charge of making the writes available and visible to other stages
        //                          typically, the barrier is stored in the first reader
        //
        // 1. First, determine if we can do without any kind of synchronization. This is the case if:
        //      - the resource has no explicit binary semaphore to synchronize with
        //      - AND all previous writes are already visible
        //      - AND the resource doesn't need a layout transition
        //      -> If all of this is true, then skip to X
        // 2. Get or create the barrier
        //      The resulting barrier might be associated to the pass, or an existing one that comes before

        // If the resource has an associated semaphore, consume it.
        // For now, the only resources that have associated semaphores are swapchain images from the presentation engine.
        let semaphore = mem::take(&mut resource.tracking.wait_binary_semaphore);
        let has_external_semaphore = semaphore != vk::Semaphore::null();
        if has_external_semaphore {
            dst_pass.external_semaphore_waits.push(SemaphoreWait {
                semaphore,
                owned: true,
                dst_stage: vk::PipelineStageFlags::TOP_OF_PIPE, // FIXME maybe?
                wait_kind: SemaphoreWaitKind::Binary,
            });
        }

        //------------------------
        let need_layout_transition = resource.tracking.layout != access.initial_layout;

        // is the access a write? for synchronization purposes, layout transitions are the same thing as a write
        let is_write = is_write_access(access.access_mask) || need_layout_transition;

        // can we ensure that all previous writes are visible?
        let writes_visible = match resource.tracking.writer {
            None => {
                // no writes, no data to see, no barrier necessary.
                true
            }
            Some(AccessTracker::Host) => {
                // no need for a barrier if host write
                // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#synchronization-submission-host-writes
                true
            }
            Some(AccessTracker::Device(writer)) => {
                // the visibility mask is only valid if this access and the last write is in the same queue
                // for cross-queue accesses, we never skip
                writer.queue() == dst_pass.snn.queue()
                    && (resource
                        .tracking
                        .visibility_mask
                        .contains(access.access_mask)
                        || resource
                            .tracking
                            .visibility_mask
                            .contains(vk::AccessFlags::MEMORY_READ))
            }
        };

        // --- (1) skip to the end if no barrier is needed
        // No barrier is needed if we waited on an external semaphore, or all writes are visible and no layout transition is necessary

        if (!has_external_semaphore && !writes_visible) || need_layout_transition {
            let q = dst_pass.snn.queue() as usize;

            // Determine the "sources" of the dependency: i.e. the passes (identified by serials),
            // that we must synchronize with.
            //
            // If we're writing to the resource, and the resource is being read, we must wait for
            // all reads to complete, and thus synchronize with the readers.
            // Otherwise, if we're only reading from the resource, or if we're writing but there are no readers,
            // we must synchronize with the last writer (we can have multiple concurrent readers).
            //
            // Note that a resource can't have both a reader and a writer at the same time.
            let sync_sources = if is_write && resource.tracking.has_readers() {
                // Write-after-read dependency
                resource.tracking.readers
            } else {
                // Write-after-write, read-after-write
                match resource.tracking.writer {
                    None => {
                        // no sources
                        QueueSerialNumbers::default()
                    }
                    Some(AccessTracker::Device(writer)) => {
                        QueueSerialNumbers::from_submission_number(writer)
                    }
                    Some(AccessTracker::Host) => {
                        // Shouldn't happen: WAW or RAW with the write being host access.
                        // FIXME: actually, it's possible when a host-mapped image, with linear storage
                        // and in the GENERAL layout is requested access with a different layout.
                        // TODO better panic message
                        panic!("unsupported dependency")
                    }
                }
            };

            add_memory_dependency(
                frame,
                dst_pass,
                sync_sources,
                PipelineBarrierDesc {
                    src_stage_mask: resource.tracking.stages,
                    dst_stage_mask: access.stage_mask,
                    memory_barrier: match &resource.kind {
                        ResourceKind::Buffer(buf) => Some(MemoryBarrierKind::Buffer {
                            resource: buf,
                            src_access_mask: resource.tracking.availability_mask,
                            dst_access_mask: access.access_mask,
                        }),
                        ResourceKind::Image(img) => Some(MemoryBarrierKind::Image {
                            resource: img,
                            src_access_mask: resource.tracking.availability_mask,
                            dst_access_mask: access.access_mask,
                            old_layout: resource.tracking.layout,
                            new_layout: access.initial_layout,
                        }),
                    },
                },
            );

            if sync_sources.is_single_source_same_queue_and_frame(q, frame.base_sn) {
                // TODO is this really necessary?
                // I think it's just there so that we can return early when syncing on the same queue (see `writes_visible`).
                // However even if we proceed, add_memory_dependency shouldn't emit a redundant barrier anyway.

                // this memory dependency makes all writes on the resource available, and
                // visible to the types specified in `access.access_mask`
                resource.tracking.availability_mask = vk::AccessFlags::empty();
                resource.tracking.visibility_mask |= access.access_mask;
            }

            // layout transitions
            if need_layout_transition {
                let image = resource.image();
                let mb = dst_pass.get_or_create_image_memory_barrier(image.handle, image.format);
                mb.old_layout = resource.tracking.layout;
                mb.new_layout = access.initial_layout;
                resource.tracking.layout = access.final_layout;
            }
        }

        if is_write_access(access.access_mask) {
            // we're writing to the resource, so reset visibility...
            resource.tracking.visibility_mask = vk::AccessFlags::empty();
            // ... but signal that there is data to be made available for this resource.
            resource.tracking.availability_mask |= access.access_mask;
        }

        // update output stage
        // FIXME I have doubts about this code
        if is_write {
            resource.tracking.stages = access.stage_mask;
            resource.tracking.clear_readers();
            resource.tracking.writer = Some(AccessTracker::Device(dst_pass.snn));
        } else {
            // update the resource readers
            resource.tracking.readers = resource.tracking.readers.join_serial(dst_pass.snn);
        }

        // record the access in the pass
        //
        dst_pass.accesses.insert(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }

    /// Ends the current pass.
    pub fn finish(mut self) {
        let pass = unsafe {
            // SAFETY: self.pass not used afterwards, including Drop
            ManuallyDrop::take(&mut self.pass)
        };
        self.frame.inner.passes.push(pass);

        if self.frame.inner.collect_sync_debug_info {
            let mut info = SyncDebugInfo::new();

            // current resource tracking info
            let objects = self.frame.context.device.objects.lock().unwrap();
            for (id, r) in objects.resources.iter() {
                info.tracking.insert(id, r.tracking);
            }
            // current sync table
            info.xq_sync_table = self.frame.inner.xq_sync_table;
            self.frame.inner.sync_debug_info.push(info);
        }

        // skip running the panicking destructor
        mem::forget(self)
    }

    /// Adds a resource to a resource group.
    fn add_resource_to_group(&self, resource_id: ResourceId, group_id: ResourceGroupId) {
        let mut objects = self.frame.context.device.objects.lock().unwrap();
        let objects = &mut *objects;
        let group = objects
            .resource_groups
            .get_mut(group_id)
            .expect("invalid or expired resource group");
        let mut resource = objects
            .resources
            .get_mut(resource_id)
            .expect("invalid resource");
        assert!(resource.group.is_none());

        // set group
        resource.group = Some(group_id);
        // set additional serials and stages to wait for in this group
        match resource.tracking.writer {
            Some(AccessTracker::Device(writer)) => group
                .wait_serials
                .join_assign(QueueSerialNumbers::from_submission_number(writer)),
            Some(AccessTracker::Host) => {
                // FIXME why? it would make sense to add all upload buffers in a frame to a single group
                panic!("host-accessible resources cannot be added to a group")
            }
            None => {
                // FIXME this might not warrant a panic: the resource will simply be
                // frozen in an uninitialized state.
                panic!("tried to add an unwritten resource to a group")
            }
        }
        group.src_stage_mask |= resource.tracking.stages;
        group.src_access_mask |= resource.tracking.availability_mask;
    }

    /// Adds an image to a resource group.
    pub fn freeze_image(&self, image_id: ImageId, group_id: ResourceGroupId) {
        self.add_resource_to_group(image_id.0, group_id)
    }

    /// Adds a buffer to a resource group.
    pub fn freeze_buffer(&self, buffer_id: BufferId, group_id: ResourceGroupId) {
        self.add_resource_to_group(buffer_id.0, group_id)
    }

    /// Registers an access to a resource group
    // FIXME: don't specify access and dst stage: those should be set in the group; avoid putting
    // more than one execution and visibility barrier
    pub fn reference_group(&mut self, id: ResourceGroupId) {
        let objects = self.frame.context.device.objects.lock().unwrap();

        // we just have to wait for the SNNs and stages of the group.
        let group = objects
            .resource_groups
            .get(id)
            .expect("invalid or expired resource group");

        add_memory_dependency(
            &mut self.frame.inner,
            &mut self.pass,
            group.wait_serials,
            PipelineBarrierDesc {
                src_stage_mask: group.src_stage_mask, // the mask will be big (combined stages of all writes)
                dst_stage_mask: group.dst_stage_mask,
                memory_barrier: Some(MemoryBarrierKind::Global {
                    src_access_mask: group.src_access_mask,
                    dst_access_mask: group.dst_access_mask,
                }),
            },
        );
    }
}

#[derive(Copy, Clone, Debug)]
pub struct FrameCreateInfo {
    pub happens_after: GpuFuture,
    pub collect_debug_info: bool,
}

impl Default for FrameCreateInfo {
    fn default() -> Self {
        FrameCreateInfo {
            happens_after: Default::default(),
            collect_debug_info: false,
        }
    }
}

/// Determines on which queue a pass will be scheduled.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum PassType {
    Graphics,
    Compute,
    Transfer,
    Present,
}

impl PassType {
    /// Returns the queue index assigned for this pass type.
    fn queue_index(&self, device: &Device, async_pass: bool) -> usize {
        (match self {
            PassType::Compute if async_pass => device.queues_info.indices.compute,
            PassType::Transfer if async_pass => device.queues_info.indices.transfer,
            PassType::Present { .. } => device.queues_info.indices.present,
            _ => device.queues_info.indices.graphics,
        }) as usize
    }
}

#[derive(Clone, Debug)]
pub struct PresentOperationResult {
    pub swapchain: vk::SwapchainKHR,
    pub result: vk::Result,
}

#[derive(Clone, Debug)]
/// The result of a frame submission
pub struct FrameSubmitResult {
    /// GPU future for the frame.
    pub future: GpuFuture,
    /// Results of present operations.
    pub present_results: Vec<PresentOperationResult>,
}

impl<'a, UserContext> fmt::Debug for Frame<'a, UserContext> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Frame")
            .field("base_serial", &self.inner.base_sn)
            .field("frame_serial", &self.inner.frame_number)
            .finish()
    }
}

impl<'a, UserContext> Frame<'a, UserContext> {
    /// Adds a dependency on a GPU future object.
    ///
    /// Execution of the *whole* frame will wait for the operation represented by the future to complete.
    pub fn add_dependency(&mut self, future: GpuFuture) {
        self.inner.wait_init = self.inner.wait_init.join(future.serials);
    }

    /// Starts a graphics pass.
    pub fn start_graphics_pass<'frame>(
        &'frame mut self,
        name: &str,
    ) -> PassBuilder<'frame, 'a, UserContext> {
        self.start_pass(name, PassType::Graphics, false)
    }

    /// Starts a compute pass
    pub fn start_compute_pass<'frame>(
        &'frame mut self,
        name: &str,
        async_compute: bool,
    ) -> PassBuilder<'frame, 'a, UserContext> {
        self.start_pass(name, PassType::Compute, async_compute)
    }

    /// Starts a transfer pass
    pub fn start_transfer_pass<'frame>(
        &'frame mut self,
        name: &str,
        async_transfer: bool,
    ) -> PassBuilder<'frame, 'a, UserContext> {
        self.start_pass(name, PassType::Transfer, async_transfer)
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&mut self, name: &str, image: &SwapchainImage) {
        let mut pass = self.start_pass(name, PassType::Present, false);

        pass.pass.eval_callback = Some(PassEvaluationCallback::Present {
            swapchain: image.swapchain_handle,
            image_index: image.image_index,
        });
        pass.add_image_dependency(
            image.image_info.id,
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::ALL_COMMANDS, // ?
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        pass.finish();
    }

    /// Common code for `start_xxx_pass`
    fn start_pass<'frame>(
        &'frame mut self,
        name: &str,
        ty: PassType,
        async_pass: bool,
    ) -> PassBuilder<'frame, 'a, UserContext> {
        let frame_index = self.inner.passes.len();
        // not sure why we need to do it in this order
        self.inner.current_sn += 1;
        let serial = self.inner.current_sn;
        let q = ty.queue_index(&self.context.device, async_pass);
        let snn = SubmissionNumber::new(q, serial);
        PassBuilder {
            frame: self,
            pass: ManuallyDrop::new(Pass::new(name, frame_index, snn)),
        }
    }

    /// Dumps the frame to a JSON object.
    pub fn dump(&self, file_name_prefix: Option<&str>) {
        use serde_json::json;
        use std::fs::File;

        let objects = self.context.device.objects.lock().unwrap();
        let resources = &objects.resources;

        // passes
        let mut passes_json = Vec::new();
        for (pass_index, p) in self.inner.passes.iter().enumerate() {
            let image_memory_barriers_json: Vec<_> = p
                .image_memory_barriers
                .iter()
                .map(|imb| {
                    let id = objects.image_resource_by_handle(imb.image);
                    let name = &resources.get(id).unwrap().name;
                    json!({
                        "type": "image",
                        "srcAccessMask": format!("{:?}", imb.src_access_mask),
                        "dstAccessMask": format!("{:?}", imb.dst_access_mask),
                        "oldLayout": format!("{:?}", imb.old_layout),
                        "newLayout": format!("{:?}", imb.new_layout),
                        "handle": format!("{:#x}", imb.image.as_raw()),
                        "id": format!("{:?}", id.data()),
                        "name": name
                    })
                })
                .collect();

            let buffer_memory_barriers_json: Vec<_> = p
                .buffer_memory_barriers
                .iter()
                .map(|bmb| {
                    let id = objects.buffer_resource_by_handle(bmb.buffer);
                    let name = &resources.get(id).unwrap().name;
                    json!({
                        "type": "buffer",
                        "srcAccessMask": format!("{:?}", bmb.src_access_mask),
                        "dstAccessMask": format!("{:?}", bmb.dst_access_mask),
                        "handle": format!("{:#x}", bmb.buffer.as_raw()),
                        "id": format!("{:?}", id.data()),
                        "name": name
                    })
                })
                .collect();

            let accesses_json: Vec<_> = p
                .accesses
                .iter()
                .map(|a| {
                    let r = resources.get(a.id).unwrap();
                    let name = &r.name;
                    let (ty, handle) = match r.kind {
                        ResourceKind::Buffer(ref buf) => ("buffer", buf.handle.as_raw()),
                        ResourceKind::Image(ref img) => ("image", img.handle.as_raw()),
                    };

                    json!({
                        "id": format!("{:?}", a.id.data()),
                        "name": name,
                        "handle": format!("{:#x}", handle),
                        "type": ty,
                        "accessMask": format!("{:?}", a.access_mask),
                    })
                })
                .collect();

            let mut pass_json = json!({
                "name": p.name,
                "queue": p.snn.queue(),
                "serial": p.snn.serial(),
                "accesses": accesses_json,
                "barriers": {
                    "srcStageMask": format!("{:?}", p.src_stage_mask),
                    "dstStageMask": format!("{:?}", p.dst_stage_mask),
                    "imageMemoryBarriers": image_memory_barriers_json,
                    "bufferMemoryBarriers": buffer_memory_barriers_json,
                },
                "wait": {
                    "serials": p.wait_serials.0,
                    "waitDstStages": format!("{:?}", p.wait_dst_stages),
                },
                "waitExternal": !p.external_semaphore_waits.is_empty(),
            });

            // additional debug information
            if self.inner.collect_sync_debug_info {
                let sync_debug_info = &self.inner.sync_debug_info[pass_index];

                let mut resource_tracking_json = Vec::new();
                for (id, tracking) in sync_debug_info.tracking.iter() {
                    let name = &resources.get(id).unwrap().name;

                    let (host_write, writer_queue, writer_serial) = match tracking.writer {
                        None => (false, 0, 0),
                        Some(AccessTracker::Device(snn)) => (false, snn.queue(), snn.serial()),
                        Some(AccessTracker::Host) => (true, 0, 0),
                    };

                    resource_tracking_json.push(json!({
                        "id": format!("{:?}", id.data()),
                        "name": name,
                        "readers": tracking.readers.0,
                        "hostWrite": host_write,
                        "writerQueue": writer_queue,
                        "writerSerial": writer_serial,
                        "layout": format!("{:?}", tracking.layout),
                        "availabilityMask": format!("{:?}", tracking.availability_mask),
                        "visibilityMask": format!("{:?}", tracking.visibility_mask),
                        "stages": format!("{:?}", tracking.stages),
                        "binarySemaphore": tracking.wait_binary_semaphore.as_raw(),
                    }));
                }

                let xq_sync_json: Vec<_> =
                    sync_debug_info.xq_sync_table.iter().map(|v| v.0).collect();

                pass_json.as_object_mut().unwrap().insert(
                    "syncDebugInfo".to_string(),
                    json!({
                        "resourceTrackingInfo": resource_tracking_json,
                        "crossQueueSyncTable": xq_sync_json,
                    }),
                );
            }

            passes_json.push(pass_json);
        }

        let frame_json = json!({
            "frameSerial": self.inner.frame_number.0,
            "baseSerial": self.inner.base_sn,
            "passes": passes_json,
        });

        let file = File::create(format!(
            "{}-{}.json",
            file_name_prefix.unwrap_or("frame"),
            self.inner.frame_number.0
        ))
        .expect("could not open file for dumping JSON frame information");
        serde_json::to_writer_pretty(file, &frame_json).unwrap();
    }

    fn print_frame_info(&self, passes: &[Pass<UserContext>], temporaries: &[ResourceId]) {
        let objects = self.context.device.objects.lock().unwrap();
        let resources = &objects.resources;

        println!("=============================================================");
        println!("Passes:");
        for p in passes.iter() {
            println!("- `{}` ({:?})", p.name, p.snn);
            if p.wait_serials != Default::default() {
                println!("    semaphore wait:");
                if p.wait_serials[0] != 0 {
                    println!("        0:{}|{:?}", p.wait_serials[0], p.wait_dst_stages[0]);
                }
                if p.wait_serials[1] != 0 {
                    println!("        1:{}|{:?}", p.wait_serials[1], p.wait_dst_stages[1]);
                }
                if p.wait_serials[2] != 0 {
                    println!("        2:{}|{:?}", p.wait_serials[2], p.wait_dst_stages[2]);
                }
                if p.wait_serials[3] != 0 {
                    println!("        3:{}|{:?}", p.wait_serials[3], p.wait_dst_stages[3]);
                }
            }
            println!(
                "    input execution barrier: {:?}->{:?}",
                p.src_stage_mask, p.dst_stage_mask
            );
            println!("    input memory barriers:");
            for imb in p.image_memory_barriers.iter() {
                let id = objects.image_resource_by_handle(imb.image);
                print!("        image handle={:?} ", imb.image);
                if !id.is_null() {
                    print!("(id={:?}, name={})", id, resources.get(id).unwrap().name);
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?} layout:{:?}->{:?}",
                    imb.src_access_mask, imb.dst_access_mask, imb.old_layout, imb.new_layout
                );
            }
            for bmb in p.buffer_memory_barriers.iter() {
                let id = objects.buffer_resource_by_handle(bmb.buffer);
                print!("        buffer handle={:?} ", bmb.buffer);
                if !id.is_null() {
                    print!("(id={:?}, name={})", id, resources.get(id).unwrap().name);
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?}",
                    bmb.src_access_mask, bmb.dst_access_mask
                );
            }

            //println!("    output stage: {:?}", p.output_stage_mask);
            if p.signal_queue_timelines {
                println!("    semaphore signal: {:?}", p.snn);
            }
        }

        println!("Final resource states: ");

        for &id in temporaries.iter() {
            let resource = resources.get(id).unwrap();
            println!("`{}`", resource.name);
            println!("    stages={:?}", resource.tracking.stages);
            println!("    avail={:?}", resource.tracking.availability_mask);
            println!("    vis={:?}", resource.tracking.visibility_mask);
            println!("    layout={:?}", resource.tracking.layout);

            if resource.tracking.has_readers() {
                println!("    readers: ");
                if resource.tracking.readers[0] != 0 {
                    println!("        0:{}", resource.tracking.readers[0]);
                }
                if resource.tracking.readers[1] != 0 {
                    println!("        1:{}", resource.tracking.readers[1]);
                }
                if resource.tracking.readers[2] != 0 {
                    println!("        2:{}", resource.tracking.readers[2]);
                }
                if resource.tracking.readers[3] != 0 {
                    println!("        3:{}", resource.tracking.readers[3]);
                }
            }
            if resource.tracking.has_writer() {
                println!("    writer: {:?}", resource.tracking.writer);
            }
            match resource.ownership {
                ResourceOwnership::External => {}
                ResourceOwnership::OwnedResource { ref allocation, .. } => match allocation {
                    Some(ResourceAllocation::Default { allocation: _ }) => {
                        println!("    allocation: exclusive");
                    }
                    Some(ResourceAllocation::External { device_memory }) => {
                        println!(
                            "    allocation: external, device memory {:016x}",
                            device_memory.as_raw()
                        );
                    }
                    Some(ResourceAllocation::Transient {
                        device_memory,
                        offset,
                    }) => {
                        println!(
                            "    allocation: transient, device memory {:016x}@{:016x}",
                            device_memory.as_raw(),
                            offset
                        );
                    }
                    None => {
                        println!("    allocation: none (unallocated)");
                    }
                },
            }
        }
    }

    /// Finishes building the frame and submits all the passes to the command queues.
    pub fn finish(self, user_context: &mut UserContext) -> FrameSubmitResult {
        //assert!(self.device.context_state.is_building_frame, "not building a frame");

        //self.dump(None);
        //self.print_frame_info(&frame.passes, &frame.temporaries);

        // First, wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the frames that are not in use anymore.
        self.context.wait_for_frames_in_flight();

        let last_sn = self.inner.current_sn;

        // Submit the frame
        let submit_result = self.context.submit_frame(self.inner, user_context);

        // Update last submitted pass SN
        self.context.last_sn = last_sn;
        self.context.device.end_frame();

        submit_result
    }
}

impl Context {
    /// Starts a new frame. The execution of the frame can optionally be synchronized
    /// with the given future in `happens_after`.
    ///
    /// However, regardless of this, individual passes in the frame may still synchronize with earlier frames
    /// because of resource dependencies.
    pub fn start_frame<'a, UserContext>(
        &'a mut self,
        create_info: FrameCreateInfo,
    ) -> Frame<'a, UserContext> {
        let base_sn = self.last_sn;
        let wait_init = create_info.happens_after.serials;

        // Full CPU-side frame processing
        let span = trace_span!("frame", base_sn).entered();
        // DAG build only
        let build_span = trace_span!("DAG build").entered();

        let frame_number = FrameNumber(self.submitted_frame_count + 1);

        // update the context state in the device
        self.device.start_frame(frame_number);

        Frame {
            context: self,
            inner: FrameInner {
                base_sn,
                current_sn: base_sn,
                frame_number,
                wait_init,
                temporaries: vec![],
                temporary_set: TemporarySet::new(),
                passes: vec![],
                xq_sync_table: Default::default(),
                collect_sync_debug_info: create_info.collect_debug_info,
                sync_debug_info: Vec::new(),
                span,
                build_span,
            },
        }
    }
}
