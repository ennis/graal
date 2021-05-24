use crate::{
    context::{
        is_write_access,
        pass::{
            Pass, PassCommands, ResourceAccess, SemaphoreSignal, SemaphoreSignalKind,
            SemaphoreWait, SemaphoreWaitKind,
        },
        BufferId, ImageId, ResourceAccessDetails, ResourceId, ResourceKind,
        ResourceTrackingInfo, TypedBufferInfo,
        CommandContext, FrameInFlight, FrameSerialNumber, GpuFuture, QueueSerialNumbers,
        SubmissionNumber,
    },
    vk,
    vk::Handle,
    Context,
    MAX_QUEUES,
};
use slotmap::Key;
use std::{
    cell::{RefCell, RefMut},
    mem,
};
use tracing::trace_span;
use crate::context::{ResourceOwnership, ResourceAllocation, local_pass_index};
use crate::swapchain::SwapchainImage;
use crate::context::transient::allocate_memory_for_transients;

type TemporarySet = std::collections::BTreeSet<ResourceId>;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AccessTypeInfo {
    pub access_mask: vk::AccessFlags,
    pub stage_mask: vk::PipelineStageFlags,
    pub layout: vk::ImageLayout,
}

/// Builder object for passes.
pub struct PassBuilder<'a, 'frame> {
    frame: &'frame mut FrameInner<'a>,
    pass: Pass<'a>,
    _span: tracing::span::EnteredSpan,
}

impl<'a, 'frame> PassBuilder<'a, 'frame> {
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
    pub fn register_image_access(
        &mut self,
        id: ImageId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) {
        self.frame.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                initial_layout,
                final_layout,
                access_mask,
                stage_mask,
            },
        )
    }

    pub fn register_buffer_access(
        &mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    ) {
        self.frame.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                initial_layout: vk::ImageLayout::UNDEFINED,
                final_layout: vk::ImageLayout::UNDEFINED,
                access_mask,
                stage_mask,
            },
        )
    }

    /// Sets the command handler for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_commands(
        &mut self,
        commands: impl FnOnce(&mut CommandContext, vk::CommandBuffer) + 'a,
    ) {
        self.pass.commands = Some(PassCommands::CommandBuffer(Box::new(commands)));
    }

    pub fn set_queue_commands(
        &mut self,
        commands: impl FnOnce(&mut CommandContext, vk::Queue) + 'a,
    ) {
        self.pass.commands = Some(PassCommands::Queue(Box::new(commands)));
    }
}

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

// ---------------------------------------------------------------------------------------
struct FrameInner<'a> {
    base_serial: u64,
    context: &'a mut Context,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the frame
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass<'a>>,
    /// Serials to wait for before executing the frame.
    wait_init: QueueSerialNumbers,
    /// Cross-queue synchronization table
    /// TODO detailed description
    xq_sync_table: [QueueSerialNumbers; MAX_QUEUES],

    collect_sync_debug_info: bool,
    sync_debug_info: Vec<SyncDebugInfo>,
}

impl<'a> FrameInner<'a> {
    /// Registers an access to a resource within the specified pass and updates the dependency graph.
    ///
    /// This is the meat of the automatic synchronization system: given the known state of the resources,
    /// this function infers the necessary execution barriers, memory barriers, and layout transitions,
    /// and updates the state of the resources.
    fn add_resource_dependency(
        &mut self,
        dst_pass: &mut Pass<'a>,
        id: ResourceId,
        access: &ResourceAccessDetails,
    ) {
        //------------------------
        // first, add the resource into the set of temporaries used within this frame
        let resource = self.context.resources.get_mut(id).unwrap();
        if self.temporary_set.insert(id) {
            self.temporaries.push(id);
        }

        // set first use
        if !resource.tracking.first_access.is_valid() {
            resource.tracking.first_access = dst_pass.snn;
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
        // note: the visibility mask is only valid if this access and the last write is in the same queue
        // for cross-queue accesses, we never skip
        let writes_visible = resource.tracking.writer.queue() == dst_pass.snn.queue()
            && (resource
                .tracking
                .visibility_mask
                .contains(access.access_mask)
                || resource
                    .tracking
                    .visibility_mask
                    .contains(vk::AccessFlags::MEMORY_READ));

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
                QueueSerialNumbers::from_submission_number(resource.tracking.writer)
            };

            // from here, we have two possible methods of synchronization:
            // 1. semaphore signal/wait: this is straightforward, there's not a lot we can do
            // 2. pipeline barrier: we can choose **where** to put the barrier,
            //    we can put it anywhere between the source and the destination pass
            // -> if the source is just before, then use a pipeline barrier
            // -> if there's a gap, use an event?
            let base_serial = self.base_serial;

            // whether `sync_sources` identifies a single source pass in the queue that one we're
            // submitting on
            let single_local_source_in_same_queue = sync_sources
                .iter()
                .enumerate()
                .all(|(i, &sn)| (i != q && sn == 0) || (i == q && (sn == 0 || sn > base_serial)));

            if !single_local_source_in_same_queue {
                // Either:
                // - there are multiple sources across several queues
                // - the source is in a different queue
                // - the source is in an older frame
                // In those cases, a semaphore wait is necessary to synchronize.

                // go through each non-zero source
                for (iq, &sn) in sync_sources.iter().enumerate() {
                    if sn == 0 {
                        continue;
                    }

                    // look in the cross-queue sync table to see if there's already an execution dependency
                    // between the source (sn) and us.
                    if self.xq_sync_table[q].0[iq] >= sn {
                        // already synced
                        continue;
                    }

                    // we're adding a semaphore wait: update sync table
                    self.xq_sync_table[q].0[iq] = sn;

                    dst_pass.wait_serials.0[iq] = sn;
                    dst_pass.wait_dst_stages[iq] |= access.stage_mask;

                    if sn > self.base_serial {
                        // furthermore, if source and destination are in the same frame, add
                        // an edge to the depgraph (regardless of whether we added a semaphore or not)
                        let src_pass_index = local_pass_index(sn, self.base_serial);
                        //let dst_pass_index = dst_pass.frame_index;
                        let src_pass = &mut self.passes[src_pass_index];
                        //src_pass.succs.push(dst_pass_index);
                        //dst_pass.preds.push(src_pass_index);
                        src_pass.signal_queue_timelines = true;
                    }
                }
            } else {
                // There's only one source pass, which furthermore is on the same queue, and in the
                // same frame as the destination. In this case, we can use a pipeline barrier for
                // synchronization.

                let src_sn = sync_sources[q];
                let src_stage_mask = resource.tracking.stages;
                let dst_stage_mask = access.stage_mask;

                // sync dst=q, src=q
                if self.xq_sync_table[q][q] >= src_sn {
                    // if we're already synchronized with the source via a cross-queue (xq) wait
                    // (a.k.a. semaphore), we don't need to add a memory barrier.
                    // Note that layout transitions are handled separately, outside this condition.
                } else {
                    // not synced with a semaphore, see if there's already a pipeline barrier
                    // that ensures the execution dependency between the source (src_sn) and us

                    let local_src_index = local_pass_index(src_sn, self.base_serial);

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
                    let barrier_pass = self.passes[local_src_index + 1..]
                        .iter_mut()
                        .find_map(|p| {
                            if p.snn.queue() == q
                                && p.src_stage_mask.contains(src_stage_mask)
                                && p.dst_stage_mask.contains(dst_stage_mask)
                            {
                                Some(p)
                            } else {
                                None
                            }
                        })
                        .unwrap_or(dst_pass);

                    // add our stages to the execution dependency
                    barrier_pass.src_stage_mask |= src_stage_mask;
                    barrier_pass.dst_stage_mask |= dst_stage_mask;

                    // now deal with the memory dependency
                    match &resource.kind {
                        ResourceKind::Image(img) => {
                            let mb = barrier_pass
                                .get_or_create_image_memory_barrier(img.handle, img.format);
                            mb.src_access_mask |= resource.tracking.availability_mask;
                            mb.dst_access_mask |= access.access_mask;
                            // Also specify the layout transition here.
                            // This is redundant with the code after that handles the layout transition,
                            // but we might not always go through here when a layout transition is necessary.
                            // With Sync2, just set these to UNDEFINED.
                            mb.old_layout = resource.tracking.layout;
                            mb.new_layout = access.initial_layout;
                        }
                        ResourceKind::Buffer(buf) => {
                            let mb = barrier_pass.get_or_create_buffer_memory_barrier(buf.handle);
                            mb.src_access_mask |= resource.tracking.availability_mask;
                            mb.dst_access_mask |= access.access_mask;
                        }
                    }

                    // this memory dependency makes all writes on the resource available, and
                    // visible to the types specified in `access.access_mask`
                    resource.tracking.availability_mask = vk::AccessFlags::empty();
                    resource.tracking.visibility_mask |= access.access_mask;
                }
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
            resource.tracking.writer = dst_pass.snn;
        } else {
            // update the resource readers
            resource.tracking.readers = resource.tracking.readers.join_serial(dst_pass.snn);
        }

        // record the access in the pass
        dst_pass.accesses.push(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }

    fn push_pass(&mut self, pass: Pass<'a>) {
        self.passes.push(pass);

        if self.collect_sync_debug_info {
            let mut info = SyncDebugInfo::new();
            // current resource tracking info
            for (id, r) in self.context.resources.iter() {
                info.tracking.insert(id, r.tracking);
            }
            // current sync table
            info.xq_sync_table = self.xq_sync_table;
            self.sync_debug_info.push(info);
        }
    }

}


fn print_frame_info(context: &Context, passes: &[Pass], temporaries: &[ResourceId]) {
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
            let id = context.image_resource_by_handle(imb.image);
            print!("        image handle={:?} ", imb.image);
            if !id.is_null() {
                print!(
                    "(id={:?}, name={})",
                    id,
                    context.resources.get(id).unwrap().name
                );
            } else {
                print!("(unknown resource)");
            }
            println!(
                " access_mask:{:?}->{:?} layout:{:?}->{:?}",
                imb.src_access_mask, imb.dst_access_mask, imb.old_layout, imb.new_layout
            );
        }
        for bmb in p.buffer_memory_barriers.iter() {
            let id = context.buffer_resource_by_handle(bmb.buffer);
            print!("        buffer handle={:?} ", bmb.buffer);
            if !id.is_null() {
                print!(
                    "(id={:?}, name={})",
                    id,
                    context.resources.get(id).unwrap().name
                );
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
        let resource = context.resources.get(id).unwrap();
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
            ResourceOwnership::Referenced => {},
            ResourceOwnership::Owned {
                ref allocation, ..
            } => {
                match allocation {
                    Some(ResourceAllocation::Default { allocation }) => {
                        println!("    allocation: exclusive");
                    },
                    Some(ResourceAllocation::External { device_memory }) => {
                        println!("    allocation: external, device memory {:016x}", device_memory.as_raw());
                    },
                    Some(ResourceAllocation::Transient { device_memory, offset }) => {
                        println!("    allocation: transient, device memory {:016x}@{:016x}", device_memory.as_raw(), offset);
                    }
                    None => {
                        println!("    allocation: none (unallocated)");
                    }
                }
            }
        }
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
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
}

pub struct Frame<'a> {
    base_serial: u64,
    frame_serial: FrameSerialNumber,
    inner: RefCell<FrameInner<'a>>,
    span: tracing::span::EnteredSpan,
    build_span: tracing::span::EnteredSpan,
}

impl<'a> Frame<'a> {
    /// Returns the context from which this frame was started
    pub fn context(&self) -> RefMut<Context> {
        RefMut::map(self.inner.borrow_mut(), |inner| inner.context)
    }

    /// Returns this frame's serial
    pub fn serial(&self) -> FrameSerialNumber {
        self.frame_serial
    }

    /// Adds a dependency on a GPU future object.
    ///
    /// Execution of the *whole* frame will wait for the operation represented by the future to complete.
    pub fn add_frame_dependency(&self, future: GpuFuture) {
        let mut inner = self.inner.borrow_mut();
        inner.wait_init = inner.wait_init.join(future.serials);
    }

    pub fn add_graphics_pass(&self, name: &str, handler: impl FnOnce(&mut PassBuilder<'a, '_>)) {
        self.add_pass(name, PassType::Graphics, false, handler)
    }

    /// Starts building a compute pass
    pub fn add_compute_pass(
        &self,
        name: &str,
        async_compute: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassType::Compute, async_compute, handler)
    }

    /// Starts building a transfer pass
    pub fn add_transfer_pass(
        &self,
        name: &str,
        async_transfer: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassType::Transfer, async_transfer, handler)
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&self, name: &str, image: &SwapchainImage) {
        self.add_pass(
            name,
            PassType::Present {
                swapchain: image.swapchain_handle,
                image_index: image.image_index,
            },
            false,
            |builder| {
                builder.register_image_access(
                    image.image_info.id,
                    vk::AccessFlags::MEMORY_READ,
                    vk::PipelineStageFlags::ALL_COMMANDS, // ?
                    vk::ImageLayout::PRESENT_SRC_KHR,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            },
        );
    }

    /// Common code for `build_xxx_pass`
    fn add_pass(
        &self,
        name: &str,
        ty: PassType,
        async_pass: bool,
        handler: impl FnOnce(&mut PassBuilder<'a, '_>),
    ) {
        let mut inner = self.inner.borrow_mut();
        let frame_index = inner.passes.len();
        let serial = inner.context.get_next_serial();

        let q = match ty {
            PassType::Compute if async_pass => inner.context.device.queues_info.indices.compute,
            PassType::Transfer if async_pass => inner.context.device.queues_info.indices.transfer,
            PassType::Present { .. } => inner.context.device.queues_info.indices.present,
            _ => inner.context.device.queues_info.indices.graphics,
        } as usize;

        let snn = SubmissionNumber::new(q, serial);

        let pass = Pass::new(name, frame_index, snn);

        let mut builder = PassBuilder {
            frame: &mut *inner,
            pass,
            _span: trace_span!("add_pass", ?snn).entered(),
        };

        handler(&mut builder);

        let mut pass = builder.pass;

        // pass the swapchain and image index if this is a present pass
        // TODO: use a queue operation callback instead
        if let PassType::Present {
            swapchain,
            image_index,
        } = ty
        {
            pass.commands = Some(PassCommands::Present {
                swapchain,
                image_index,
            })
        }

        inner.push_pass(pass);
    }

    /// Dumps the frame to a JSON object.
    pub fn dump(&self, file_name_prefix: Option<&str>) {
        use serde_json::json;
        use std::fs::File;

        let inner = self.inner.borrow_mut();

        // passes
        let mut passes_json = Vec::new();
        for (pass_index, p) in inner.passes.iter().enumerate() {
            let image_memory_barriers_json: Vec<_> = p
                .image_memory_barriers
                .iter()
                .map(|imb| {
                    let id = inner.context.image_resource_by_handle(imb.image);
                    let name = &inner.context.resources.get(id).unwrap().name;
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
                    let id = inner.context.buffer_resource_by_handle(bmb.buffer);
                    let name = &inner.context.resources.get(id).unwrap().name;
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
                    let r = inner.context.resources.get(a.id).unwrap();
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
            if inner.collect_sync_debug_info {
                let sync_debug_info = &inner.sync_debug_info[pass_index];

                let mut resource_tracking_json = Vec::new();
                for (id, tracking) in sync_debug_info.tracking.iter() {
                    let name = &inner.context.resources.get(id).unwrap().name;
                    resource_tracking_json.push(json!({
                        "id": format!("{:?}", id.data()),
                        "name": name,
                        "readers": tracking.readers.0,
                        "writerQueue": tracking.writer.queue(),
                        "writerSerial": tracking.writer.serial(),
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
            "frameSerial": self.frame_serial.0,
            "baseSerial": self.base_serial,
            "passes": passes_json,
        });

        let file = File::create(format!(
            "{}-{}.json",
            file_name_prefix.unwrap_or("frame"),
            self.frame_serial.0
        ))
        .expect("could not open file for dumping JSON frame information");
        serde_json::to_writer_pretty(file, &frame_json).unwrap();
    }



    /// Finishes building the frame and submits all the passes to the command queues.
    pub fn finish(self) -> GpuFuture {
        // end build span
        self.build_span.exit();

        let mut inner = self.inner.into_inner();
        let context = inner.context;

        // First, wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the frames that are not in use anymore.
        context.wait_for_frames_in_flight();

        // Allocate and assign memory for all transient resources of this frame.
        let transient_allocations = allocate_memory_for_transients(context, inner.base_serial, &inner.passes, &inner.temporaries);


        // All resources now have a block of device memory assigned. We're ready to
        // build the command buffers and submit them to the device queues.
        let submission_result = context.submit_frame(&mut inner.passes, inner.wait_init);

        let serials = submission_result.signalled_serials;

        print_frame_info(context, &inner.passes, &inner.temporaries);

        // Add this frame to the list of "frames in flight": frames that might be executing on the device.
        // When this frame is completed, all resources of the frame will be automatically recycled.
        // This includes:
        // - device memory blocks for transient allocations
        // - command buffers (in command pools)
        // - image views
        // - framebuffers
        // - descriptor sets
        context.in_flight.push_back(FrameInFlight {
            signalled_serials: serials,
            transient_allocations,
            command_pools: submission_result.command_pools,
            semaphores: submission_result.semaphores,
        });

        context.submitted_frame_count += 1;
        context.dump_state();

        GpuFuture { serials }
    }
}

impl Context {
    /// Starts a new frame. The execution of the frame can optionally be synchronized
    /// with the given future in `happens_after`.
    ///
    /// However, regardless of this, individual passes in the frame may still synchronize with earlier frames
    /// because of resource dependencies.
    pub fn start_frame(&mut self, create_info: FrameCreateInfo) -> Frame {
        let base_serial = self.next_serial;

        let wait_init = create_info.happens_after.serials;

        // Full CPU-side frame processing
        let span = trace_span!("frame", base_serial).entered();
        // DAG build only
        let build_span = trace_span!("DAG build").entered();

        Frame {
            base_serial,
            frame_serial: FrameSerialNumber(self.submitted_frame_count + 1),
            inner: RefCell::new(FrameInner {
                base_serial,
                context: self,
                wait_init,
                temporaries: vec![],
                temporary_set: TemporarySet::new(),
                passes: vec![],
                xq_sync_table: Default::default(),
                collect_sync_debug_info: create_info.collect_debug_info,
                sync_debug_info: Vec::new(),
            }),
            span,
            build_span,
        }
    }
}
