//! Automatic synchronization.
use crate::{
    context::{
        local_pass_index, FrameCommand, Pass, ResourceAccess, ResourceAccessDetails, SemaphoreWait, SemaphoreWaitKind,
        SubmitInfo, SyncDebugInfo,
    },
    device::{
        AccessTracker, BufferResource, DeviceObjects, ImageResource, ResourceGroupMap, ResourceKind, ResourceMap,
    },
    is_write_access, BufferId, Context, Device, DeviceProgress, Frame, FrameNumber, ResourceGroupId, ResourceId,
    SubmissionNumber, MAX_QUEUES,
};
use ash::vk;
use std::mem;

type TemporarySet = std::collections::BTreeSet<ResourceId>;

/// Current autosync state.
struct SyncState {
    //frame_number: FrameNumber,
    base_sn: u64,
    current_sn: u64,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the frame
    temporary_set: TemporarySet,
    /// Serials to wait for before executing the frame.
    initial_wait: DeviceProgress,

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
    xq_sync_table: [DeviceProgress; MAX_QUEUES],

    collect_sync_debug_info: bool,
    sync_debug_info: Vec<SyncDebugInfo>,
    //descriptor_sets: Vec<vk::DescriptorSet>,
    //framebuffers: Vec
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
    state: &mut SyncState,
    passes: &mut [Pass<'a, UserContext>],
    dst_pass: &mut Pass<'a, UserContext>,
    sources: DeviceProgress,
    barrier: PipelineBarrierDesc,
) {
    let q = dst_pass.snn.queue();

    // from here, we have two possible methods of synchronization:
    // 1. semaphore signal/wait: this is straightforward, there's not a lot we can do
    // 2. pipeline barrier: we can choose **where** to put the barrier,
    //    we can put it anywhere between the source and the destination pass
    // -> if the source is just before, then use a pipeline barrier
    // -> if there's a gap, use an event?

    if !sources.is_single_source_same_queue_and_frame(q, state.base_sn) {
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
            if state.xq_sync_table[q].0[iq] >= sn {
                // already synced
                continue;
            }

            // we're adding a semaphore wait: update sync table
            state.xq_sync_table[q].0[iq] = sn;

            dst_pass.wait_serials.0[iq] = sn;
            dst_pass.wait_dst_stages[iq] |= barrier.dst_stage_mask;

            if sn > state.base_sn {
                let src_pass_index = local_pass_index(sn, state.base_sn);
                let src_pass = &mut passes[src_pass_index];
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
        if state.xq_sync_table[q][q] >= src_sn {
            // if we're already synchronized with the source via a cross-queue (xq) wait
            // (a.k.a. semaphore), we don't need to add a memory barrier.
            // Note that layout transitions are handled separately, outside this condition.
        } else {
            // not synced with a semaphore, see if there's already a pipeline barrier
            // that ensures the execution dependency between the source (src_sn) and us

            let local_src_index = local_pass_index(src_sn, state.base_sn);

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
            let barrier_pass = passes[local_src_index..]
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
                    let mb = barrier_pass.get_or_create_image_memory_barrier(resource.handle, resource.format);
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

/// Registers an access to a resource within the specified pass and updates the dependency graph.
///
/// This is the meat of the automatic synchronization system: given the known state of the resources,
/// this function infers the necessary execution barriers, memory barriers, and layout transitions,
/// and updates the state of the resources.
fn add_resource_dependency<'a, UserContext>(
    resources: &mut ResourceMap,
    state: &mut SyncState,
    passes: &mut [Pass<'a, UserContext>],
    dst_pass: &mut Pass<'a, UserContext>,
    access: &ResourceAccess,
) {
    let id = access.id;
    let resource = match resources.get_mut(id) {
        None => {
            panic!("invalid resource ID: {:?}", id)
        }
        Some(id) => id,
    };

    assert!(!resource.discarded, "referenced a discarded resource: {:?}", resource);
    // we can't synchronize on a resource that belongs to a group: we synchronize on the group instead
    assert!(
        resource.group.is_none(),
        "cannot synchronize on a resource belonging to a group; synchronize on the group instead"
    );

    //------------------------
    // first, add the resource into the set of temporaries used within this frame
    if state.temporary_set.insert(id) {
        state.temporaries.push(id);
    }

    // if the resource has not been accessed yet, set the first access field
    if resource.tracking.first_access.is_none() {
        resource.tracking.first_access = Some(AccessTracker::Device(dst_pass.snn));
    }

    // Definitions:
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

    let need_layout_transition = resource.tracking.layout != access.initial_layout;

    // is the access a write? for synchronization purposes, layout transitions are the same thing as a write
    let is_write = is_write_access(access.access_mask) || need_layout_transition;

    // can we ensure that all previous writes are visible?
    let writes_visible = match resource.tracking.writer {
        None => {
            // no previous writer recorded on the resource, so no writes, no data to see, and no barrier necessary.
            true
        }
        Some(AccessTracker::Host) => {
            // the last write was from the host, the spec says that there's no need for a barrier for host writes
            // https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#synchronization-submission-host-writes
            true
        }
        Some(AccessTracker::Device(writer)) => {
            // the visibility mask is only valid if this access and the last write is in the same queue
            // for cross-queue accesses, we never skip
            writer.queue() == dst_pass.snn.queue()
                && (resource.tracking.visibility_mask.contains(access.access_mask)
                    || resource.tracking.visibility_mask.contains(vk::AccessFlags::MEMORY_READ))
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
                    DeviceProgress::default()
                }
                Some(AccessTracker::Device(writer)) => DeviceProgress::from_submission_number(writer),
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
            state,
            passes,
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

        if sync_sources.is_single_source_same_queue_and_frame(q, state.base_sn) {
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
}

fn add_resource_to_group(
    resources: &mut ResourceMap,
    resource_groups: &mut ResourceGroupMap,
    resource_id: ResourceId,
    group_id: ResourceGroupId,
) {
    let group = resource_groups
        .get_mut(group_id)
        .expect("invalid or expired resource group");
    let mut resource = resources.get_mut(resource_id).expect("invalid resource");
    assert!(resource.group.is_none());

    // set group
    resource.group = Some(group_id);
    // set additional serials and stages to wait for in this group
    match resource.tracking.writer {
        Some(AccessTracker::Device(writer)) => group
            .wait_serials
            .join_assign(DeviceProgress::from_submission_number(writer)),
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

/// Registers an access to a resource group.
fn add_group_dependency<'a, UserContext>(
    state: &mut SyncState,
    resource_groups: &mut ResourceGroupMap,
    passes: &mut [Pass<'a, UserContext>],
    dst_pass: &mut Pass<'a, UserContext>,
    id: ResourceGroupId,
) {
    // we just have to wait for the SNNs and stages of the group.
    let group = resource_groups.get(id).expect("invalid or expired resource group");

    add_memory_dependency(
        state,
        passes,
        dst_pass,
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

/// Computes the necessary barriers between passes in a frame, and updates the last known resource states.
///
/// # Results
/// * the synchronization fields of `Pass`es are initialized
/// * the resource tracking information in `resources` are updated.
pub(super) fn synchronize_and_track_resources<'a, UserContext>(
    device: &Device,
    resources: &mut ResourceMap,
    resource_groups: &mut ResourceGroupMap,
    frame: &mut Frame<'a, UserContext>,
    base_sn: u64,
    initial_wait: DeviceProgress,
) -> u64 {
    // Play back the recorded frame, updating the resource states as we go
    let mut sync_state = SyncState {
        base_sn,
        current_sn: base_sn,
        //frame_number,
        initial_wait,
        temporaries: vec![],
        temporary_set: TemporarySet::new(),
        xq_sync_table: Default::default(),
        collect_sync_debug_info: false,
        sync_debug_info: Vec::new(),
    };

    let num_passes = frame.passes.len();
    let mut i_cmd = 0;
    for i in 0..num_passes {
        // execute pre-pass commands
        loop {
            if i_cmd < frame.commands.len() && frame.commands[i_cmd].0 == i {
                let (_, cmd) = frame.commands[i_cmd];
                match cmd {
                    FrameCommand::FreezeResource { resource, group } => {
                        add_resource_to_group(resources, resource_groups, resource, group);
                    }
                    FrameCommand::DestroyImage { image } => {
                        resources.get_mut(image.0).unwrap().discarded = true;
                    }
                    FrameCommand::DestroyBuffer { buffer } => {
                        resources.get_mut(buffer.0).unwrap().discarded = true;
                    }
                }
                i_cmd += 1;
            } else {
                break;
            }
        }

        let (prev_passes, next_passes) = frame.passes.split_at_mut(i);
        let pass = &mut next_passes[0];

        // assign a SN to the pass
        sync_state.current_sn += 1;
        let serial = sync_state.current_sn;
        let q = pass.ty.queue_index(&device, pass.async_queue);
        pass.snn = SubmissionNumber::new(q, serial);

        // track resource states
        let accesses = mem::take(&mut pass.accesses);
        for access in accesses.iter() {
            add_resource_dependency(resources, &mut sync_state, prev_passes, pass, access);
        }
    }

    sync_state.current_sn
}
