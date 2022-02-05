//! Allocation of memory for transient resources in a frame.
use crate::{
    ash::vk,
    context::{local_pass_index, Pass},
    resource::{AccessTracker, Resource, ResourceAllocation, ResourceKind},
    AllocationRequirements, Context, ResourceId, ResourceOwnership,
};
use fixedbitset::FixedBitSet;
use slotmap::SecondaryMap;
use tracing::trace_span;

// --- Reachability matrix -------------------------------------------------------------------------

/// Returns whether stage a comes logically earlier than stage b.
#[rustfmt::skip]
fn logically_earlier(a: vk::PipelineStageFlags, b: vk::PipelineStageFlags) -> bool {
    const DI : vk::Flags64 = vk::PipelineStageFlags::DRAW_INDIRECT.as_raw() as u64;
    const II : vk::Flags64 = vk::PipelineStageFlags2KHR::INDEX_INPUT.as_raw() as u64;
    const VAI : vk::Flags64 = vk::PipelineStageFlags2KHR::VERTEX_ATTRIBUTE_INPUT.as_raw() as u64;
    const VS : vk::Flags64 = vk::PipelineStageFlags::VERTEX_SHADER.as_raw() as u64;
    const TCS : vk::Flags64 = vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER.as_raw() as u64;
    const TES : vk::Flags64 = vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw() as u64;
    const GS : vk::Flags64 = vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw() as u64;
    const TF : vk::Flags64 = vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT.as_raw() as u64;
    const FSR : vk::Flags64 = vk::PipelineStageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw() as u64;
    const EFT : vk::Flags64 = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw() as u64;
    const FS : vk::Flags64 = vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw() as u64;
    const LFT : vk::Flags64 = vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw() as u64;
    const CAO : vk::Flags64 = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw() as u64;
    const CS : vk::Flags64 = vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() as u64;
    const TS : vk::Flags64 = 0x00080000; // VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
    const MS : vk::Flags64 = 0x00100000; // VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV
    const TR : vk::Flags64 = vk::PipelineStageFlags::TRANSFER.as_raw() as u64;

    fn test(flags: vk::Flags64, mask: vk::Flags64) -> bool { (flags & mask) != 0 }

    let a = a.as_raw() as u64;
    let b = b.as_raw() as u64;

    match a {
        // draw & compute pipeline ordering
        DI  => test(b, DI | CS | TS | MS | II | VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        CS  => test(b,      CS                                                                             ),
        TS  => test(b,           TS | MS                                       | FSR | EFT | FS | LFT | CAO),
        MS  => test(b,                MS                                       | FSR | EFT | FS | LFT | CAO),
        II  => test(b,                     II | VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        VAI => test(b,                          VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        VS  => test(b,                                VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        TCS => test(b,                                     TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        TES => test(b,                                           TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        GS  => test(b,                                                 GS | TF | FSR | EFT | FS | LFT | CAO),
        TF  => test(b,                                                      TF | FSR | EFT | FS | LFT | CAO),
        FSR => test(b,                                                           FSR | EFT | FS | LFT | CAO),
        EFT => test(b,                                                                 EFT | FS | LFT | CAO),
        FS  => test(b,                                                                       FS | LFT | CAO),
        LFT => test(b,                                                                            LFT | CAO),
        CAO => test(b,                                                                                  CAO),
        // transfer
        TR  => test(b, TR),
        _ => false,
    }
}

// --- Reachability matrix -------------------------------------------------------------------------
struct Reachability {
    m: Vec<FixedBitSet>,
}

impl Reachability {
    fn is_reachable(&self, from: usize, to: usize) -> bool {
        self.m[to][from]
    }
}

fn disjoint_index_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b && a < v.len() && b < v.len());
    unsafe {
        (
            &mut *(v.get_unchecked_mut(a) as *mut _),
            &mut *(v.get_unchecked_mut(b) as *mut _),
        )
    }
}

fn compute_reachability<UserContext>(passes: &[Pass<UserContext>]) -> Reachability {
    let len = passes.len();
    let mut m = Vec::new();
    m.resize_with(passes.len(), || FixedBitSet::with_capacity(len));

    for i in 0..len {
        for &p in passes[i].preds.iter() {
            m[i].set(p, true);
            let (mi, mp) = disjoint_index_mut(&mut m, i, p);
            *mi |= &*mp;
        }
    }

    Reachability { m }
}

type AllocMap = SecondaryMap<ResourceId, SharedAllocEntry>;

/// Index of an allocation
#[derive(Copy, Clone, Debug)]
struct SharedAllocEntry {
    /// Index of the allocation.
    index: usize,
    /// Whether this entry has already been reused.
    dead_and_recycled: bool,
}

unsafe fn bind_resource_memory(
    device: &ash::Device,
    resource: &Resource,
    device_memory: vk::DeviceMemory,
    offset: vk::DeviceSize,
) {
    match &resource.kind {
        ResourceKind::Image(img) => {
            device
                .bind_image_memory(img.handle, device_memory, offset)
                .unwrap();
        }
        ResourceKind::Buffer(buf) => {
            device
                .bind_buffer_memory(buf.handle, device_memory, offset)
                .unwrap();
        }
    }
}

/// Allocates memory for the resources specified in `temporaries`.
/// If a resource is not used anymore, it might share its memory with another (aliasing).
// FIXME: this is broken and wrong, replace with v2; it's not currently used anyway
pub(crate) fn allocate_memory_for_transients<UserContext>(
    context: &mut Context,
    base_serial: u64,
    passes: &[Pass<UserContext>],
    temporaries: &[ResourceId],
) -> Vec<gpu_allocator::vulkan::Allocation> {
    let _span = trace_span!("allocate_memory_for_transients").entered();

    let reachability = compute_reachability(&passes);
    let mut resources = context
        .device
        .objects
        .lock()
        .expect("could not lock resources");

    //------ shared alloc state------
    // For each transient resource, its shared allocation index.
    let mut shared_alloc_map: SecondaryMap<ResourceId, SharedAllocEntry> = Default::default();
    // The requirements of each shared allocation.
    let mut shared_alloc_requirements: Vec<AllocationRequirements> = Vec::new();

    fn get_allocation_requirements(resource: &Resource) -> Option<AllocationRequirements> {
        match &resource.ownership {
            ResourceOwnership::External => {
                // skip non-owned resources
                None
            }
            ResourceOwnership::OwnedResource {
                requirements,
                allocation,
            } => {
                if allocation.is_some() {
                    // skip already allocated resources
                    None
                } else {
                    Some(*requirements)
                }
            }
        }
    }

    for &id in temporaries.iter() {
        // TARGET = the resource we want to alias with
        let resource = resources.resources.get(id).unwrap();

        let allocation_requirements = if let Some(req) = get_allocation_requirements(resource) {
            req
        } else {
            continue;
        };

        if resource.discarded {
            //--------------------------------------------------------------------------------------
            // Aliasing
            // the resource is marked discarded, which means that the user does not care about
            // the resource anymore: we are free to alias its memory block with something else

            // try to find a suitable resource to alias with (the "target")
            let mut aliased = false;
            'alias: for &target_id in temporaries.iter() {
                if target_id == id {
                    // don't alias with the same resource...
                    continue;
                }

                let target_resource = resources.resources.get(target_id).unwrap();

                // get the index of the shared allocation for this resource, or skip if not allocated
                let (target_alloc_entry, target_requirements) =
                    if let Some(entry) = shared_alloc_map.get_mut(target_id) {
                        if entry.dead_and_recycled {
                            continue;
                        } else {
                            let index = entry.index;
                            (entry, &mut shared_alloc_requirements[index])
                        }
                    } else {
                        continue;
                    };

                // Check that the allocation requirements of the two resources can be made compatible.
                let target_adjusted_requirements = if let Some(req) =
                    target_requirements.adjusted_requirements(&allocation_requirements)
                {
                    // yep
                    req
                } else {
                    // nope, skip this resource
                    continue;
                };

                // ----- now for the complicated part, we need to check that the pass that first accesses
                // the resource does not overlap with the passes that access the target resource

                let src_first_access = match target_resource.tracking.first_access {
                    None => {
                        // the resource is never accessed? which means that in theory we can alias
                        // it with anything
                        tracing::warn!(
                            "aliasable resource {:?}({}) is never accessed by the device",
                            target_id,
                            target_resource.name
                        );
                        0
                    }
                    Some(AccessTracker::Device(snn)) => snn.serial(),
                    Some(AccessTracker::Host) => {
                        // trying to alias the memory of a host-visible resource: not possible
                        panic!("tried to alias the memory of a host-visible resource")
                    }
                };

                // To re-use the memory of SRC in DST, SRC must be _dead_ before the first use of DST.
                // A resource is dead from the point of view of a pass if this pass has an execution
                // dependency on all last readers and writers of the resource.
                for &reader in target_resource.tracking.readers.iter() {
                    if reader != 0
                        && (reader >= src_first_access
                            || !reachability.is_reachable(
                                local_pass_index(reader, base_serial),
                                local_pass_index(src_first_access, base_serial),
                            ))
                    {
                        continue 'alias;
                    }
                }

                let writer = match target_resource.tracking.writer {
                    None => 0,
                    Some(AccessTracker::Device(writer)) => writer.serial(),
                    Some(AccessTracker::Host) => {
                        // don't alias with a host-visible resource
                        // FIXME should we even be able to reach this line?
                        continue 'alias;
                    }
                };
                if writer != 0
                    && (writer >= src_first_access
                        || !reachability.is_reachable(
                            local_pass_index(writer, base_serial),
                            local_pass_index(src_first_access, base_serial),
                        ))
                {
                    continue;
                }

                // if we reach here, then SRC is dead, and from a synchronization point of view
                // the resources may alias.

                // success: the two resources may alias, and the memory requirements have been adjusted
                // now update the allocation map
                target_alloc_entry.dead_and_recycled = true;
                let target_alloc_index = target_alloc_entry.index;
                *target_requirements = target_adjusted_requirements;
                shared_alloc_map.insert(
                    id,
                    SharedAllocEntry {
                        index: target_alloc_index,
                        dead_and_recycled: false,
                    },
                );

                aliased = true;
                break;
            }

            if !aliased {
                // we could not alias with any existing resource, so create a new allocation for the resource
                let index = shared_alloc_requirements.len();
                shared_alloc_requirements.push(allocation_requirements);
                shared_alloc_map.insert(
                    id,
                    SharedAllocEntry {
                        index,
                        dead_and_recycled: false,
                    },
                );
            }
        } else {
            //--------------------------------------------------------------------------------------
            // Exclusive
            // The resource is not marked discarded, which means that we must preserve the contents
            // of the resource: don't alias its memory

            let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
                name: "",
                requirements: allocation_requirements.mem_req,
                location: allocation_requirements.location,
                linear: false,
            };
            let allocation = context
                .device
                .allocator
                .lock()
                .unwrap()
                .allocate(&allocation_create_desc)
                .expect("failed to allocate device memory");

            unsafe {
                bind_resource_memory(
                    context.vulkan_device(),
                    resource,
                    allocation.memory(),
                    allocation.offset(),
                );
            }

            resources
                .resources
                .get_mut(id)
                .unwrap()
                .set_allocation(ResourceAllocation::Default { allocation });
        }
    }

    // now allocate each entry in the shared allocation map
    let mut shared_allocations = Vec::with_capacity(shared_alloc_requirements.len());

    for req in shared_alloc_requirements.iter() {
        let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name: "",
            location: req.location,
            requirements: req.mem_req,
            linear: false, // FIXME
        };
        let allocation = context
            .device
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        shared_allocations.push(allocation);
    }

    // finally, bind the shared allocations to the corresponding resources

    for &id in temporaries.iter() {
        let resource = resources.resources.get_mut(id).unwrap();

        let alloc_index = if let Some(entry) = shared_alloc_map.get(id) {
            entry.index
        } else {
            // the memory for the resource is not shareable
            continue;
        };

        let alloc = &shared_allocations[alloc_index];

        unsafe {
            bind_resource_memory(
                &context.device.device,
                resource,
                alloc.memory(),
                alloc.offset(),
            );
        }

        resource.set_allocation(ResourceAllocation::Transient {
            device_memory: unsafe { alloc.memory() },
            offset: alloc.offset(),
        })
    }

    shared_allocations
}
