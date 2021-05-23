use crate::{
    ash::{version::DeviceV1_0, vk},
    context::{
        local_pass_index, pass::Pass, AllocationRequirements, Resource, ResourceAllocation,
        ResourceKind, ResourceOwnership,
    },
    Context, ResourceId,
};
use fixedbitset::FixedBitSet;
use slotmap::SecondaryMap;
use tracing::trace_span;

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

fn compute_reachability(passes: &[Pass]) -> Reachability {
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

fn allocate_or_alias_memory(
    context: &mut Context,
    base_serial: u64,
    passes: &[Pass],
    temporaries: &[ResourceId],
    allocation_requirements: &mut Vec<AllocationRequirements>,
    allocation_map: &mut AllocMap,
) {
}

/// Index of an allocation
#[derive(Copy, Clone, Debug)]
struct SharedAllocEntry {
    /// Index of the allocation.
    index: usize,
    /// Whether this entry has already been reused.
    dead_and_recycled: bool,
}

/// Used to construct a set of allocation requirements for transient resources.
///
/// Allocations are identified by index, and multiple resources can share the same allocation.
/// Each allocation has "requirements" `AllocationRequirements`, which are updated progressively
/// as `assign_allocation_to_resource` is called.
struct SharedAllocationsBuilder {
    /// For each resource, the assigned allocation index.
    entries: SecondaryMap<ResourceId, SharedAllocEntry>,

    /// The requirements of each allocation.
    requirements: Vec<AllocationRequirements>,
}

impl SharedAllocationsBuilder {
    fn new() -> SharedAllocationsBuilder {
        SharedAllocationsBuilder {
            entries: Default::default(),
            requirements: vec![],
        }
    }

    /// Returns the index of the allocation for the resource.
    fn get_alloc_requirements(&mut self, id: ResourceId) -> Option<&mut AllocationRequirements> {
        if let Some(entry) = self.entries.get(id) {
            if entry.dead_and_recycled {
                None
            } else {
                Some(&mut self.requirements[entry.index])
            }
        } else {
            None
        }
    }

    /*/// Marks the foll
    fn mark_as_reused(&mut self, id: ResourceId) {
        *self.entries.get_mut(id).unwrap().dead_and_recycled = true;
    }*/
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
pub(crate) fn allocate_memory_for_transients(
    context: &mut Context,
    base_serial: u64,
    passes: &[Pass],
    temporaries: &[ResourceId],
) -> Vec<vk_mem::Allocation> {
    let _span = trace_span!("allocate_memory_for_transients").entered();

    let reachability = compute_reachability(&passes);

    //------ shared alloc state------
    // For each transient resource, its shared allocation index.
    let mut shared_alloc_map: SecondaryMap<ResourceId, SharedAllocEntry> = Default::default();
    // The requirements of each shared allocation.
    let mut shared_alloc_requirements: Vec<AllocationRequirements> = Vec::new();

    fn get_allocation_requirements(resource: &Resource) -> Option<AllocationRequirements> {
        match &resource.ownership {
            ResourceOwnership::Referenced => {
                // skip non-owned resources
                None
            }
            ResourceOwnership::Owned {
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
        let resource = context.resources.get(id).unwrap();

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
                let target_resource = context.resources.get(id).unwrap();
                if target_id == id {
                    // don't alias with the same resource...
                    continue;
                }

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

                let src_first_access = target_resource.tracking.first_access.serial();

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

                let writer = target_resource.tracking.writer.serial();
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

            let allocation_create_info = vk_mem::AllocationCreateInfo {
                required_flags: allocation_requirements.required_flags,
                preferred_flags: allocation_requirements.preferred_flags,
                ..Default::default()
            };
            let (allocation, allocation_info) = context
                .device
                .allocator
                .allocate_memory(&allocation_requirements.mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");

            unsafe {
                bind_resource_memory(
                    context.vulkan_device(),
                    resource,
                    allocation_info.get_device_memory(),
                    allocation_info.get_offset() as u64,
                );
            }

            context.resources.get_mut(id).unwrap().set_allocation(ResourceAllocation::Default {
                allocation
            });
        }
    }

    // now allocate each entry in the shared allocation map
    let mut shared_allocations = Vec::with_capacity(shared_alloc_requirements.len());
    let mut shared_allocation_infos = Vec::with_capacity(shared_alloc_requirements.len());

    for req in shared_alloc_requirements.iter() {
        let allocation_create_info = vk_mem::AllocationCreateInfo {
            required_flags: req.required_flags,
            preferred_flags: req.preferred_flags,
            ..Default::default()
        };
        let (allocation, allocation_info) = context
            .device
            .allocator
            .allocate_memory(&req.mem_req, &allocation_create_info)
            .expect("failed to allocate device memory");
        shared_allocations.push(allocation);
        shared_allocation_infos.push(allocation_info);
    }

    // finally, bind the shared allocations to the corresponding resources
    for &id in temporaries.iter() {
        let resource = context.resources.get_mut(id).unwrap();

        let alloc_index = if let Some(entry) = shared_alloc_map.get(id) {
            entry.index
        } else {
            // the memory for the resource is not shareable
            continue;
        };

        let alloc_info = &shared_allocation_infos[alloc_index];

        unsafe {
            bind_resource_memory(
                &context.device.device,
                resource,
                alloc_info.get_device_memory(),
                alloc_info.get_offset() as u64,
            );
        }

        resource.set_allocation(ResourceAllocation::Transient {
            device_memory: alloc_info.get_device_memory(),
            offset: alloc_info.get_offset() as u64,
        })
    }

    shared_allocations
}
