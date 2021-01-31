use crate::device::Device;
use crate::handle::{UniqueHandle, UniqueHandleVec};
use crate::pass::{Pass, PassKind, ResourceAccess, SubmissionNumber};
use crate::MAX_QUEUES;
use crate::VULKAN_INSTANCE;
use ash::version::DeviceV1_0;
use ash::vk;
use core::ptr;
use fixedbitset::FixedBitSet;
use slotmap::{new_key_type, Key, SecondaryMap, SlotMap};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::mem;
use std::os::raw::c_void;
use std::rc::Rc;

fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => panic!("unsupported number of samples"),
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ImageProperties {
    pub mip_levels: u32,
    pub array_layers: u32,
    pub samples: u32,
    pub tiling: vk::ImageTiling,
}

fn get_mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

fn is_read_access(mask: vk::AccessFlags) -> bool {
    mask.intersects(
        vk::AccessFlags::INDIRECT_COMMAND_READ
            | vk::AccessFlags::INDEX_READ
            | vk::AccessFlags::VERTEX_ATTRIBUTE_READ
            | vk::AccessFlags::UNIFORM_READ
            | vk::AccessFlags::INPUT_ATTACHMENT_READ
            | vk::AccessFlags::SHADER_READ
            | vk::AccessFlags::COLOR_ATTACHMENT_READ
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
            | vk::AccessFlags::TRANSFER_READ
            | vk::AccessFlags::HOST_READ
            | vk::AccessFlags::MEMORY_READ
            | vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_READ_EXT
            | vk::AccessFlags::CONDITIONAL_RENDERING_READ_EXT
            | vk::AccessFlags::COLOR_ATTACHMENT_READ_NONCOHERENT_EXT
            | vk::AccessFlags::ACCELERATION_STRUCTURE_READ_KHR
            | vk::AccessFlags::SHADING_RATE_IMAGE_READ_NV
            | vk::AccessFlags::FRAGMENT_DENSITY_MAP_READ_EXT
            | vk::AccessFlags::COMMAND_PREPROCESS_READ_NV,
    )
}

fn is_write_access(mask: vk::AccessFlags) -> bool {
    mask.intersects(
        vk::AccessFlags::SHADER_WRITE
            | vk::AccessFlags::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags::TRANSFER_WRITE
            | vk::AccessFlags::HOST_WRITE
            | vk::AccessFlags::MEMORY_WRITE
            | vk::AccessFlags::TRANSFORM_FEEDBACK_WRITE_EXT
            | vk::AccessFlags::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
            | vk::AccessFlags::ACCELERATION_STRUCTURE_WRITE_KHR
            | vk::AccessFlags::COMMAND_PREPROCESS_WRITE_NV,
    )
}

fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM_S8_UINT => true,
        vk::Format::D24_UNORM_S8_UINT => true,
        vk::Format::D32_SFLOAT_S8_UINT => true,
        _ => false,
    }
}

fn is_depth_only_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::D16_UNORM => true,
        vk::Format::X8_D24_UNORM_PACK32 => true,
        vk::Format::D32_SFLOAT => true,
        _ => false,
    }
}

fn is_stencil_only_format(fmt: vk::Format) -> bool {
    match fmt {
        vk::Format::S8_UINT => true,
        _ => false,
    }
}

fn format_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_only_format(fmt) {
        vk::ImageAspectFlags::DEPTH
    } else if is_stencil_only_format(fmt) {
        vk::ImageAspectFlags::STENCIL
    } else if is_depth_and_stencil_format(fmt) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

#[derive(Copy, Clone, Debug)]
struct AllocationRequirements {
    mem_req: vk::MemoryRequirements,
    required_flags: vk::MemoryPropertyFlags,
    preferred_flags: vk::MemoryPropertyFlags,
}

impl AllocationRequirements {
    fn try_adjust(&mut self, other: &AllocationRequirements) -> bool {
        if self.required_flags != other.required_flags {
            return false;
        }
        if self.mem_req.memory_type_bits != other.mem_req.memory_type_bits {
            return false;
        }
        self.mem_req.alignment = self.mem_req.alignment.max(other.mem_req.alignment);
        self.mem_req.size = self.mem_req.size.max(other.mem_req.size);
        true
    }
}

new_key_type! {
    pub struct ResourceId;
}

struct ImageResource {
    handle: UniqueHandle<vk::Image>,
    format: vk::Format,
}

struct BufferResource {
    handle: UniqueHandle<vk::Buffer>,
}

/// Represents a resource access in a pass.
pub(crate) struct ResourceAccessDetails {
    layout: vk::ImageLayout,
    access_mask: vk::AccessFlags,
    input_stage: vk::PipelineStageFlags,
    output_stage: vk::PipelineStageFlags,
}

enum ResourceKind {
    Buffer(BufferResource),
    Image(ImageResource),
}

struct ResourceTrackingInfo {
    readers: [u64; MAX_QUEUES],
    writer: SubmissionNumber,
    layout: vk::ImageLayout,
    availability_mask: vk::AccessFlags,
    visibility_mask: vk::AccessFlags,
    stages: vk::PipelineStageFlags,
    wait_binary_semaphore: UniqueHandle<vk::Semaphore>,
}

impl ResourceTrackingInfo {
    fn has_writer(&self) -> bool {
        self.writer.is_valid()
    }

    fn has_readers(&self) -> bool {
        self.readers.iter().any(|&x| x != 0)
    }

    fn clear_readers(&mut self) {
        self.readers = [0; MAX_QUEUES];
    }
}

impl Default for ResourceTrackingInfo {
    fn default() -> Self {
        ResourceTrackingInfo {
            readers: [0; 4],
            writer: Default::default(),
            layout: Default::default(),
            availability_mask: Default::default(),
            visibility_mask: Default::default(),
            stages: Default::default(),
            wait_binary_semaphore: UniqueHandle::null(),
        }
    }
}

struct Resource {
    name: String,
    user_ref_count: usize,
    owned: bool,
    allocation_requirements: AllocationRequirements,
    allocation: vk_mem::Allocation,
    allocation_info: Option<vk_mem::AllocationInfo>,
    tracking: ResourceTrackingInfo,
    tmp_index: Option<usize>,
    kind: ResourceKind,
}

/// Adds an execution dependency between a source and destination pass, identified by their submission numbers.
fn add_execution_dependency(
    src_snn: SubmissionNumber,
    src: Option<&mut Pass>,
    dst: &mut Pass,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    if let Some(src) = src {
        // --- Intra-batch synchronization
        if src_snn.queue() != dst.snn.queue() {
            // cross-queue dependency w/ timeline semaphore
            src.signal_after = true;
            let q = src_snn.queue() as usize;
            dst.wait_before = true;
            dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
            dst.wait_dst_stages[q] |= dst_stage_mask;
        } else {
            // same-queue dependency, a pipeline barrier is sufficient
            dst.src_stage_mask |= src.output_stage_mask;
        }

        dst.preds.push(src.batch_index);
        src.succs.push(dst.batch_index);
    } else {
        // --- Inter-batch synchronization w/ timeline semaphore
        let q = src_snn.queue() as usize;
        dst.wait_before = true;
        dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
        dst.wait_dst_stages[q] |= dst_stage_mask;
    }
}

type TemporarySet = std::collections::BTreeSet<ResourceId>;

fn register_temporary<'a>(
    resources: &'a mut ResourceMap,
    temporaries: &mut Vec<ResourceId>,
    temporary_set: &mut TemporarySet,
    id: ResourceId,
) -> &'a mut Resource {
    if temporary_set.insert(id) {
        //let index = temporaries.len();
        temporaries.push(id);
    }

    let resource = resources.get_mut(id).unwrap();
    resource
}

///
fn disjoint_index_mut<T>(v: &mut [T], a: usize, b: usize) -> (&mut T, &mut T) {
    assert!(a != b && a < v.len() && b < v.len());
    unsafe {
        (
            &mut *(v.get_unchecked_mut(a) as *mut _),
            &mut *(v.get_unchecked_mut(b) as *mut _),
        )
    }
}

struct Reachability {
    m: Vec<FixedBitSet>,
}

impl Reachability {
    fn is_reachable(&self, from: usize, to: usize) -> bool {
        self.m[to][from]
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

pub struct Batch<'ctx> {
    start_serial: u64,
    context: &'ctx mut Context,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the batch
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass>,
}

impl<'ctx> Batch<'ctx> {
    fn new(context: &'ctx mut Context) -> Batch<'ctx> {
        Batch {
            start_serial: context.next_serial,
            context,
            temporaries: vec![],
            temporary_set: TemporarySet::new(),
            passes: vec![],
        }
    }

    pub fn build_render_pass<'a>(&'a mut self, name: &str) -> PassBuilder<'ctx, 'a> {
        let queues_info = self.context.device.queues_info;
        self.build_next_pass(name, queues_info.indices.graphics, PassKind::Render)
    }

    pub fn build_compute_pass<'a>(
        &'a mut self,
        name: &str,
        async_compute: bool,
    ) -> PassBuilder<'ctx, 'a> {
        let queues_info = self.context.device.queues_info;
        let queue_index = if async_compute {
            queues_info.indices.compute
        } else {
            queues_info.indices.graphics
        };
        self.build_next_pass(name, queue_index, PassKind::Compute)
    }

    pub fn build_transfer_pass<'a>(
        &'a mut self,
        name: &str,
        async_transfer: bool,
    ) -> PassBuilder<'ctx, 'a> {
        let queues_info = self.context.device.queues_info;
        let queue_index = if async_transfer {
            queues_info.indices.transfer
        } else {
            queues_info.indices.graphics
        };
        self.build_next_pass(name, queue_index, PassKind::Transfer)
    }

    fn build_next_pass<'a>(
        &'a mut self,
        name: &str,
        queue_index: u8,
        kind: PassKind,
    ) -> PassBuilder<'ctx, 'a> {
        let serial = self.context.get_next_serial();
        let batch_index = self.passes.len();
        let snn = SubmissionNumber::new(queue_index, serial);

        PassBuilder {
            batch: self,
            pass: Pass::new(name, batch_index, snn, kind),
        }
    }

    /// Called by `PassBuilder::finish`.
    fn finish_pass(&mut self, pass: Pass) {
        self.passes.push(pass)
    }

    /// Helper to find the pass given a submission number.
    fn get_pass_mut(
        start_serial: u64,
        passes: &mut [Pass],
        snn: SubmissionNumber,
    ) -> Option<&mut Pass> {
        if snn.serial() <= start_serial {
            None
        } else {
            let pass_index = (snn.serial() - start_serial - 1) as usize;
            Some(&mut passes[pass_index])
        }
    }

    ///
    fn add_resource_dependency(
        &mut self,
        pass: &mut Pass,
        id: ResourceId,
        access: &ResourceAccessDetails,
    ) {
        let resource = register_temporary(
            &mut self.context.resources,
            &mut self.temporaries,
            &mut self.temporary_set,
            id,
        );

        //let pass_index = (snn.serial() - self.start_serial - 1) as usize;
        //let old_layout = resource.tracking.layout;
        //let src_access_mask = resource.tracking.availability_mask;
        let is_write = !access.output_stage.is_empty() || resource.tracking.layout != access.layout;

        // update input stage mask
        pass.input_stage_mask |= access.input_stage;

        // handle external semaphore dependency
        let semaphore = mem::take(&mut resource.tracking.wait_binary_semaphore);
        if !semaphore.is_null() {
            pass.wait_binary_semaphores.push(semaphore);
            pass.wait_before = true;
        }

        if is_write {
            if !resource.tracking.has_readers() && resource.tracking.has_writer() {
                // write-after-write
                add_execution_dependency(
                    resource.tracking.writer,
                    Self::get_pass_mut(
                        self.start_serial,
                        &mut self.passes,
                        resource.tracking.writer,
                    ),
                    pass,
                    access.input_stage,
                );
            } else {
                // write-after-read
                for q in 0..MAX_QUEUES {
                    if resource.tracking.readers[q] != 0 {
                        let src_snn = SubmissionNumber::new(q as u8, resource.tracking.readers[q]);
                        add_execution_dependency(
                            src_snn,
                            Self::get_pass_mut(self.start_serial, &mut self.passes, src_snn),
                            pass,
                            access.input_stage,
                        );
                    }
                }
            }
            // update the resource writer
            resource.tracking.clear_readers();
            resource.tracking.writer = pass.snn;
            pass.output_stage_mask = access.output_stage;
        } else {
            if resource.tracking.has_writer() {
                // read-after-write
                // NOTE a read without a write is probably an uninitialized access
                add_execution_dependency(
                    resource.tracking.writer,
                    Self::get_pass_mut(
                        self.start_serial,
                        &mut self.passes,
                        resource.tracking.writer,
                    ),
                    pass,
                    access.input_stage,
                );
            }
            let q = pass.snn.queue() as usize;
            // update the resource readers
            resource.tracking.readers[q] = resource.tracking.readers[q].max(pass.snn.serial());
        }

        // --- memory barriers

        // are all writes to the resource visible to the requested access type?
        let writes_visible = resource
            .tracking
            .visibility_mask
            .contains(vk::AccessFlags::MEMORY_READ)
            || resource
                .tracking
                .visibility_mask
                .contains(access.access_mask);
        // is the layout of the resource different? do we need a transition?
        let layout_transition = resource.tracking.layout != access.layout;
        // is there a possible write-after-write hazard, that requires a memory dependency?
        let write_after_write_hazard =
            is_write && is_write_access(resource.tracking.availability_mask);

        if !writes_visible || layout_transition || write_after_write_hazard {
            // no need to make memory visible if we're only writing to the resource
            let dst_access_mask = if !is_read_access(access.access_mask) {
                Default::default()
            } else {
                access.access_mask
            };
            // the resource access needs a memory barrier
            match &resource.kind {
                ResourceKind::Image(img) => {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: format_aspect_mask(img.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    };

                    pass.image_memory_barriers.push(vk::ImageMemoryBarrier {
                        src_access_mask: resource.tracking.availability_mask,
                        dst_access_mask,
                        old_layout: resource.tracking.layout,
                        new_layout: access.layout,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: img.handle.get_inner(),
                        subresource_range,
                        ..Default::default()
                    })
                }
                ResourceKind::Buffer(buf) => {
                    pass.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                        src_access_mask: resource.tracking.availability_mask,
                        dst_access_mask,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: buf.handle.get_inner(),
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    })
                }
            }
            resource.tracking.availability_mask = Default::default();
            // update the access types that can now see the resource
            resource.tracking.visibility_mask |= access.access_mask;
            resource.tracking.layout = access.layout;
        }

        // all previous writes are flushed
        if is_write_access(access.access_mask) {
            resource.tracking.availability_mask |= access.access_mask;
        }

        pass.accesses.push(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }

    pub fn finish(mut self) {
        // here we go
        println!("Passes:");
        for p in self.passes.iter() {
            println!("- `{}` ({:?})", p.name, p.snn);
            if p.wait_before {
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
                p.src_stage_mask, p.input_stage_mask
            );
            println!("    input memory barriers:");
            for imb in p.image_memory_barriers.iter() {
                let id = self.context.image_resource_by_handle(imb.image);
                print!("        Image handle={:?} ", imb.image);
                if !id.is_null() {
                    print!(
                        "(id={:?}, name={})",
                        id,
                        self.context.resources.get(id).unwrap().name
                    );
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?} layout:{:?}->{:?}",
                    imb.src_access_mask, imb.dst_access_mask, imb.old_layout, imb.new_layout
                );
            }

            println!("    output stage: {:?}", p.output_stage_mask);
            if p.signal_after {
                println!("    semaphore signal: {:?}", p.snn);
            }
        }

        println!("Final resource states: ");
        for id in self.temporaries {
            let resource = self.context.resources.get(id).unwrap();
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
        }
    }
}

/// Represents a queue submission (a call to vkQueueSubmit or vkQueuePresent)
struct QueueSubmission {
    wait_serials: [u64; MAX_QUEUES],
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    base_snn: SubmissionNumber,
    signal_snn: SubmissionNumber,
    wait_binary_semaphores: Vec<vk::Semaphore>,
    signal_binary_semaphores: Vec<vk::Semaphore>,
    render_finished_semaphore: Option<vk::Semaphore>,
    swapchains: Vec<vk::SwapchainKHR>,
    swapchain_image_indices: Vec<u32>,
    command_buffer: vk::CommandBuffer,
}

impl QueueSubmission {
    fn new() -> QueueSubmission {
        QueueSubmission {
            wait_serials: [0; MAX_QUEUES],
            wait_dst_stages: [Default::default(); MAX_QUEUES],
            base_snn: Default::default(),
            signal_snn: Default::default(),
            wait_binary_semaphores: vec![],
            signal_binary_semaphores: vec![],
            render_finished_semaphore: Default::default(),
            swapchains: vec![],
            swapchain_image_indices: vec![],
            command_buffer: Default::default(),
        }
    }
}

struct SubmissionBuilder {
    open_submissions: [Option<usize>; MAX_QUEUES],
    submissions: Vec<QueueSubmission>,
}

impl SubmissionBuilder {
    fn new() -> SubmissionBuilder {
        SubmissionBuilder {
            open_submissions: [None; MAX_QUEUES],
            submissions: Vec::new(),
        }
    }

    fn get_open_submission(&mut self, queue: u8) -> &mut QueueSubmission {
        if let Some(i) = self.open_submissions[queue as usize] {
            &mut self.submissions[i]
        } else {
            let i = self.submissions.len();
            self.open_submissions[queue as usize] = Some(i);
            self.submissions.push(QueueSubmission::new());
            self.submissions.last_mut().unwrap()
        }
    }
}

pub struct PassBuilder<'ctx, 'batch> {
    batch: &'batch mut Batch<'ctx>,
    pass: Pass,
}

impl<'ctx, 'batch> PassBuilder<'ctx, 'batch> {
    pub fn add_image_usage(
        &mut self,
        image: ResourceId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
        layout: vk::ImageLayout,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            image,
            &ResourceAccessDetails {
                layout,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    pub fn finish(mut self) {
        self.batch.finish_pass(self.pass)
    }
}

struct QueueSubmissionContext<'a> {
    base_serial: u64,
    resources: &'a ResourceMap,
    temporaries: &'a [ResourceId],
    passes: &'a [Pass],
    reachability: Reachability,
}

/*struct DeviceMemoryAllocationInfo {
    allocation: vk_mem::Allocation,
    allocation_info: vk_mem::AllocationInfo,
}*/

struct ResourceAllocationState {
    index: usize,
    dead_and_recycled: bool,
}

struct ResourceAllocationBuilder {
    allocations: Vec<AllocationRequirements>,
    allocation_map: SecondaryMap<ResourceId, ResourceAllocationState>,
}

impl ResourceAllocationBuilder {
    fn new() -> ResourceAllocationBuilder {
        ResourceAllocationBuilder {
            allocations: vec![],
            allocation_map: Default::default(),
        }
    }

    fn assign_memory(
        &mut self,
        ctx: &QueueSubmissionContext,
        pass: &Pass,
        resource_id: ResourceId,
    ) {
        let resource = ctx.resources.get(resource_id).unwrap();

        if !resource.owned
            || resource.allocation_info.is_some()
            || self.allocation_map.get(resource_id).is_some()
        {
            return;
        }

        let mut aliased = false;

        for &alias_candidate_id in ctx.temporaries.iter() {
            let alias_candidate = ctx.resources.get(alias_candidate_id).unwrap();

            if !alias_candidate.owned {
                continue;
            }

            // skip if the resource has user handles pointing to it that may live beyond the current batch
            if alias_candidate.user_ref_count > 0 {
                continue;
            }

            let mut alloc_state =
                if let Some(alloc_state) = self.allocation_map.get_mut(alias_candidate_id) {
                    // skip if the resource is already dead, and its memory was already reused
                    if alloc_state.dead_and_recycled {
                        continue;
                    }
                    alloc_state
                } else {
                    // skip is not allocated yet
                    continue;
                };

            // if we want to use the resource, the resource must be dead (no more uses in subsequent tasks),
            // and there must be an execution dependency chain between the current task and all tasks that last accessed the resource
            let mut live = false;

            // Consider the resource to be live if:
            // 1. the reader is in a previous batch, there's no way to know if the
            // current task has an execution dependency on it, so exclude this resource.
            // 2. the reader is in a future serial
            // 3. there's no execution dependency chain from the reader to the current task.
            live |= alias_candidate.tracking.readers.iter().any(|&read_serial| {
                read_serial != 0
                    && (read_serial <= ctx.base_serial
                        || read_serial >= pass.snn.serial()
                        || ctx.reachability.is_reachable(
                            (read_serial - ctx.base_serial - 1) as usize,
                            pass.batch_index,
                        ))
            });

            let write_serial = alias_candidate.tracking.writer.serial();
            live = live
                || (write_serial != 0
                    && (write_serial <= ctx.base_serial
                        || write_serial >= pass.snn.serial()
                        || ctx.reachability.is_reachable(
                            (write_serial - ctx.base_serial - 1) as usize,
                            pass.batch_index,
                        )));

            if live {
                continue;
            }

            // the resource is dead, try to reuse
            let dead_alloc = &mut self.allocations[alloc_state.index];

            if !dead_alloc.try_adjust(&resource.allocation_requirements) {
                continue;
            }

            // the two resources may alias; the requirements have been adjusted
            // update the allocation map
            let index = alloc_state.index;
            alloc_state.dead_and_recycled = true;

            self.allocation_map.insert(
                resource_id,
                ResourceAllocationState {
                    index,
                    dead_and_recycled: false,
                },
            );

            aliased = true;
            break;
        }

        if !aliased {
            // new allocation
            let index = self.allocations.len();
            self.allocations.push(resource.allocation_requirements);
            self.allocation_map.insert(
                resource_id,
                ResourceAllocationState {
                    index,
                    dead_and_recycled: false,
                },
            );
        }
    }
}

fn build_queue_submissions(
    base_serial: u64,
    resources: &ResourceMap,
    temporaries: &[ResourceId],
    passes: &[Pass],
) -> Vec<QueueSubmission> {
    let ctx = QueueSubmissionContext {
        base_serial,
        resources,
        temporaries,
        passes,
        reachability: compute_reachability(passes),
    };
    let mut alloc_builder = ResourceAllocationBuilder::new();
    let mut submission_builder = SubmissionBuilder::new();

    for p in passes {
        for access in p.accesses.iter() {
            alloc_builder.assign_memory(&ctx, p, access.id);
        }
    }

    unimplemented!()
}

type ResourceMap = SlotMap<ResourceId, Resource>;

struct InFlightBatch {
    resources: TemporarySet,
    signalled_serials: [u64; MAX_QUEUES],
    consumed_semaphores: Vec<vk::Semaphore>,
}

pub struct Context {
    device: Device,
    completed_serials: [u64; MAX_QUEUES],
    next_serial: u64,
    timelines: [UniqueHandle<vk::Semaphore>; MAX_QUEUES],
    resources: ResourceMap,
    in_flight: VecDeque<InFlightBatch>,
    available_semaphores: Vec<vk::Semaphore>,
    vk_khr_swapchain: ash::extensions::khr::Swapchain
}

impl Context {
    pub fn new(device: Device) -> Context {
        let mut timelines: [UniqueHandle<vk::Semaphore>; MAX_QUEUES] = Default::default();

        let mut timeline_create_info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: &mut timeline_create_info as *mut _ as *mut c_void,
            ..Default::default()
        };

        for i in timelines.iter_mut() {
            *i = unsafe {
                UniqueHandle::new(
                    device
                        .device
                        .create_semaphore(&semaphore_create_info, None)
                        .expect("failed to create semaphore"),
                )
            };
        }

        let vk_khr_swapchain = ash::extensions::khr::Swapchain::new(&*VULKAN_INSTANCE, &device.device);

        Context {
            device,
            completed_serials: [0; MAX_QUEUES],
            next_serial: 0,
            timelines,
            resources: SlotMap::with_key(),
            in_flight: VecDeque::new(),
            available_semaphores: vec![],
            vk_khr_swapchain
        }
    }

    fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
        self.resources
            .iter()
            .find_map(|(id, r)| match &r.kind {
                ResourceKind::Image(img) => {
                    if img.handle.get_inner() == handle {
                        Some(id)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or(ResourceId::null())
    }

    fn get_next_serial(&mut self) -> u64 {
        self.next_serial += 1;
        self.next_serial
    }

    // steps:
    // 1. build submissions and allocate memory for resources
    // 2. fill command buffers
    // 3. submit
    // 4. record the batch, prepare for the next one

    fn enqueue_passes(
        &mut self,
        //temporary_set: TemporarySet,
        base_serial: u64,
        temporaries: Vec<ResourceId>,
        passes: Vec<Pass>,
    ) {
        let reachability = compute_reachability(&passes);
        let submissions =
            build_queue_submissions(base_serial, &self.resources, &temporaries, &passes);
    }

    unsafe fn do_submit(&mut self, submissions: &[QueueSubmission]) {
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut used_semaphores = Vec::new();
        let mut last_signalled_timeline_values = [0u64; MAX_QUEUES];

        for s in submissions.iter() {
            signal_semaphores.clear();
            signal_semaphore_values.clear();
            wait_semaphores.clear();
            wait_semaphore_values.clear();
            wait_semaphore_dst_stages.clear();

            let q = s.signal_snn.queue() as usize;
            signal_semaphores.push(self.timelines[q].get_inner());
            signal_semaphore_values.push(s.signal_snn.serial());
            last_signalled_timeline_values[q] = s.signal_snn.serial();

            for (i, &w) in s.wait_serials.iter().enumerate() {
                if w != 0 {
                    wait_semaphores.push(self.timelines[i].get_inner());
                    wait_semaphore_values.push(w);
                    wait_semaphore_dst_stages.push(s.wait_dst_stages[i]);
                }
            }

            for &b in s.wait_binary_semaphores.iter() {
                wait_semaphores.push(b);
                wait_semaphore_values.push(0);
                wait_semaphore_dst_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE); // TODO
                used_semaphores.push(b);
                // TODO reclaim semaphore
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
                command_buffer_count: 1,
                p_command_buffers: &s.command_buffer,
                signal_semaphore_count: signal_semaphores.len() as u32,
                p_signal_semaphores: signal_semaphores.as_ptr(),
                ..Default::default()
            };

            let queue: vk::Queue = self.device.queues_info.queues[q];
            self.device
                .device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
                .expect("queue submission failed");

            if let Some(render_finished_semaphore) = s.render_finished_semaphore {
                let present_info = vk::PresentInfoKHR {
                    wait_semaphore_count: 1,
                    p_wait_semaphores: &render_finished_semaphore,
                    swapchain_count: s.swapchains.len() as u32,
                    p_swapchains: s.swapchains.as_ptr(),
                    p_image_indices: s.swapchain_image_indices.as_ptr(),
                    p_results: ptr::null_mut(),
                    ..Default::default()
                };

                self.vk_khr_swapchain
                    .queue_present(queue, &present_info)
                    .expect("present failed");
                used_semaphores.push(render_finished_semaphore);
            }
        }
    }

    pub fn create_image_resource(
        &mut self,
        name: &str,
        image_type: vk::ImageType,
        usage: vk::ImageUsageFlags,
        format: vk::Format,
        extent: vk::Extent3D,
        mem_required_flags: vk::MemoryPropertyFlags,
        mem_preferred_flags: vk::MemoryPropertyFlags,
        properties: &ImageProperties,
    ) -> ResourceId {
        let create_info = vk::ImageCreateInfo {
            image_type,
            format,
            extent,
            mip_levels: properties.mip_levels,
            array_layers: properties.array_layers,
            samples: get_vk_sample_count(properties.samples),
            tiling: properties.tiling,
            usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };

        let handle = unsafe {
            self.device
                .device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };

        let mem_req = unsafe { self.device.device.get_image_memory_requirements(handle) };

        let id = self.resources.insert(Resource {
            name: name.to_string(),
            //user_ref_count: (),
            user_ref_count: 0,
            owned: false,
            allocation_requirements: AllocationRequirements {
                mem_req,
                required_flags: mem_required_flags,
                preferred_flags: mem_preferred_flags,
            },
            allocation: vk_mem::Allocation::null(),
            allocation_info: None,
            tracking: Default::default(),
            tmp_index: None,
            kind: ResourceKind::Image(ImageResource {
                handle: UniqueHandle::new(handle),
                format,
            }),
        });
        id
    }

    pub fn create_buffer_resource(
        &mut self,
        name: &str,
        usage: vk::BufferUsageFlags,
        byte_size: u64,
        mem_required_flags: vk::MemoryPropertyFlags,
        mem_preferred_flags: vk::MemoryPropertyFlags,
    ) -> ResourceId {
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: byte_size,
            usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };

        let handle = unsafe {
            self.device
                .device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };

        let mem_req = unsafe { self.device.device.get_buffer_memory_requirements(handle) };

        let id = self.resources.insert(Resource {
            name: name.to_string(),
            //user_ref_count: (),
            user_ref_count: 1,
            owned: false,
            allocation_requirements : AllocationRequirements {
                mem_req,
                required_flags: mem_required_flags,
                preferred_flags: mem_preferred_flags
            },
            allocation: vk_mem::Allocation::null(),
            allocation_info: None,
            tracking: Default::default(),
            tmp_index: None,
            kind: ResourceKind::Buffer(BufferResource {
                handle: UniqueHandle::new(handle),
            }),
        });
        id
    }

    pub fn start_batch(&mut self) -> Batch {
        Batch::new(self)
    }
}
