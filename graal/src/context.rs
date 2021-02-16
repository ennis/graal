use crate::device::Device;
use crate::pass::{
    BatchSerialNumber, Pass, PassKind, QueueSerialNumbers, ResourceAccess, SubmissionNumber,
};
use crate::resource::{
    AllocationRequirements, BufferResource, BufferResourceCreateInfo, ImageResource,
    ImageResourceCreateInfo, Resource, ResourceAccessDetails, ResourceKind, ResourceMemory,
    ResourceMemoryInfo, ResourceTrackingInfo,
};
use crate::swapchain::Swapchain;
use crate::vk::Handle;
use crate::MAX_QUEUES;
use ash::version::{DeviceV1_0, DeviceV1_2};
use ash::vk;
use fixedbitset::FixedBitSet;
use slotmap::{SecondaryMap, SlotMap};
use std::alloc::Layout;
use std::collections::VecDeque;
use std::ffi::CString;
use std::mem;
use std::os::raw::c_void;
use std::ptr;

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

pub unsafe fn place_aligned(layout: &Layout, ptr: &mut *mut u8, space: &mut usize) -> *mut u8 {
    let ptr_usize = *ptr as usize;

    let mut off = ptr_usize & (layout.align() - 1);
    if off > 0 {
        off = layout.align() - off;
    }
    if ptr_usize + off + layout.size() > *space {
        ptr::null_mut()
    } else {
        *space -= off + layout.size();
        *ptr = ptr.add(off);
        *ptr
    }
}

struct UploadChunk {
    allocation: vk_mem::Allocation,
    buffer: vk::Buffer,
    base: *mut u8,
    ptr: *mut u8,
    space: usize,
}

impl UploadChunk {
    pub fn new(
        device: &Device,
        memory_usage: vk_mem::MemoryUsage,
        buffer_usage: vk::BufferUsageFlags,
        size: usize,
    ) -> UploadChunk {
        let alloc_info = vk_mem::AllocationCreateInfo {
            flags: vk_mem::AllocationCreateFlags::MAPPED,
            usage: memory_usage,
            ..Default::default()
        };

        let buffer_create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: size as u64,
            usage: buffer_usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: device.queues_info.queue_count as u32,
            p_queue_family_indices: device.queues_info.families.as_ptr(),
            ..Default::default()
        };

        let (buffer, allocation, alloc_info) = device
            .allocator
            .create_buffer(&buffer_create_info, &alloc_info)
            .expect("failed to allocate buffer");
        UploadChunk {
            allocation,
            buffer,
            base: alloc_info.get_mapped_data(),
            ptr: alloc_info.get_mapped_data(),
            space: size,
        }
    }

    pub unsafe fn allocate(
        &mut self,
        layout: &Layout,
    ) -> Option<(*mut u8, vk::Buffer, vk::DeviceSize)> {
        let ptr = place_aligned(layout, &mut self.ptr, &mut self.space);
        if !ptr.is_null() {
            Some((ptr, self.buffer, ptr.offset_from(self.base) as u64))
        } else {
            None
        }
    }
}

/// A pool of CPU-visible memory used for staging resources or uploading dynamic data.
struct UploadPool {
    usage: vk_mem::MemoryUsage,
    chunk_size: usize,
    chunks: Vec<UploadChunk>,
    dedicated: Vec<vk_mem::Allocation>,
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
            let q = src_snn.queue();
            dst.wait_before = true;
            dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
            dst.wait_dst_stages[q as usize] |= dst_stage_mask;
        } else {
            // same-queue dependency, a pipeline barrier is sufficient
            dst.src_stage_mask |= src.output_stage_mask;
        }

        dst.preds.push(src.batch_index);
        src.succs.push(dst.batch_index);
    } else {
        // --- Inter-batch synchronization w/ timeline semaphore
        let q = src_snn.queue();
        dst.wait_before = true;
        dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
        dst.wait_dst_stages[q as usize] |= dst_stage_mask;
    }
}

type TemporarySet = std::collections::BTreeSet<ResourceId>;

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

#[derive(Copy, Clone, Debug)]
pub struct SwapchainImage {
    pub swapchain_id: SwapchainId,
    pub image_id: ResourceId,
    pub image_index: u32,
}

pub struct Batch<'a> {
    base_serial: u64,
    context: &'a mut Context,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the batch
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass<'a>>,
}

impl<'a> Batch<'a> {
    fn new(context: &'a mut Context) -> Batch<'a> {
        Batch {
            base_serial: context.next_serial,
            context,
            temporaries: vec![],
            temporary_set: TemporarySet::new(),
            passes: vec![],
        }
    }

    /// Returns the context from which this batch was started
    pub fn context(&mut self) -> &mut Context {
        self.context
    }

    /// Returns this batch's serial
    pub fn serial(&self) -> BatchSerialNumber {
        BatchSerialNumber(self.context.submitted_batch_count + 1)
    }

    /// Starts building a graphics pass
    pub fn build_graphics_pass<'b>(&'b mut self, name: &str) -> PassBuilder<'a, 'b> {
        let queues_info = self.context.device.queues_info;
        self.build_pass(name, queues_info.indices.graphics, PassKind::Render)
    }

    /// Starts building a compute pass
    pub fn build_compute_pass<'b>(
        &'b mut self,
        name: &str,
        async_compute: bool,
    ) -> PassBuilder<'a, 'b> {
        let queues_info = self.context.device.queues_info;
        let queue_index = if async_compute {
            queues_info.indices.compute
        } else {
            queues_info.indices.graphics
        };
        self.build_pass(name, queue_index, PassKind::Compute)
    }

    /// Starts building a transfer pass
    pub fn build_transfer_pass<'b>(
        &'b mut self,
        name: &str,
        async_transfer: bool,
    ) -> PassBuilder<'a, 'b> {
        let queues_info = self.context.device.queues_info;
        let queue_index = if async_transfer {
            queues_info.indices.transfer
        } else {
            queues_info.indices.graphics
        };
        self.build_pass(name, queue_index, PassKind::Transfer)
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&mut self, name: &str, image: &SwapchainImage) {
        let queue_index = self.context.device.queues_info.indices.present;
        let swapchain = self
            .context
            .swapchains
            .get(image.swapchain_id)
            .unwrap()
            .handle;
        let mut pass_builder = self.build_pass(
            name,
            queue_index,
            PassKind::Present {
                swapchain,
                image_index: image.image_index,
            },
        );
        pass_builder.add_image_usage(
            image.image_id,
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::ALL_COMMANDS,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        pass_builder.finish()
    }

    /// Common code for `build_xxx_pass`
    fn build_pass<'b>(
        &'b mut self,
        name: &str,
        queue_index: u8,
        kind: PassKind,
    ) -> PassBuilder<'a, 'b> {
        let serial = self.context.get_next_serial();
        let batch_index = self.passes.len();
        let snn = SubmissionNumber::new(queue_index, serial);

        PassBuilder {
            batch: self,
            pass: Pass::new(name, batch_index, snn, kind),
        }
    }

    /// Called by `PassBuilder::finish`.
    fn finish_pass(&mut self, pass: Pass<'a>) {
        self.passes.push(pass)
    }

    /// Helper to find the pass given a submission number.
    fn get_pass_mut<'pass, 'b>(
        start_serial: u64,
        passes: &'pass mut [Pass<'b>],
        snn: SubmissionNumber,
    ) -> Option<&'pass mut Pass<'b>> {
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
        //------------------------
        // first, add the resource into the set of temporaries used within this batch
        let resource = self.context.resources.get_mut(id).unwrap();
        if self.temporary_set.insert(id) {
            // this is the first time the resource has been used in the batch
            match resource.memory {
                ResourceMemory::Aliasable(_) => {
                    if resource.tracking.has_writer() || resource.tracking.has_readers() {
                        panic!("transient resource was already used in a previous batch")
                    }
                }
                _ => {}
            }

            self.temporaries.push(id);
        }

        //------------------------
        let is_write = !access.output_stage.is_empty() || resource.tracking.layout != access.layout;

        // update input stage mask
        pass.input_stage_mask |= access.input_stage;

        // handle external semaphore dependency
        let semaphore = mem::take(&mut resource.tracking.wait_binary_semaphore);
        if semaphore != vk::Semaphore::null() {
            pass.wait_binary_semaphores.push(semaphore);
            pass.wait_before = true;
        }

        //------------------------
        // infer execution dependencies
        if is_write {
            if !resource.tracking.has_readers() && resource.tracking.has_writer() {
                // write-after-write
                add_execution_dependency(
                    resource.tracking.writer,
                    Self::get_pass_mut(
                        self.base_serial,
                        &mut self.passes,
                        resource.tracking.writer,
                    ),
                    pass,
                    access.input_stage,
                );
            } else {
                // write-after-read
                for q in 0..MAX_QUEUES as u8 {
                    if resource.tracking.readers[q] != 0 {
                        let src_snn = SubmissionNumber::new(q, resource.tracking.readers[q]);
                        add_execution_dependency(
                            src_snn,
                            Self::get_pass_mut(self.base_serial, &mut self.passes, src_snn),
                            pass,
                            access.input_stage,
                        );
                    }
                }
            }
            // update the resource writer
            pass.output_stage_mask = access.output_stage;
        } else {
            if resource.tracking.has_writer() {
                // read-after-write
                // NOTE a read without a write is probably an uninitialized access
                add_execution_dependency(
                    resource.tracking.writer,
                    Self::get_pass_mut(
                        self.base_serial,
                        &mut self.passes,
                        resource.tracking.writer,
                    ),
                    pass,
                    access.input_stage,
                );
            }
        }

        //------------------------
        // infer memory barriers

        // Q: do we need a memory barrier?
        // A: we need a memory barrier if
        //      - if the operation needs to see all previous writes to the resource:
        //          - if the resource visibility mask doesn't contain the requested access type
        //      - if a layout transition is necessary
        //
        // Note: if the pass overwrites the resource entirely, then the operation technically doesn't need to
        // see the last version of the resource.

        // are all writes to the resource visible to the requested access type?
        let writes_visible =
                // resource was last written in a previous batch, so all writes are made visible
                // by the semaphore wait inserted by the execution dependency
            resource.tracking.writer.serial() < self.base_serial ||
                // resource visible to all MEMORY_READ, or to the requested mask
            resource
                .tracking
                .visibility_mask
                .contains(vk::AccessFlags::MEMORY_READ | access.access_mask);
        // is the layout of the resource different? do we need a transition?
        let layout_transition = resource.tracking.layout != access.layout;
        // is there a possible write-after-write hazard, that requires a memory dependency?
        let write_after_write_hazard =
            is_write && is_write_access(resource.tracking.availability_mask);

        if !writes_visible || layout_transition || write_after_write_hazard {
            // if the last writer of the serial is in another batch, all writes are made available (FIXME and visible?) because of the semaphore
            // wait inserted by the execution dependency. Otherwise, we need to consider the available writes on the resource.
            let src_access_mask = if resource.tracking.writer.serial() < self.base_serial {
                vk::AccessFlags::empty()
            } else {
                resource.tracking.availability_mask
            };
            // no need to make memory visible if we're only writing to the resource
            let dst_access_mask = if !is_read_access(access.access_mask) {
                vk::AccessFlags::empty()
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
                        src_access_mask,
                        dst_access_mask,
                        old_layout: resource.tracking.layout,
                        new_layout: access.layout,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: img.handle,
                        subresource_range,
                        ..Default::default()
                    })
                }
                ResourceKind::Buffer(buf) => {
                    pass.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                        src_access_mask,
                        dst_access_mask,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: buf.handle,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    })
                }
            }
            // all previous writes to the resource have been made available by the barrier ...
            resource.tracking.availability_mask = vk::AccessFlags::empty();
            // ... but not *made visible* to all access types: update the access types that can now see the resource
            resource.tracking.visibility_mask |= access.access_mask;
            resource.tracking.layout = access.layout;
        }

        // all previous writes are flushed
        if is_write_access(access.access_mask) {
            resource.tracking.availability_mask |= access.access_mask;
        }

        // update output stage
        // FIXME doubt
        if is_write {
            resource.tracking.stages = access.output_stage;
            resource.tracking.clear_readers();
            resource.tracking.writer = pass.snn;
        } else {
            // update the resource readers
            resource
                .tracking
                .readers
                .assign_max_serial(pass.snn.queue(), pass.snn.serial());
        }

        pass.accesses.push(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }

    /// Finishes building the batch and submits all the passes to the command queues.
    pub fn finish(self) {
        // here we go
        println!(
            "====== Batch #{} ======",
            self.context.submitted_batch_count + 1
        );
        /*println!("Passes:");
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
                print!("        image handle={:?} ", imb.image);
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
        for &id in self.temporaries.iter() {
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
        }*/

        self.context
            .enqueue_passes(self.base_serial, self.temporaries, self.passes)
    }
}

/// Represents a queue submission (a call to vkQueueSubmit or vkQueuePresent)
struct SubmissionBatch {
    wait_serials: QueueSerialNumbers,
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    signal_snn: SubmissionNumber,
    wait_binary_semaphores: Vec<vk::Semaphore>, // TODO arrayvec
    signal_binary_semaphores: Vec<vk::Semaphore>, // TODO arrayvec
    command_buffers: Vec<vk::CommandBuffer>,
}

impl SubmissionBatch {
    fn new() -> SubmissionBatch {
        SubmissionBatch {
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
        !self.signal_snn.is_valid() && self.command_buffers.is_empty()
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

impl Default for SubmissionBatch {
    fn default() -> Self {
        SubmissionBatch::new()
    }
}

/// Builder object for passes.
pub struct PassBuilder<'a, 'batch> {
    batch: &'batch mut Batch<'a>,
    pass: Pass<'a>,
}

impl<'a, 'batch> PassBuilder<'a, 'batch> {
    /// Registers an image access made by this pass.
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

    pub fn add_buffer_usage(
        &mut self,
        buffer: ResourceId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            buffer,
            &ResourceAccessDetails {
                layout: vk::ImageLayout::UNDEFINED,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    /// Sets the command handler for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_commands(&mut self, commands: impl FnOnce(&Context, vk::CommandBuffer) + 'a) {
        self.pass.commands = Some(Box::new(commands));
    }

    /// Finishes the recording of this pass.
    pub fn finish(self) {
        self.batch.finish_pass(self.pass)
    }
}

struct CommandAllocator {
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

slotmap::new_key_type! {
    pub struct SwapchainId;
    pub struct ResourceId;
}

type ResourceMap = SlotMap<ResourceId, Resource>;
type SwapchainMap = SlotMap<SwapchainId, Swapchain>;

struct InFlightBatch {
    signalled_serials: QueueSerialNumbers,
    transient_allocations: Vec<vk_mem::Allocation>,
    command_pools: Vec<CommandAllocator>,
}

pub struct Context {
    /// Context owns a device
    device: Device,
    /// The serial to be used for the next pass (used by `Batch`)
    next_serial: u64,
    /// Array containing the last completed pass serials for each queue
    completed_serials: QueueSerialNumbers,
    /// Number of submitted batches
    submitted_batch_count: u64,
    /// Number of completed batches
    completed_batch_count: u64,
    /// Array containing the last submitted pass serials for each queue
    last_submitted_serials: QueueSerialNumbers,
    /// Timeline semaphores for each queue, used for cross-queue synchronization
    timelines: [vk::Semaphore; MAX_QUEUES],
    /// Resource map. Cleaned up before every batch submission by deleting unused and unreferenced resources.
    resources: ResourceMap,
    /// Swapchains.
    swapchains: SwapchainMap,
    /// Batches that are currently executing on the GPU.
    in_flight: VecDeque<InFlightBatch>,
    /// Pool of recycled semaphores.
    available_semaphores: Vec<vk::Semaphore>,
    /// Pool of recycled command pools.
    available_command_pools: Vec<CommandAllocator>,
}

impl Context {
    /// Creates a new context.
    pub fn new(device: Device) -> Context {
        let mut timelines: [vk::Semaphore; MAX_QUEUES] = Default::default();

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
                device
                    .device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("failed to create semaphore")
            };
        }

        Context {
            device,
            completed_serials: Default::default(),
            next_serial: 0,
            submitted_batch_count: 0,
            completed_batch_count: 0,
            timelines,
            last_submitted_serials: Default::default(),
            resources: SlotMap::with_key(),
            swapchains: SlotMap::with_key(),
            in_flight: VecDeque::new(),
            available_semaphores: vec![],
            available_command_pools: vec![],
        }
    }

    /// Returns the `ash::Device` associated with this context.
    pub fn device(&self) -> &ash::Device {
        &self.device.device
    }

    /// Returns the handle of the corresponding image resource.
    /// Panics if `id` does not refer to an image resource.
    pub fn image_handle(&self, id: ResourceId) -> vk::Image {
        self.resources.get(id).unwrap().image().handle
    }

    /// Returns the handle of the corresponding buffer resource.
    /// Panics if `id` does not refer to a buffer resource.
    pub fn buffer_handle(&self, id: ResourceId) -> vk::Buffer {
        self.resources.get(id).unwrap().buffer().handle
    }

    /// Returns whether the given batch, identified by its serial, has completed execution.
    pub fn is_batch_completed(&self, serial: BatchSerialNumber) -> bool {
        self.completed_batch_count >= serial.0
    }

    fn set_debug_object_name(
        &self,
        object_type: vk::ObjectType,
        object_handle: u64,
        name: &str,
        transient: bool,
    ) {
        unsafe {
            let name = if transient {
                format!("{}.{}", name, self.next_serial)
            } else {
                name.to_string()
            };
            let object_name = CString::new(name.as_str()).unwrap();
            self.device
                .vk_ext_debug_utils
                .debug_utils_set_object_name(
                    self.device.device.handle(),
                    &vk::DebugUtilsObjectNameInfoEXT {
                        object_type,
                        object_handle,
                        p_object_name: object_name.as_ptr(),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }

    fn create_command_pool(&mut self, queue_index: u8) -> CommandAllocator {
        let queue_family = self.device.queues_info.families[queue_index as usize];
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

    /// Creates a binary semaphore (or return a previously used semaphore that is unsignalled).
    fn create_semaphore(&mut self) -> vk::Semaphore {
        if let Some(semaphore) = self.available_semaphores.pop() {
            return semaphore;
        }

        unsafe {
            let create_info = vk::SemaphoreCreateInfo {
                ..Default::default()
            };
            self.device
                .device
                .create_semaphore(&create_info, None)
                .expect("failed to create semaphore")
        }
    }

    /*fn image_resource_by_handle(&self, handle: vk::Image) -> ResourceId {
        self.resources
            .iter()
            .find_map(|(id, r)| match &r.kind {
                ResourceKind::Image(img) => {
                    if img.handle == handle {
                        Some(id)
                    } else {
                        None
                    }
                }
                _ => None,
            })
            .unwrap_or(ResourceId::null())
    }*/

    fn get_next_serial(&mut self) -> u64 {
        self.next_serial += 1;
        self.next_serial
    }

    fn destroy_resource(device: &Device, resource: &mut Resource) {
        // destroy the object, if we're responsible for it
        if resource.should_delete {
            match &mut resource.kind {
                ResourceKind::Buffer(buf) => {
                    unsafe {
                        // TODO safety
                        device
                            .device
                            .destroy_buffer(mem::take(&mut buf.handle), None);
                    }
                }
                ResourceKind::Image(img) => {
                    unsafe {
                        // TODO safety
                        device
                            .device
                            .destroy_image(mem::take(&mut img.handle), None)
                    }
                }
            }
        }

        // deallocate its memory, if it was allocated for this object exclusively
        match resource.memory {
            ResourceMemory::Exclusive(allocation) => {
                device.allocator.free_memory(&allocation).unwrap()
            }
            _ => {}
        }
    }

    fn cleanup_resources(&mut self) {
        // time to cleanup resources
        // we retain only resources that have a non-zero user refcount (the user is still holding a reference to the resource),
        // and resources that have reader or writer passes that have not yet completed
        let completed_serials = self.completed_serials;
        let device = &self.device;

        self.resources.retain(|_id, r| {
            // refcount != 0 OR any reader not completed OR writer not completed
            let keep = r.user_ref_count != 0
                || r.tracking.readers > completed_serials
                || r.tracking.writer.serial() > completed_serials.serial(r.tracking.writer.queue());
            if !keep {
                Self::destroy_resource(device, r);
            }
            keep
        })
    }

    fn wait_for_batches_in_flight(&mut self) {
        // pacing
        while self.in_flight.len() >= 2 {
            // two batches in flight already, must wait for the oldest one
            let mut b = self.in_flight.pop_front().unwrap();
            let wait_info = vk::SemaphoreWaitInfo {
                semaphore_count: self.timelines.len() as u32,
                p_semaphores: self.timelines.as_ptr(),
                p_values: b.signalled_serials.0.as_ptr(),
                ..Default::default()
            };
            unsafe {
                self.device
                    .device
                    .wait_semaphores(&wait_info, 10_000_000_000)
                    .expect("error waiting for batch");
            }

            // update completed serials
            // we just waited on those serials, so we know they are completed
            self.completed_serials = b.signalled_serials;

            // recycle command pools
            for cb_pool in b.command_pools.iter_mut() {
                cb_pool.reset(&self.device.device)
            }
            self.available_command_pools.append(&mut b.command_pools);

            // given the new completed serials, free resources that have expired
            self.cleanup_resources();

            // free transient allocations
            for alloc in b.transient_allocations.iter() {
                self.device.allocator.free_memory(alloc).unwrap();
            }

            // bump completed batch count
            self.completed_batch_count += 1;
        }
    }

    fn dump_state(&self) {
        /*println!("Number of batches in flight: {}", self.in_flight.len());
        println!("Resources:");
        for (id, r) in self.resources.iter() {
            println!("- {:?}: {:?}", id, r);
        }
        println!("Available semaphores:");
        for s in self.available_semaphores.iter() {
            println!("- {:?}", s);
        }
        println!("Available command pools:");
        for cmd_pool in self.available_command_pools.iter() {
            println!(
                "- VkCommandPool {:?}: queue family #{}, {} used, {} free",
                cmd_pool.command_pool,
                cmd_pool.queue_family,
                cmd_pool.used.len(),
                cmd_pool.free.len()
            );
        }*/
        println!("VMA stats:");
        if let Ok(stats) = self.device.allocator.calculate_stats() {
            println!(
                "- number of allocations: {} in {} device memory blocks",
                stats.total.allocationCount, stats.total.blockCount
            );
            println!(
                "- memory usage: {} kB ({} kB used + {} kB unused)",
                (stats.total.usedBytes + stats.total.unusedBytes) / 1000,
                stats.total.usedBytes / 1000,
                stats.total.unusedBytes / 1000
            );
        }
    }

    pub fn current_batch_index(&self) -> u64 {
        self.submitted_batch_count
    }

    fn enqueue_passes(
        &mut self,
        base_serial: u64,
        temporaries: Vec<ResourceId>,
        mut passes: Vec<Pass>,
    ) {
        self.wait_for_batches_in_flight();
        let transient_allocations =
            self.allocate_transient_memory(base_serial, &temporaries, &passes);
        let command_pools = self.submit_passes(&mut passes);
        self.in_flight.push_back(InFlightBatch {
            signalled_serials: self.last_submitted_serials,
            transient_allocations,
            command_pools,
        });
        self.submitted_batch_count += 1;
        self.dump_state();
    }

    fn allocate_transient_memory(
        &mut self,
        base_serial: u64,
        temporaries: &[ResourceId],
        passes: &[Pass],
    ) -> Vec<vk_mem::Allocation> {
        #[derive(Copy, Clone, Debug)]
        struct AllocIndex {
            index: usize,
            dead_and_recycled: bool,
        }

        let reachability = compute_reachability(passes);
        // alloc index -> alloc requirements
        let mut requirements: Vec<AllocationRequirements> = Vec::new();
        // resource id -> allocation mapping (index+state)
        let mut alloc_map: SecondaryMap<ResourceId, AllocIndex> = SecondaryMap::new();

        for pass in passes {
            // --- assign memory for all resources accessed in this task
            for access in pass.accesses.iter() {
                let resource_id = access.id;
                let resource = self.resources.get(resource_id).unwrap();

                // already allocated
                if alloc_map.get(resource_id).is_some() {
                    continue;
                }

                let alloc_req = match &resource.memory {
                    ResourceMemory::Aliasable(req) => *req,
                    // not a transient, nothing to allocate
                    _ => continue,
                };

                let mut aliased = false;

                'alias: for &alias_candidate_id in temporaries.iter() {
                    if alias_candidate_id == resource_id {
                        continue;
                    }

                    let alias_candidate = self.resources.get(alias_candidate_id).unwrap();

                    // skip if not aliasable
                    let _alias_alloc_req = match &alias_candidate.memory {
                        ResourceMemory::Aliasable(req) => req,
                        _ => continue,
                    };

                    // skip if the resource has user handles pointing to it that may live beyond the current batch
                    if alias_candidate.user_ref_count > 0 {
                        continue;
                    }

                    let mut alloc_state =
                        if let Some(alloc_state) = alloc_map.get_mut(alias_candidate_id) {
                            // skip if the resource is already dead, and its memory was already reused
                            if alloc_state.dead_and_recycled {
                                continue;
                            }
                            alloc_state
                        } else {
                            // skip if not allocated yet
                            continue;
                        };

                    // if we want to use the resource, the resource must be dead (no more uses in subsequent tasks),
                    // and there must be an execution dependency chain between the current task and all tasks that last accessed the resource

                    for &read_serial in alias_candidate.tracking.readers.iter() {
                        // Consider the resource to be live if:
                        // 1. the reader is in a previous batch (we don't have info about execution dependencies between passes in different batches)
                        // 2. the reader comes after this pass
                        // 3. there's no execution dependency chain from the reader to the current task.

                        if read_serial != 0
                            && (read_serial <= base_serial
                                || read_serial >= pass.snn.serial()
                                || !reachability.is_reachable(
                                    (read_serial - base_serial - 1) as usize,
                                    pass.batch_index,
                                ))
                        {
                            continue 'alias;
                        }
                    }

                    let write_serial = alias_candidate.tracking.writer.serial();
                    if write_serial != 0
                        && (write_serial <= base_serial
                            || write_serial >= pass.snn.serial()
                            || !reachability.is_reachable(
                                (write_serial - base_serial - 1) as usize,
                                pass.batch_index,
                            ))
                    {
                        continue;
                    }

                    // the resource is dead, try to reuse
                    let dead_alloc = &mut requirements[alloc_state.index];

                    if !dead_alloc.try_adjust(&alloc_req) {
                        continue;
                    }

                    // the two resources may alias; the requirements have been adjusted
                    // update the allocation map
                    let index = alloc_state.index;
                    alloc_state.dead_and_recycled = true;

                    alloc_map.insert(
                        resource_id,
                        AllocIndex {
                            index,
                            dead_and_recycled: false,
                        },
                    );

                    aliased = true;
                    break;
                }

                if !aliased {
                    // new allocation
                    let index = requirements.len();
                    requirements.push(alloc_req);
                    alloc_map.insert(
                        resource_id,
                        AllocIndex {
                            index,
                            dead_and_recycled: false,
                        },
                    );
                }
            }
        }

        // --- print some debug info
        println!("Memory blocks:");
        for (i, req) in requirements.iter().enumerate() {
            println!(" block #{}: {:?}", i, req);
        }
        println!("Memory block assignments:");
        for &tmp in temporaries {
            if let Some(alloc_state) = alloc_map.get(tmp) {
                println!(
                    "{} => {:?}",
                    self.resources.get(tmp).unwrap().name,
                    alloc_state
                );
            } else {
                println!("{} => N/A", self.resources.get(tmp).unwrap().name);
            }
        }

        // now allocate device memory
        let mut allocations = Vec::with_capacity(requirements.len());
        let mut allocation_infos = Vec::with_capacity(requirements.len());

        for alloc_req in requirements.iter() {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                ..Default::default()
            };
            let (allocation, allocation_info) = self
                .device
                .allocator
                .allocate_memory(&alloc_req.mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            allocations.push(allocation);
            allocation_infos.push(allocation_info);
        }

        // and assign them to the resources
        for &tmp in temporaries {
            if let Some(alloc_index) = alloc_map.get(tmp) {
                let resource = self.resources.get_mut(tmp).unwrap();
                let alloc_info = &allocation_infos[alloc_index.index];
                match &resource.kind {
                    ResourceKind::Image(img) => unsafe {
                        self.device
                            .device
                            .bind_image_memory(
                                img.handle,
                                alloc_info.get_device_memory(),
                                alloc_info.get_offset() as u64,
                            )
                            .unwrap();
                    },
                    ResourceKind::Buffer(buf) => unsafe {
                        self.device
                            .device
                            .bind_buffer_memory(
                                buf.handle,
                                alloc_info.get_device_memory(),
                                alloc_info.get_offset() as u64,
                            )
                            .unwrap();
                    },
                }
            }
        }

        allocations
    }

    fn submit_batch(&mut self, q: u8, sb: &SubmissionBatch) {
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();

        // end command buffers
        for &cb in sb.command_buffers.iter() {
            unsafe { self.device.device.end_command_buffer(cb).unwrap() }
        }

        // setup timeline signal
        signal_semaphores.push(self.timelines[q as usize]);
        signal_semaphore_values.push(sb.signal_snn.serial());
        self.last_submitted_serials[q] = sb.signal_snn.serial();

        // binary semaphore signals
        for &s in sb.signal_binary_semaphores.iter() {
            signal_semaphores.push(s);
            signal_semaphore_values.push(0);
        }

        // setup timeline waits
        for (i, &w) in sb.wait_serials.iter().enumerate() {
            if w != 0 {
                wait_semaphores.push(self.timelines[i]);
                wait_semaphore_values.push(w);
                wait_semaphore_dst_stages.push(sb.wait_dst_stages[i]);
            }
        }

        // setup binary semaphore waits
        for &s in sb.wait_binary_semaphores.iter() {
            wait_semaphores.push(s);
            wait_semaphore_values.push(0);
            // TODO
            wait_semaphore_dst_stages.push(vk::PipelineStageFlags::TOP_OF_PIPE);
            // after the submission, the semaphore will be in an unsignalled state,
            // ready to be reused
            self.available_semaphores.push(s);
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
            command_buffer_count: sb.command_buffers.len() as u32,
            p_command_buffers: sb.command_buffers.as_ptr(),
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

    fn submit_passes(&mut self, passes: &mut [Pass]) -> Vec<CommandAllocator> {
        // current submission batches per queue
        let mut submission_batches: [SubmissionBatch; MAX_QUEUES] = Default::default();
        // one command pool per queue (might not be necessary if the queues belong to the same family,
        // but they usually don't)
        let mut command_pools: [Option<CommandAllocator>; MAX_QUEUES] = Default::default();

        for p in passes.iter_mut() {
            let q = p.snn.queue();
            if p.wait_before {
                // the pass needs a semaphore wait, so it needs a separate batch
                // close the batches on all queues that the pass waits on
                for i in 0..MAX_QUEUES as u8 {
                    if !submission_batches[i as usize].is_empty()
                        && (i == q || p.wait_serials[i] != 0)
                    {
                        self.submit_batch(i as u8, &submission_batches[i as usize]);
                        submission_batches[i as usize].reset();
                    }
                }
            }

            let sb: &mut SubmissionBatch = &mut submission_batches[q as usize];
            if p.wait_before {
                sb.wait_serials = p.wait_serials;
                sb.wait_dst_stages = p.wait_dst_stages;
                sb.wait_binary_semaphores = p.wait_binary_semaphores.clone();
            }

            // ensure that a command pool has been allocated for the queue
            let command_pool: &mut CommandAllocator = command_pools[q as usize]
                .get_or_insert_with(|| self.create_command_pool(p.snn.queue()));
            // append to the last command buffer of the batch, otherwise create another one

            if sb.command_buffers.is_empty() {
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
                sb.command_buffers.push(cb);
            };

            let cb = sb.command_buffers.last().unwrap().clone();

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
                || p.input_stage_mask != vk::PipelineStageFlags::BOTTOM_OF_PIPE
                || !p.buffer_memory_barriers.is_empty()
                || !p.image_memory_barriers.is_empty()
            {
                let src_stage_mask = if p.src_stage_mask.is_empty() {
                    vk::PipelineStageFlags::TOP_OF_PIPE
                } else {
                    p.src_stage_mask
                };
                let dst_stage_mask = if p.input_stage_mask.is_empty() {
                    vk::PipelineStageFlags::BOTTOM_OF_PIPE
                } else {
                    p.input_stage_mask
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

            match p.kind {
                PassKind::Present {
                    swapchain,
                    image_index,
                } => {
                    // present operation:
                    // modify the current batch to signal a semaphore and close it
                    let render_finished_semaphore = self.create_semaphore();
                    // FIXME if the swapchain image is last modified by another queue,
                    // then this batch contains no commands, only one timeline wait
                    // and one binary semaphore signal.
                    // This could be optimized by signalling a binary semaphore on the pass
                    // that modifies the swapchain image, but at the cost of code complexity
                    // and maintainability.
                    // Eventually, the presentation engine might support timeline semaphores
                    // directly, which will make this entire problem vanish.
                    sb.signal_binary_semaphores.push(render_finished_semaphore);
                    self.submit_batch(q, sb);
                    sb.reset();
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
                    self.available_semaphores.push(render_finished_semaphore);
                }
                _ => {
                    if let Some(handler) = p.commands.take() {
                        handler(self, cb);
                    }

                    // update signalled serial for the batch (pass serials are guaranteed to be increasing)
                    sb.signal_snn = p.snn;
                }
            }

            unsafe {
                self.device.vk_ext_debug_utils.cmd_end_debug_utils_label(cb);
            }

            if p.signal_after {
                // the pass needs a semaphore signal: this terminates the batch on the queue
                self.submit_batch(q, sb);
                sb.reset();
            }
        }

        // close unfinished batches
        for sb in submission_batches.iter() {
            if !sb.is_empty() {
                self.submit_batch(sb.signal_snn.queue(), sb)
            }
        }

        //
        command_pools
            .iter_mut()
            .filter_map(|cmd_pool| cmd_pool.take())
            .collect()
    }

    /// Creates a new image resource.
    ///
    pub fn create_image_resource(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
    ) -> ResourceId {
        // for now all resources are CONCURRENT, because that's the only way they can
        // be read across multiple queues.
        // Maybe exclusive ownership will be needed at some point, but then we should prevent
        // them from being used across multiple queues. I know that there's the possibility of doing
        // a "queue ownership transfer", but that shit is incomprehensible.
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.image_type,
            format: image_info.format,
            extent: image_info.extent,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: image_info.tiling,
            usage: image_info.usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.device.device.get_image_memory_requirements(handle) };
        let memory = if image_info.transient {
            ResourceMemory::Aliasable(AllocationRequirements {
                mem_req,
                required_flags: memory_info.required_flags,
                preferred_flags: memory_info.preferred_flags,
            })
        } else {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                preferred_flags: memory_info.preferred_flags,
                required_flags: memory_info.required_flags,
                ..Default::default()
            };
            let (alloc, alloc_info) = self
                .device
                .allocator
                .allocate_memory(&mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            unsafe {
                self.device
                    .device
                    .bind_image_memory(
                        handle,
                        alloc_info.get_device_memory(),
                        alloc_info.get_offset() as u64,
                    )
                    .unwrap();
            }
            ResourceMemory::Exclusive(alloc)
        };
        let id = self.resources.insert(Resource {
            name: name.to_string(),
            user_ref_count: 0, // start at 1 if not transient?
            memory,
            tracking: Default::default(),
            should_delete: true,
            kind: ResourceKind::Image(ImageResource {
                handle,
                format: image_info.format,
            }),
        });
        self.set_debug_object_name(
            vk::ObjectType::IMAGE,
            handle.as_raw(),
            name,
            image_info.transient,
        );
        id
    }

    pub fn create_buffer_resource(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_info: &BufferResourceCreateInfo,
    ) -> (ResourceId, *mut u8) {
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: buffer_info.byte_size,
            usage: buffer_info.usage,
            sharing_mode: if self.device.queues_info.queue_count == 1 {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };
        let mem_req = unsafe { self.device.device.get_buffer_memory_requirements(handle) };
        let (memory, mapped_ptr) = if buffer_info.transient && !buffer_info.map_on_create {
            // We can delay allocation only if the user requests a transient resource and
            // if the resource does not need to be mapped immediately.
            let memory = ResourceMemory::Aliasable(AllocationRequirements {
                mem_req,
                required_flags: memory_info.required_flags,
                preferred_flags: memory_info.preferred_flags,
            });
            (memory, ptr::null_mut())
        } else {
            let allocation_create_info = vk_mem::AllocationCreateInfo {
                flags: if buffer_info.map_on_create {
                    vk_mem::AllocationCreateFlags::MAPPED
                } else {
                    vk_mem::AllocationCreateFlags::NONE
                },
                preferred_flags: memory_info.preferred_flags,
                required_flags: memory_info.required_flags,
                ..Default::default()
            };
            let (alloc, alloc_info) = self
                .device
                .allocator
                .allocate_memory(&mem_req, &allocation_create_info)
                .expect("failed to allocate device memory");
            unsafe {
                self.device
                    .device
                    .bind_buffer_memory(
                        handle,
                        alloc_info.get_device_memory(),
                        alloc_info.get_offset() as u64,
                    )
                    .unwrap();
            }
            let memory = ResourceMemory::Exclusive(alloc);
            let mapped_ptr = if buffer_info.map_on_create {
                let ptr = alloc_info.get_mapped_data();
                assert!(!ptr.is_null(), "failed to map buffer");
                ptr
            } else {
                ptr::null_mut()
            };
            (memory, mapped_ptr)
        };
        let id = self.resources.insert(Resource {
            name: name.to_string(),
            user_ref_count: 0,
            memory,
            tracking: Default::default(),
            should_delete: true,
            kind: ResourceKind::Buffer(BufferResource { handle }),
        });
        self.set_debug_object_name(
            vk::ObjectType::BUFFER,
            handle.as_raw(),
            name,
            buffer_info.transient,
        );
        (id, mapped_ptr)
    }

    pub unsafe fn create_swapchain(
        &mut self,
        surface: vk::SurfaceKHR,
        initial_size: (u32, u32),
    ) -> SwapchainId {
        self.swapchains
            .insert(Swapchain::new(&self.device, surface, initial_size))
    }

    pub unsafe fn acquire_next_image(&mut self, swapchain_id: SwapchainId) -> SwapchainImage {
        let image_available = self.create_semaphore();
        let swapchain = self.swapchains.get_mut(swapchain_id).unwrap();
        let (image_index, _suboptimal) =
            swapchain.acquire_next_image(&self.device, image_available);

        // create a resource entry so that we can track usages of the swapchain image
        let image_id = self.resources.insert(Resource {
            name: format!("swapchain {:?} image #{}", swapchain.handle, image_index),
            user_ref_count: 0,
            tracking: ResourceTrackingInfo {
                wait_binary_semaphore: image_available,
                ..Default::default()
            },
            memory: ResourceMemory::External,
            // swapchain images are owned by the swapchain so we shouldn't delete them
            should_delete: false,
            kind: ResourceKind::Image(ImageResource {
                handle: swapchain.images[image_index as usize],
                format: swapchain.format,
            }),
        });

        SwapchainImage {
            swapchain_id,
            image_id,
            image_index,
        }
    }

    ///
    pub unsafe fn resize_swapchain(&mut self, swapchain: SwapchainId, size: (u32, u32)) {
        let swapchain = self.swapchains.get_mut(swapchain).unwrap();
        swapchain.resize(&self.device, size);
    }

    //
    pub fn destroy_swapchain(&mut self, swapchain: SwapchainId) {
        unimplemented!()
    }

    pub fn start_batch(&mut self) -> Batch {
        Batch::new(self)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // TODO
    }
}
