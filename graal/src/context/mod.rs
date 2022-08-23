mod sync;

use crate::{
    device::{AccessTracker, BufferResource, Device, ResourceAllocation, ResourceKind, ResourceTrackingInfo},
    serial::{FrameNumber, QueueSerialNumbers, SubmissionNumber},
    BufferId, ImageId, ImageInfo, ImageRegistrationInfo, ResourceGroupId, ResourceId, ResourceOwnership,
    ResourceRegistrationInfo, Swapchain, SwapchainImage, MAX_QUEUES,
};

use crate::{
    context::sync::synchronize_and_track_resources,
    device::{ImageResource, Resource, ResourceGroupMap, ResourceMap},
};
use ash::vk;
use std::{
    collections::{HashSet, VecDeque},
    ffi::CString,
    fmt, mem,
    mem::ManuallyDrop,
    ops::Deref,
    os::raw::c_void,
    ptr,
    sync::Arc,
};
use tracing::trace_span;

/// Maximum time to wait for batches to finish in `SubmissionState::wait`.
pub(crate) const SEMAPHORE_WAIT_TIMEOUT_NS: u64 = 5_000_000_000;

pub(crate) fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        0 => vk::SampleCountFlags::TYPE_1,
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

/*
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
}*/

pub fn is_write_access(mask: vk::AccessFlags) -> bool {
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

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT
    )
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, vk::Format::S8_UINT)
}

pub fn format_aspect_mask(fmt: vk::Format) -> vk::ImageAspectFlags {
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

/// Represents a resource access in a pass.
#[derive(Debug)]
pub(crate) struct ResourceAccessDetails {
    pub(crate) initial_layout: vk::ImageLayout,
    pub(crate) final_layout: vk::ImageLayout,
    pub(crate) access_mask: vk::AccessFlags,
    pub(crate) stage_mask: vk::PipelineStageFlags,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AccessTypeInfo {
    pub access_mask: vk::AccessFlags,
    pub stage_mask: vk::PipelineStageFlags,
    pub layout: vk::ImageLayout,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PASSES
////////////////////////////////////////////////////////////////////////////////////////////////////

/// The type of callback functions invoked when recording command buffers.
pub type CommandCallback<'a, UserContext> =
    Box<dyn FnOnce(&mut RecordingContext, &mut UserContext, vk::CommandBuffer) + 'a>;

/// The type of callback functions invoked when submitting work to a queue.
pub type QueueCallback<'a, UserContext> = Box<dyn FnOnce(&mut RecordingContext, &mut UserContext, vk::Queue) + 'a>;

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
    pub(crate) fn queue_index(&self, device: &Device, async_pass: bool) -> usize {
        (match self {
            PassType::Compute if async_pass => device.queues_info.indices.compute,
            PassType::Transfer if async_pass => device.queues_info.indices.transfer,
            PassType::Present { .. } => device.queues_info.indices.present,
            _ => device.queues_info.indices.graphics,
        }) as usize
    }
}

enum PassEvaluationCallback<'a, UserContext> {
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
    Queue(QueueCallback<'a, UserContext>),
    CommandBuffer(CommandCallback<'a, UserContext>),
}

struct ResourceAccess {
    id: ResourceId,
    access_mask: vk::AccessFlags,
    stage_mask: vk::PipelineStageFlags,
    initial_layout: vk::ImageLayout,
    final_layout: vk::ImageLayout,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreWaitKind {
    Binary,
    Timeline(u64),
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreSignalKind {
    Binary,
    Timeline(u64),
}

/// Represents a semaphore wait operation outside of the queue timelines.
#[derive(Clone, Debug)]
pub struct SemaphoreWait {
    /// The semaphore in question
    semaphore: vk::Semaphore,
    /// Whether the semaphore is internally managed (owned by the context).
    /// If true, the semaphore will be reclaimed by the context after it is consumed (waited on).
    /// FIXME this is never read?
    owned: bool,
    /// Destination stage
    dst_stage: vk::PipelineStageFlags,
    /// The kind of wait operation.
    wait_kind: SemaphoreWaitKind,
}

#[derive(Clone, Debug)]
pub(crate) struct SemaphoreSignal {
    semaphore: vk::Semaphore,
    signal_kind: SemaphoreSignalKind,
}

/// A pass within a frame.
pub(crate) struct Pass<'a, UserContext> {
    /// Name of this pass, for debugging purposes.
    name: String,

    /// Index of the pass in the frame.
    frame_index: usize,

    ty: PassType,
    async_queue: bool,

    /// Pass callback.
    eval: Option<PassEvaluationCallback<'a, UserContext>>,

    /// Resource dependencies.
    accesses: Vec<ResourceAccess>,

    /// Semaphores to wait on before starting the pass.
    external_semaphore_waits: Vec<SemaphoreWait>,

    /// Semaphores to signal after finishing the pass.
    external_semaphore_signals: Vec<SemaphoreSignal>,

    // --- All fields below this line are computed by autosync--------------------------------------
    /// Submission number of the pass. Set during frame submission.
    snn: SubmissionNumber,

    /// Predecessors of the pass (all passes that must happen before this one).
    preds: Vec<usize>,

    /// Successors of the pass (all passes for which this task is a predecessor).
    //pub(crate) succs: Vec<usize>,

    /// Whether the queue timeline semaphores must be signalled after the pass.
    signal_queue_timelines: bool,

    src_stage_mask: vk::PipelineStageFlags,
    dst_stage_mask: vk::PipelineStageFlags,
    image_memory_barriers: Vec<vk::ImageMemoryBarrier>,
    buffer_memory_barriers: Vec<vk::BufferMemoryBarrier>,
    global_memory_barrier: Option<vk::MemoryBarrier>,

    wait_serials: QueueSerialNumbers,
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
}

impl<'a, UserContext> Pass<'a, UserContext> {
    /// Helper function to setup an image memory barrier entry.
    fn get_or_create_image_memory_barrier(
        &mut self,
        handle: vk::Image,
        format: vk::Format,
    ) -> &mut vk::ImageMemoryBarrier {
        if let Some(b) = self.image_memory_barriers.iter_mut().position(|b| b.image == handle) {
            &mut self.image_memory_barriers[b]
        } else {
            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: format_aspect_mask(format),
                base_mip_level: 0,
                level_count: vk::REMAINING_MIP_LEVELS,
                base_array_layer: 0,
                layer_count: vk::REMAINING_ARRAY_LAYERS,
            };
            self.image_memory_barriers.push(vk::ImageMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::empty(),
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::UNDEFINED,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image: handle,
                subresource_range,
                ..Default::default()
            });
            self.image_memory_barriers.last_mut().unwrap()
        }
    }

    /// Helper function to setup a buffer memory barrier entry.
    fn get_or_create_buffer_memory_barrier(&mut self, handle: vk::Buffer) -> &mut vk::BufferMemoryBarrier {
        if let Some(b) = self.buffer_memory_barriers.iter_mut().position(|b| b.buffer == handle) {
            &mut self.buffer_memory_barriers[b]
        } else {
            self.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::empty(),
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                buffer: handle,
                offset: 0,
                size: vk::WHOLE_SIZE,
                ..Default::default()
            });
            self.buffer_memory_barriers.last_mut().unwrap()
        }
    }

    fn get_or_create_global_memory_barrier(&mut self) -> &mut vk::MemoryBarrier {
        self.global_memory_barrier.get_or_insert_with(Default::default)
    }

    fn new(name: &str, index: usize, ty: PassType, async_queue: bool) -> Pass<'a, UserContext> {
        Pass {
            name: name.to_string(),
            eval: None,
            accesses: vec![],
            ty,
            async_queue,
            snn: SubmissionNumber::default(),
            preds: vec![],
            signal_queue_timelines: false,
            src_stage_mask: Default::default(),
            dst_stage_mask: Default::default(),
            image_memory_barriers: vec![],
            buffer_memory_barriers: vec![],
            global_memory_barrier: None,
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            external_semaphore_waits: vec![],
            external_semaphore_signals: vec![],
            frame_index: index,
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// PassBuilder
////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct PassBuilder<'a, 'b, UserContext> {
    frame: &'a mut Frame<'b, UserContext>,
    pass: ManuallyDrop<Pass<'b, UserContext>>,
}

impl<'a, 'b, UserContext> Drop for PassBuilder<'a, 'b, UserContext> {
    fn drop(&mut self) {
        panic!("PassBuilder object dropped. Use `PassBuilder::finished` instead")
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
    pub fn add_external_semaphore_signal(&mut self, semaphore: vk::Semaphore, signal_kind: SemaphoreSignalKind) {
        self.pass
            .external_semaphore_signals
            .push(SemaphoreSignal { semaphore, signal_kind })
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
        self.pass.accesses.push(ResourceAccess {
            id: id.resource_id(),
            access_mask,
            stage_mask,
            initial_layout,
            final_layout,
        })
    }

    pub fn add_buffer_dependency(
        &mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    ) {
        self.pass.accesses.push(ResourceAccess {
            id: id.resource_id(),
            access_mask,
            stage_mask,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::UNDEFINED,
        })
    }

    /// Sets the command buffer recording function for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_record_callback(&mut self, record_cb: CommandCallback<'b, UserContext>) {
        self.pass.eval = Some(PassEvaluationCallback::CommandBuffer(record_cb));
    }

    pub fn set_submit_callback(&mut self, submit_cb: QueueCallback<'b, UserContext>) {
        self.pass.eval = Some(PassEvaluationCallback::Queue(submit_cb));
    }

    /// Ends the current pass.
    pub fn finish(mut self) {
        let pass = unsafe {
            // SAFETY: self.pass not used afterwards, including Drop
            ManuallyDrop::take(&mut self.pass)
        };
        self.frame.passes.push(pass);

        /*if self.frame.inner.collect_sync_debug_info {
            let mut info = SyncDebugInfo::new();

            // current resource tracking info
            let objects = self.frame.context.device.objects.lock().unwrap();
            for (id, r) in objects.resources.iter() {
                info.tracking.insert(id, r.tracking);
            }
            // current sync table
            info.xq_sync_table = self.frame.inner.xq_sync_table;
            self.frame.inner.sync_debug_info.push(info);
        }*/

        // skip running the panicking destructor
        mem::forget(self)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// FRAMES
////////////////////////////////////////////////////////////////////////////////////////////////////

/*#[derive(Copy, Clone, Debug)]
enum FreezeResource {
    FreezeImage { id: ImageId, group_id: ResourceGroupId },
    FreezeBuffer { id: BufferId, group_id: ResourceGroupId },
}*/

#[derive(Copy, Clone, Debug)]
enum FrameCommand {
    FreezeResource {
        resource: ResourceId,
        group: ResourceGroupId,
    },
    DestroyImage {
        image: ImageId,
    },
    DestroyBuffer {
        buffer: BufferId,
    },
}

pub struct Frame<'a, UserContext> {
    passes: Vec<Pass<'a, UserContext>>,
    initial_wait: QueueSerialNumbers,
    commands: Vec<(usize, FrameCommand)>,
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
            //.field("base_serial", &self.inner.base_sn)
            //.field("frame_serial", &self.inner.frame_number)
            .finish()
    }
}

impl<'a, UserContext> Frame<'a, UserContext> {
    pub fn new() -> Frame<'a, UserContext> {
        Frame {
            passes: vec![],
            initial_wait: Default::default(),
            commands: vec![],
        }
    }

    /// Adds a dependency on a GPU future object.
    ///
    /// Execution of the *whole* frame will wait for the operation represented by the future to complete.
    pub fn add_dependency(&mut self, future: GpuFuture) {
        self.initial_wait = self.initial_wait.join(future.serials);
    }

    /// Adds an image to a resource group.
    pub fn freeze_image(&mut self, image_id: ImageId, group_id: ResourceGroupId) {
        self.commands.push((
            self.passes.len(),
            FrameCommand::FreezeResource {
                resource: image_id.resource_id(),
                group: group_id,
            },
        ));
    }

    /// Adds a buffer to a resource group.
    pub fn freeze_buffer(&mut self, buffer_id: BufferId, group_id: ResourceGroupId) {
        self.commands.push((
            self.passes.len(),
            FrameCommand::FreezeResource {
                resource: buffer_id.resource_id(),
                group: group_id,
            },
        ));
    }

    pub fn destroy_image(&mut self, image_id: ImageId) {
        self.commands
            .push((self.passes.len(), FrameCommand::DestroyImage { image: image_id }));
    }

    pub fn destroy_buffer(&mut self, buffer_id: BufferId) {
        self.commands
            .push((self.passes.len(), FrameCommand::DestroyBuffer { buffer: buffer_id }));
    }

    /// Starts a graphics pass.
    pub fn start_graphics_pass<'frame>(&'frame mut self, name: &str) -> PassBuilder<'frame, 'a, UserContext> {
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

        pass.pass.eval = Some(PassEvaluationCallback::Present {
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
        let pass_index = self.passes.len();
        PassBuilder {
            frame: self,
            pass: ManuallyDrop::new(Pass::new(name, pass_index, ty, async_pass)),
        }
    }

    /*/// Dumps the frame to a JSON object.
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

                let xq_sync_json: Vec<_> = sync_debug_info.xq_sync_table.iter().map(|v| v.0).collect();

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
    }*/

    /*pub fn print_frame_info(&self) {
        let passes = &self.passes;
        //let temporaries = &self.inner.temporaries;

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
                println!(" access_mask:{:?}->{:?}", bmb.src_access_mask, bmb.dst_access_mask);
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
                    Some(ResourceAllocation::Transient { device_memory, offset }) => {
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
    }*/
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// AUTOMATIC SYNCHRONIZATION
////////////////////////////////////////////////////////////////////////////////////////////////////

/// TODO document
fn local_pass_index(serial: u64, frame_base_serial: u64) -> usize {
    assert!(serial > frame_base_serial);
    (serial - frame_base_serial - 1) as usize
}

// ---------------------------------------------------------------------------------------------

/// Stores the set of resources owned by a currently executing frame.
#[derive(Debug)]
struct FrameInFlight {
    signalled_serials: QueueSerialNumbers,
    //transient_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    command_pools: Vec<CommandAllocator>,
    semaphores: Vec<vk::Semaphore>,
    //image_views: Vec<vk::ImageView>,
    //framebuffers: Vec<vk::Framebuffer>,
}

/// Represents a GPU operation that may have not finished yet.
#[derive(Copy, Clone, Debug)]
pub struct GpuFuture {
    pub(crate) serials: QueueSerialNumbers,
}

impl Default for GpuFuture {
    fn default() -> Self {
        GpuFuture::new()
    }
}

impl GpuFuture {
    /// Returns an "empty" GPU future that represents an already completed operation.
    /// Waiting on this future always returns immediately.
    pub const fn new() -> GpuFuture {
        GpuFuture {
            serials: QueueSerialNumbers::new(),
        }
    }

    /// Returns a future representing the moment when the operations represented
    /// by both `self` and `other` have completed.
    pub fn join(&self, other: GpuFuture) -> GpuFuture {
        GpuFuture {
            serials: self.serials.join(other.serials),
        }
    }
}

/// Collected debugging information about a frame.
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

#[derive(Copy, Clone, Debug, Default)]
pub struct SubmitInfo {
    pub happens_after: GpuFuture,
    pub collect_debug_info: bool,
}

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
    wait_serials: QueueSerialNumbers,
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

////////////////////////////////////////////////////////////////////////////////////////////////////
// CONTEXT
////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Context {
    pub(crate) device: Arc<Device>,
    /// Free semaphores guaranteed to be in the unsignalled state.
    pub(crate) semaphore_pool: Vec<vk::Semaphore>,
    /// Timeline semaphores for each queue, used for cross-queue and inter-frame synchronization
    pub(crate) timelines: [vk::Semaphore; MAX_QUEUES],
    /// Array containing the last submitted pass serials for each queue
    pub(crate) last_signalled_serials: QueueSerialNumbers,
    /// Pool of recycled command pools.
    pub(crate) available_command_pools: Vec<CommandAllocator>,
    /// Array containing the last completed pass serials for each queue
    pub(crate) completed_serials: QueueSerialNumbers,
    /// The serial to be used for the next pass (used by `Frame`)
    pub(crate) last_sn: u64,
    /// Frames that are currently executing on the GPU.
    in_flight: VecDeque<FrameInFlight>,
    /// Number of submitted frames
    pub(crate) submitted_frame_count: u64,
    /// Number of completed frames
    pub(crate) completed_frame_count: u64,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO good luck with that
        f.debug_struct("Context").finish()
    }
}

impl Context {
    /// Creates a new context with the given device.
    pub fn with_device(device: Device) -> Context {
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
            device: Arc::new(device),
            timelines,
            last_signalled_serials: Default::default(),
            available_command_pools: vec![],
            completed_serials: Default::default(),
            semaphore_pool: vec![],
            last_sn: 0,
            submitted_frame_count: 0,
            completed_frame_count: 0,
            in_flight: VecDeque::new(),
        }
    }

    /// Returns the `graal::Device` owned by this context.
    pub fn device(&self) -> &Arc<Device> {
        &self.device
    }

    /// Returns the `ash::Device` owned by this context.
    /// Shorthand for `self.device().device`.
    pub fn vulkan_device(&self) -> &ash::Device {
        &self.device.device
    }

    /// Creates a binary semaphore (or return a previously used semaphore that is unsignalled).
    pub fn create_semaphore(&mut self) -> vk::Semaphore {
        if let Some(semaphore) = self.semaphore_pool.pop() {
            return semaphore;
        }

        unsafe {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            self.device.device.create_semaphore(&create_info, None).unwrap()
        }
    }

    /// Precondition: each semaphore in `semaphores` must be in the unsignalled state, or somehow
    /// be guaranteed to be in the unsignalled state the next time `create_semaphore` is called.
    fn recycle_semaphores(&mut self, mut semaphores: Vec<vk::Semaphore>) {
        self.semaphore_pool.append(&mut semaphores)
    }

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(&mut self, swapchain: &Swapchain) -> Result<SwapchainImage, vk::Result> {
        let image_available = self.create_semaphore();
        let (image_index, _suboptimal) = match self.device.vk_khr_swapchain.acquire_next_image(
            swapchain.handle,
            // TODO
            1_000_000_000,
            image_available,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // recycle the semaphore before returning
                self.semaphore_pool.push(image_available);
                return Err(err);
            }
        };

        let handle = swapchain.images[image_index as usize];
        let name = format!("swapchain {:?} image #{}", handle, image_index);
        let id = self.device.register_image_resource(ImageRegistrationInfo {
            resource: ResourceRegistrationInfo {
                name: &name,
                initial_wait: Some(SemaphoreWait {
                    semaphore: image_available,
                    owned: true,
                    dst_stage: Default::default(),
                    wait_kind: SemaphoreWaitKind::Binary,
                }),
                ownership: ResourceOwnership::External,
            },
            handle,
            format: swapchain.format,
        });

        Ok(SwapchainImage {
            swapchain_handle: swapchain.handle,
            image_info: ImageInfo { id, handle },
            image_index,
        })
    }

    /// Submits the given new frame for execution on the device.
    /// The execution of the frame can optionally be synchronized with the given future in `happens_after`.
    ///
    /// However, regardless of this, individual passes in the frame may still synchronize with earlier frames
    /// because of resource dependencies.
    pub fn submit_frame<UserContext>(
        &mut self,
        user_context: &mut UserContext,
        mut frame: Frame<UserContext>,
        submit_info: &SubmitInfo,
    ) {
        let frame_number = FrameNumber(self.submitted_frame_count + 1);
        self.device.start_frame(frame_number);

        //let base_sn = self.last_sn;
        //let wait_init = submit_info.happens_after.serials;

        let last_sn = {
            let mut device_objects = self.device.objects.lock().unwrap();
            let device_objects = &mut *device_objects;
            let resources = &mut device_objects.resources;
            let resource_groups = &mut device_objects.resource_groups;
            synchronize_and_track_resources(
                &self.device,
                resources,
                resource_groups,
                &mut frame,
                self.last_sn,
                submit_info.happens_after.serials,
            )
        };

        // wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the frames that are not in use anymore.
        self.wait_for_frames_in_flight();

        //////////////////////////////////////////////////////////////////////////////////
        // Submit commands to the device queues
        self.submit_frame_inner(user_context, frame);

        self.device.end_frame();
        self.last_sn = last_sn;
    }

    /// Returns whether the given frame, identified by its serial, has completed execution.
    pub fn is_frame_completed(&self, serial: FrameNumber) -> bool {
        self.completed_frame_count >= serial.0
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // COMMAND BUFFER SUBMISSION
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub(crate) fn wait(&self, serials: &QueueSerialNumbers) {
        let _span = trace_span!("Waiting for serials", ?serials);

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

    fn submit_command_batch(&mut self, q: usize, batch: &CommandBatch, used_semaphores: &mut Vec<vk::Semaphore>) {
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
            signal_semaphores.push(self.timelines[q as usize]);
            signal_semaphore_values.push(batch.signal_snn.serial());
            self.last_signalled_serials[q] = batch.signal_snn.serial();
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

        // setup queue timeline waits
        for (i, &w) in batch.wait_serials.iter().enumerate() {
            if w != 0 {
                wait_semaphores.push(self.timelines[i]);
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

        let queue = self.device.queues_info.queues[q as usize];
        unsafe {
            self.device
                .device
                .queue_submit(queue, &[submit_info], vk::Fence::null())
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
            .available_command_pools
            .iter()
            .position(|cmd_pool| cmd_pool.queue_family == queue_family)
        {
            // found one, remove it and return it
            self.available_command_pools.swap_remove(pos)
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

    fn submit_frame_inner<UserContext>(
        &mut self,
        user_context: &mut UserContext,
        mut frame: Frame<UserContext>,
    ) -> FrameSubmitResult {
        let _span = trace_span!("Submit frame").entered();

        let mut present_results = vec![];

        // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
        // Allocate and assign memory for all transient resources of this frame.
        //let transient_allocations =
        //    allocate_memory_for_transients(self, frame.base_serial, &frame.passes, &frame.temporaries);

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

            let wait_serials = if first_pass_of_queue[q] && frame.initial_wait > self.completed_serials {
                p.wait_serials.join(frame.initial_wait)
            } else {
                p.wait_serials
            };

            first_pass_of_queue[q] = false;

            // we need to wait if we have a binary semaphore, or if it's the first pass in this queue
            // and the user specified an initial wait before starting the frame.
            let needs_semaphore_wait = wait_serials > self.completed_serials || !p.external_semaphore_waits.is_empty();

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
                    // FIXME: this always submits empty command buffers
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
        self.in_flight.push_back(FrameInFlight {
            signalled_serials: self.last_signalled_serials,
            //transient_allocations,
            command_pools,
            semaphores: used_semaphores,
        });

        // one more frame submitted
        self.submitted_frame_count += 1;

        FrameSubmitResult {
            future: GpuFuture {
                serials: self.last_signalled_serials,
            },
            present_results,
        }
    }

    /// Recycles command pools returned by `submit_frame`.
    fn recycle_command_pools(&mut self, mut allocators: Vec<CommandAllocator>) {
        for a in allocators.iter_mut() {
            a.reset(&self.device.device)
        }
        self.available_command_pools.append(&mut allocators);
    }

    /// Waits for all but the last submitted frame to finish and then recycles their resources.
    /// Calls `cleanup_resources` internally.
    fn wait_for_frames_in_flight(&mut self) {
        let _span = trace_span!("Frame pacing").entered();

        // pacing
        // FIXME instead of always waiting for the 2 previous frames, introduce "frame groups" (to bikeshed)
        // that define the granularity of the pacing. (i.e. wait for whole frame groups instead of individual frames)
        // This is because in some workloads we submit a lot of small frames that target different surfaces (e.g. for compositing layers in kyute).
        while self.in_flight.len() >= 50 {
            // two frames in flight already, must wait for the oldest one
            let f = self.in_flight.pop_front().unwrap();

            self.wait(&f.signalled_serials);

            // update completed serials
            // we just waited on those serials, so we know they are completed
            self.completed_serials = f.signalled_serials;

            // Recycle the command pools allocated for the frame. The allocated command buffers
            // can then be reused for future submissions.
            self.recycle_command_pools(f.command_pools);

            // Recycle the semaphores. They are guaranteed to be unsignalled since the frame must have
            // waited on them.
            self.recycle_semaphores(f.semaphores);

            // TODO delayed allocation/automatic aliasing is being phased out. Replace with explicitly aliased resources and stream-ordered allocators.
            /*// free transient allocations
            for alloc in f.transient_allocations {
                trace!(?alloc, "free_memory");
                self.device.allocator.borrow_mut().free(alloc).unwrap();
            }*/

            // bump completed frame count
            self.completed_frame_count += 1;
        }

        // given the new completed serials, free resources that have expired
        unsafe {
            // SAFETY: we just waited for the passes to finish
            self.device
                .cleanup_resources(self.completed_serials, FrameNumber(self.completed_frame_count))
        }
    }

    pub fn current_frame_number(&self) -> FrameNumber {
        //assert!(self.is_building_frame, "not building a frame");
        FrameNumber(self.submitted_frame_count + 1)
    }

    pub fn wait_for(&mut self, future: GpuFuture) {
        self.wait(&future.serials);
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // TODO
    }
}
