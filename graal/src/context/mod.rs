mod submit;
mod sync;

use crate::{
    context::sync::synchronize_and_track_resources,
    device::{
        AccessTracker, BufferResource, Device, ImageResource, Resource, ResourceAllocation, ResourceGroupMap,
        ResourceKind, ResourceMap, ResourceTrackingInfo,
    },
    serial::{DeviceProgress, FrameNumber, SubmissionNumber},
    BufferId, ImageId, ImageInfo, ImageRegistrationInfo, ResourceGroupId, ResourceId, ResourceOwnership,
    ResourceRegistrationInfo, Swapchain, SwapchainImage, MAX_QUEUES,
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
    time::Duration,
};
use tracing::trace_span;

pub use crate::context::submit::RecordingContext;

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
pub struct Pass<'a, UserContext> {
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

    wait_serials: DeviceProgress,
    wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
}

// TODO: verify this. The only thing that's not sync are the `*const pNext` fields in `vk::ImageMemoryBarrier`, and those are always set to null anyway.
unsafe impl<'a, UserContext> Send for Pass<'a, UserContext> {}

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

pub struct PassBuilder<'a, UserContext> {
    /// Name of this pass, for debugging purposes.
    name: String,

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
}

impl<'a, UserContext> PassBuilder<'a, UserContext> {
    /// Creates a new pass description.
    pub fn new() -> PassBuilder<'a, UserContext> {
        PassBuilder {
            name: "".to_string(),
            ty: PassType::Graphics,
            async_queue: true,
            eval: None,
            accesses: vec![],
            external_semaphore_waits: vec![],
            external_semaphore_signals: vec![],
        }
    }

    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    pub fn queue(mut self, ty: PassType) -> Self {
        self.ty = ty;
        self
    }

    /// Adds a semaphore wait operation: the pass will first wait for the specified semaphore to be signalled
    /// before starting.
    pub fn external_semaphore_wait(
        mut self,
        semaphore: vk::Semaphore,
        dst_stage: vk::PipelineStageFlags,
        wait_kind: SemaphoreWaitKind,
    ) -> Self {
        self.external_semaphore_waits.push(SemaphoreWait {
            semaphore,
            owned: false,
            dst_stage,
            wait_kind,
        });
        self
    }

    /// Adds a semaphore signal operation: when finished, the pass will signal the specified semaphore.
    pub fn external_semaphore_signal(mut self, semaphore: vk::Semaphore, signal_kind: SemaphoreSignalKind) -> Self {
        self.external_semaphore_signals
            .push(SemaphoreSignal { semaphore, signal_kind });

        self
    }

    /// Registers an image access made by this pass.
    pub fn image_dependency(
        mut self,
        id: ImageId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
        initial_layout: vk::ImageLayout,
        final_layout: vk::ImageLayout,
    ) -> Self {
        self.accesses.push(ResourceAccess {
            id: id.resource_id(),
            access_mask,
            stage_mask,
            initial_layout,
            final_layout,
        });
        self
    }

    pub fn buffer_dependency(
        mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        stage_mask: vk::PipelineStageFlags,
    ) -> Self {
        self.accesses.push(ResourceAccess {
            id: id.resource_id(),
            access_mask,
            stage_mask,
            initial_layout: vk::ImageLayout::UNDEFINED,
            final_layout: vk::ImageLayout::UNDEFINED,
        });
        self
    }

    /// Sets the command buffer recording function for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn record_callback(
        mut self,
        record_cb: impl FnOnce(&mut RecordingContext, &mut UserContext, vk::CommandBuffer) + 'a,
    ) -> Self {
        self.eval = Some(PassEvaluationCallback::CommandBuffer(Box::new(record_cb)));
        self
    }

    pub fn submit_callback(
        mut self,
        submit_cb: impl FnOnce(&mut RecordingContext, &mut UserContext, vk::Queue) + 'a,
    ) -> Self {
        self.eval = Some(PassEvaluationCallback::Queue(Box::new(submit_cb)));
        self
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

#[derive(Clone, Debug)]
pub struct PresentOperationResult {
    pub swapchain: vk::SwapchainKHR,
    pub result: vk::Result,
}

/// The result of a frame submission.
#[derive(Clone, Debug)]
pub struct SubmitResult {
    /// Progress values for each queue (reached serials) when the frame will have completed execution.
    ///
    /// To block until the frame is completed, use `Device::wait` with these progress values.
    pub progress: DeviceProgress,

    /// Results of `vkQueuePresent` calls made during submission of the frame.
    pub present_results: Vec<PresentOperationResult>,
}

pub struct Frame<'a, UserContext> {
    passes: Vec<Pass<'a, UserContext>>,
    initial_wait: DeviceProgress,
    commands: Vec<(usize, FrameCommand)>,
}

impl<'a, UserContext> Default for Frame<'a, UserContext> {
    fn default() -> Self {
        Frame::new()
    }
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

    pub fn is_empty(&self) -> bool {
        self.passes.is_empty() && self.commands.is_empty()
    }

    /// Adds a dependency on the completion of previous queue operations.
    ///
    /// Execution of the *whole* frame will wait for the operation represented by the future to complete.
    pub fn add_dependency(&mut self, progress: DeviceProgress) {
        self.initial_wait = self.initial_wait.join(progress);
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

    pub fn add_pass(&mut self, pass: PassBuilder<'a, UserContext>) {
        let pass_index = self.passes.len();
        self.passes.push(Pass {
            name: pass.name,
            frame_index: pass_index,
            ty: pass.ty,
            async_queue: pass.async_queue,
            eval: pass.eval,
            accesses: pass.accesses,
            external_semaphore_waits: pass.external_semaphore_waits,
            external_semaphore_signals: pass.external_semaphore_signals,
            snn: Default::default(),
            preds: vec![],
            signal_queue_timelines: false,
            src_stage_mask: Default::default(),
            dst_stage_mask: Default::default(),
            image_memory_barriers: vec![],
            buffer_memory_barriers: vec![],
            global_memory_barrier: None,
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
        })
    }

    pub fn destroy_image(&mut self, image_id: ImageId) {
        self.commands
            .push((self.passes.len(), FrameCommand::DestroyImage { image: image_id }));
    }

    pub fn destroy_buffer(&mut self, buffer_id: BufferId) {
        self.commands
            .push((self.passes.len(), FrameCommand::DestroyBuffer { buffer: buffer_id }));
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&mut self, name: &str, image: &SwapchainImage) {
        let mut pass = PassBuilder::new().queue(PassType::Present).image_dependency(
            image.image_info.id,
            vk::AccessFlags::MEMORY_READ,
            vk::PipelineStageFlags::ALL_COMMANDS, // ?
            vk::ImageLayout::PRESENT_SRC_KHR,
            vk::ImageLayout::PRESENT_SRC_KHR,
        );
        pass.eval = Some(PassEvaluationCallback::Present {
            swapchain: image.swapchain_handle,
            image_index: image.image_index,
        });
        self.add_pass(pass);
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
    signalled_serials: DeviceProgress,
    //transient_allocations: Vec<gpu_allocator::vulkan::Allocation>,
    command_pools: Vec<submit::CommandAllocator>,
    semaphores: Vec<vk::Semaphore>,
    //image_views: Vec<vk::ImageView>,
    //framebuffers: Vec<vk::Framebuffer>,
}

/// Collected debugging information about a frame.
struct SyncDebugInfo {
    tracking: slotmap::SecondaryMap<ResourceId, ResourceTrackingInfo>,
    xq_sync_table: [DeviceProgress; MAX_QUEUES],
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
    pub happens_after: Option<DeviceProgress>,
    pub wait_before_submit: Option<DeviceProgress>,
    pub collect_debug_info: bool,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// CONTEXT
////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct Context {
    device: Arc<Device>,
    /// Free semaphores guaranteed to be in the unsignalled state.
    semaphore_pool: Vec<vk::Semaphore>,
    /// Command buffer submission state.
    submit_state: submit::SubmitState,
    /// The serial of the last submitted pass.
    ///
    /// This is zero initially, which is a special value indicating that no pass was submitted.
    last_sn: u64,
    /// Number of submitted frames
    submitted_frame_count: u64,
}

impl fmt::Debug for Context {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        // TODO good luck with that
        f.debug_struct("Context").finish()
    }
}

impl Context {
    /// Creates a new context with the given device.
    pub(crate) unsafe fn new(device: Device) -> Context {
        Context {
            device: Arc::new(device),
            submit_state: submit::SubmitState::new(),
            semaphore_pool: vec![],
            last_sn: 0,
            submitted_frame_count: 0,
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
    unsafe fn recycle_semaphores(&mut self, mut semaphores: Vec<vk::Semaphore>) {
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
    ///
    /// # Arguments
    /// * user_context user context passed to the recording & queue callbacks
    /// * frame the frame to submit
    /// * happens_after an optional progress value to synchronize the frame with on the device
    /// * wait_before_submit an optional progress value to wait (on the CPU) for before submitting the work. Typically used for pacing purposes.
    pub fn submit_frame<UserContext>(
        &mut self,
        user_context: &mut UserContext,
        mut frame: Frame<UserContext>,
        submit_info: &SubmitInfo,
    ) -> SubmitResult {
        let frame_number = FrameNumber(self.submitted_frame_count + 1);
        self.device.enter_frame(frame_number);

        //let base_sn = self.last_sn;
        //let wait_init = submit_info.happens_after.serials;

        // update known resource tracking information to their states after this frame
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
                submit_info.happens_after.unwrap_or_default(),
            )
        };

        // wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the frames that are not in use anymore.

        // Maximum time to wait for batches to finish.
        const FRAME_PACING_TIMEOUT: Duration = Duration::from_secs(5);
        self.retire_completed_frames(submit_info.wait_before_submit, FRAME_PACING_TIMEOUT)
            .expect("failed to wait for a previous frame");

        //////////////////////////////////////////////////////////////////////////////////
        // Submit commands to the device queues
        let result = self.submit_to_device_queues(user_context, frame);

        // one more frame submitted
        self.submitted_frame_count += 1;
        self.last_sn = last_sn;

        self.device.exit_frame();
        result
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // COMMAND BUFFER SUBMISSION
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub fn current_frame_number(&self) -> FrameNumber {
        //assert!(self.is_building_frame, "not building a frame");
        FrameNumber(self.submitted_frame_count + 1)
    }
}

impl Drop for Context {
    fn drop(&mut self) {
        // TODO
    }
}
