use crate::{
    context::{submission::CommandContext, QueueSerialNumbers, SubmissionNumber},
    format_aspect_mask, ResourceId, MAX_QUEUES,
};
use ash::vk;

#[derive(Copy, Clone, Debug)]
pub(crate) struct ResourceAccess {
    pub(crate) id: ResourceId,
    pub(crate) access_mask: vk::AccessFlags,
}

pub(crate) enum PassCommands<'a> {
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
    Queue(Box<dyn FnOnce(&mut CommandContext, vk::Queue) + 'a>),
    CommandBuffer(Box<dyn FnOnce(&mut CommandContext, vk::CommandBuffer) + 'a>),
}

#[derive(Copy,Clone,Debug,Eq,PartialEq)]
pub enum SemaphoreWaitKind {
    Binary,
    Timeline(u64)
}

#[derive(Copy,Clone,Debug,Eq,PartialEq)]
pub enum SemaphoreSignalKind {
    Binary,
    Timeline(u64)
}

/// Represents a semaphore wait operation outside of the queue timelines.
#[derive(Clone,Debug)]
pub(crate) struct SemaphoreWait {
    /// The semaphore in question
    pub(crate) semaphore: vk::Semaphore,
    /// Whether the semaphore is internally managed (owned by the context).
    /// If true, the semaphore will be reclaimed by the context after it is consumed (waited on).
    pub(crate) owned: bool,
    /// Destination stage
    pub(crate) dst_stage: vk::PipelineStageFlags,
    /// The kind of wait operation.
    pub(crate) wait_kind: SemaphoreWaitKind
}

#[derive(Clone,Debug)]
pub(crate) struct SemaphoreSignal {
    pub(crate) semaphore: vk::Semaphore,
    pub(crate) signal_kind: SemaphoreSignalKind
}

pub(crate) struct Pass<'a> {
    pub(crate) name: String,

    /// Submission number of the pass.
    pub(crate) snn: SubmissionNumber,

    /// Index of the pass in the frame.
    pub(crate) frame_index: usize,

    /// Predecessors of the pass (all passes that must happen before this one).
    pub(crate) preds: Vec<usize>,

    /// Successors of the pass (all passes for which this task is a predecessor).
    //pub(crate) succs: Vec<usize>,

    /// List of accesses made by the pass to resources.
    // FIXME Right now, this is used only for debugging purposes, and when allocating memory for the resources.
    // It probably could be removed.
    pub(crate) accesses: Vec<ResourceAccess>,

    /// Whether the queue timeline semaphores must be signalled after the pass.
    pub(crate) signal_queue_timelines: bool,

    pub(crate) src_stage_mask: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask: vk::PipelineStageFlags,
    pub(crate) image_memory_barriers: Vec<vk::ImageMemoryBarrier>,
    pub(crate) buffer_memory_barriers: Vec<vk::BufferMemoryBarrier>,

    pub(crate) wait_serials: QueueSerialNumbers,
    pub(crate) wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],

    pub(crate) external_semaphore_waits: Vec<SemaphoreWait>,
    pub(crate) external_semaphore_signals: Vec<SemaphoreSignal>,

    pub(crate) commands: Option<PassCommands<'a>>,
}

impl<'a> Pass<'a> {
    pub(crate) fn get_or_create_image_memory_barrier(
        &mut self,
        handle: vk::Image,
        format: vk::Format,
    ) -> &mut vk::ImageMemoryBarrier {
        if let Some(b) = self
            .image_memory_barriers
            .iter_mut()
            .position(|b| b.image == handle)
        {
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

    pub(crate) fn get_or_create_buffer_memory_barrier(
        &mut self,
        handle: vk::Buffer,
    ) -> &mut vk::BufferMemoryBarrier {
        if let Some(b) = self
            .buffer_memory_barriers
            .iter_mut()
            .position(|b| b.buffer == handle)
        {
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

    pub(crate) fn new(name: &str, frame_index: usize, snn: SubmissionNumber) -> Pass<'a> {
        Pass {
            name: name.to_string(),
            snn,
            preds: vec![],
            //succs: vec![],
            accesses: vec![],
            signal_queue_timelines: false,
            src_stage_mask: Default::default(),
            dst_stage_mask: Default::default(),
            image_memory_barriers: vec![],
            buffer_memory_barriers: vec![],
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            external_semaphore_waits: vec![],
            external_semaphore_signals: vec![],
            frame_index,
            commands: None,
        }
    }
}
