use crate::{context::{
    descriptor::DescriptorSet, submission::CommandContext, QueueSerialNumbers, SubmissionNumber,
}, Context, DescriptorSetInterface, ResourceId, MAX_QUEUES, format_aspect_mask};
use ash::{version::DeviceV1_0, vk};
use std::{
    cmp::Ordering,
    fmt,
    ops::{Index, IndexMut},
};

#[derive(Copy, Clone, Debug)]
pub(crate) struct ResourceAccess {
    pub(crate) id: ResourceId,
    pub(crate) access_mask: vk::AccessFlags,
}

#[derive(Copy, Clone, Debug)]
pub(crate) enum PassKind {
    Compute,
    Render,
    Transfer,
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
}

#[derive(Default)]
pub(crate) struct PipelineBarrier {
    pub(crate) src_stage_mask: vk::PipelineStageFlags,
    pub(crate) dst_stage_mask: vk::PipelineStageFlags,
    pub(crate) image_memory_barriers: Vec<vk::ImageMemoryBarrier>,
    pub(crate) buffer_memory_barriers: Vec<vk::BufferMemoryBarrier>
}

impl PipelineBarrier {
    pub(crate) fn get_or_create_image_memory_barrier(
        &mut self,
        handle: vk::Image,
        format: vk::Format) -> &mut vk::ImageMemoryBarrier
    {
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

    pub(crate) fn get_or_create_buffer_memory_barrier(&mut self, handle: vk::Buffer) -> &mut vk::BufferMemoryBarrier {
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
}

pub(crate) struct Pass<'a> {
    pub(crate) name: String,

    /// Submission number of the pass.
    pub(crate) snn: SubmissionNumber,

    /// Index of the pass in the frame.
    pub(crate) frame_index: usize,

    /// @brief Predecessors of the pass (all passes that must happen before this one).
    pub(crate) preds: Vec<usize>,

    /// @brief Successors of the pass (all passes for which this task is a predecessor).
    pub(crate) succs: Vec<usize>,

    /// List of accesses made by the pass to resources.
    // FIXME Right now, this is used only for debugging purposes, and when allocating memory for the resources.
    // It probably could be removed.
    pub(crate) accesses: Vec<ResourceAccess>,

    pub(crate) pre_execution_barrier: PipelineBarrier,

    /// Access types that should be flushed (made available)
    pub(crate) availability_mask: vk::AccessFlags,

    /// Whether a signal operation must be performed on the queue after the pass.
    pub(crate) signal_after: bool,

    /// Whether the task should wait on semaphores.
    pub(crate) wait_before: bool,
    pub(crate) wait_serials: QueueSerialNumbers,
    pub(crate) wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    pub(crate) wait_binary_semaphores: Vec<vk::Semaphore>,
    pub(crate) kind: PassKind,
    pub(crate) commands: Option<Box<dyn FnOnce(&mut CommandContext, vk::CommandBuffer) + 'a>>,
}

impl<'a> Pass<'a> {
    pub(crate) fn is_present(&self) -> bool {
        match self.kind {
            PassKind::Present { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn new(
        name: &str,
        frame_index: usize,
        snn: SubmissionNumber,
        kind: PassKind,
    ) -> Pass<'a> {
        Pass {
            name: name.to_string(),
            snn,
            preds: vec![],
            succs: vec![],
            accesses: vec![],
            pre_execution_barrier: Default::default(),
            availability_mask: Default::default(),
            signal_after: false,
            wait_before: false,
            wait_serials: Default::default(),
            wait_dst_stages: Default::default(),
            wait_binary_semaphores: Vec::new(),
            kind,
            frame_index,
            commands: None,
        }
    }
}
