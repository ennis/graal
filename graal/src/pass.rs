use crate::{MAX_QUEUES, Context};
use ash::vk;
use crate::context::ResourceId;
use std::fmt;

#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct SubmissionNumber(u64);

impl SubmissionNumber {
    pub fn new(queue_index: u8, serial: u64) -> SubmissionNumber {
        assert!(serial < 1u64 << 62);
        SubmissionNumber(((queue_index as u64) << 62) | serial)
    }

    pub const fn queue(&self) -> u8 {
        (self.0 >> 62) as u8
    }

    pub const fn serial(&self) -> u64 {
        self.0 & ((1 << 62) - 1)
    }

    pub const fn is_valid(&self) -> bool {
        self.serial() != 0
    }
}

impl fmt::Debug for SubmissionNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.queue(), self.serial())
    }
}

#[derive(Copy,Clone,Debug)]
pub(crate) struct ResourceAccess {
    pub(crate) id: ResourceId,
    pub(crate) access_mask: vk::AccessFlags,
}

#[derive(Copy,Clone,Debug)]
pub(crate) enum PassKind {
    Compute,
    Render,
    Transfer,
    Present {
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    },
}

pub(crate) struct Pass<'a> {
    pub(crate) name: String,
    /// Submission number of the pass.
    pub(crate) snn: SubmissionNumber,
    /// Index of the pass in the batch.
    pub(crate) batch_index: usize,
    /// @brief Predecessors of the pass (all passes that must happen before this one).
    pub(crate) preds: Vec<usize>,
    /// @brief Successors of the pass (all passes for which this task is a predecessor).
    pub(crate) succs: Vec<usize>,
    /// List of accesses made by the pass to resources.
    // FIXME Right now, this is used only for debugging purposes, and when allocating memory for the resources.
    // It probably could be removed.
    pub(crate) accesses: Vec<ResourceAccess>,
    /// The list of image memory barriers that must be applied before executing the pass.
    pub(crate) image_memory_barriers: Vec<vk::ImageMemoryBarrier>,
    /// The list of buffer memory barriers that must be applied before executing the pass.
    pub(crate) buffer_memory_barriers: Vec<vk::BufferMemoryBarrier>,
    /// Source stage mask for the pre-execution barrier.
    pub(crate) src_stage_mask: vk::PipelineStageFlags,
    /// Destination stage mask for the pre-execution barrier.
    pub(crate) input_stage_mask: vk::PipelineStageFlags,
    pub(crate) output_stage_mask: vk::PipelineStageFlags,
    /// Whether a signal operation must be performed on the queue after the pass.
    pub(crate) signal_after: bool,
    /// Whether the task should wait on semaphores.
    pub(crate) wait_before: bool,
    pub(crate) wait_serials: [u64; MAX_QUEUES],
    pub(crate) wait_dst_stages: [vk::PipelineStageFlags; MAX_QUEUES],
    pub(crate) wait_binary_semaphores: Vec<vk::Semaphore>,
    pub(crate) kind: PassKind,
    pub(crate) commands: Option<Box<dyn FnOnce(&Context, vk::CommandBuffer) + 'a>>,
}

impl<'a> Pass<'a> {
    pub(crate) fn is_present(&self) -> bool {
        match self.kind{
            PassKind::Present {..} => true,
            _ => false
        }
    }

    pub(crate) fn new_render_pass(name: &str, batch_index: usize, snn: SubmissionNumber) -> Pass<'a> {
        Self::new(name, batch_index, snn, PassKind::Render {})
    }

    pub(crate) fn new_compute_pass(name: &str, batch_index: usize, snn: SubmissionNumber) -> Pass<'a> {
        Self::new(name, batch_index, snn, PassKind::Compute {})
    }

    pub(crate) fn new_transfer_pass(name: &str, batch_index: usize, snn: SubmissionNumber) -> Pass<'a> {
        Self::new(name, batch_index, snn, PassKind::Transfer {})
    }

    pub(crate) fn new_present_pass(
        name: &str,
        batch_index: usize,
        snn: SubmissionNumber,
        swapchain: vk::SwapchainKHR,
        image_index: u32,
    ) -> Pass<'a> {
        Self::new(
            name,
            batch_index,
            snn,
            PassKind::Present {
                swapchain,
                image_index,
            },
        )
    }

    pub(crate) fn new(name: &str, batch_index: usize, snn: SubmissionNumber, kind: PassKind) -> Pass<'a> {
        Pass {
            name: name.to_string(),
            snn,
            preds: vec![],
            succs: vec![],
            accesses: vec![],
            image_memory_barriers: vec![],
            buffer_memory_barriers: vec![],
            src_stage_mask: Default::default(),
            input_stage_mask: Default::default(),
            output_stage_mask: Default::default(),
            signal_after: false,
            wait_before: false,
            wait_serials: [0; MAX_QUEUES],
            wait_dst_stages: [Default::default(); MAX_QUEUES],
            wait_binary_semaphores: Vec::new(),
            kind,
            batch_index,
            commands: None
        }
    }
}
