//! Contains code related to the construction of frames and passes.
use crate::{
    context::{
        is_write_access, BufferId, Frame, FrameInner, GpuFuture, ImageId, Pass, PassEvaluationCallback,
        RecordingContext, ResourceAccessDetails, ResourceId, ResourceKind, SemaphoreSignal, SemaphoreSignalKind,
        SemaphoreWait, SemaphoreWaitKind, SyncDebugInfo, TemporarySet,
    },
    device::{AccessTracker, BufferResource, ImageResource, ResourceAllocation},
    format_aspect_mask,
    serial::{FrameNumber, QueueSerialNumbers, SubmissionNumber},
    vk,
    vk::Handle,
    Context, Device, ResourceGroupId, ResourceOwnership, SwapchainImage, MAX_QUEUES,
};
use slotmap::Key;
use std::{collections::HashSet, fmt, mem, mem::ManuallyDrop};
use tracing::trace_span;
