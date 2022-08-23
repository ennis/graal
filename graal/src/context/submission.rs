//! Code related to the submission of commands contained in frames to GPU queues (`vkQueueSubmit`, presentation).
use crate::{
    context::{
        frame::{Frame, FrameSubmitResult, PresentOperationResult},
        FrameInFlight, FrameInner, PassEvaluationCallback, SemaphoreSignal, SemaphoreSignalKind, SemaphoreWait,
        SemaphoreWaitKind, SEMAPHORE_WAIT_TIMEOUT_NS,
    },
    serial::{QueueSerialNumbers, SubmissionNumber},
    vk, Context, GpuFuture, MAX_QUEUES,
};
use std::{
    ffi::{c_void, CString},
    ops::Deref,
    ptr,
};
use tracing::trace_span;
