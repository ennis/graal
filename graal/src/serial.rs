//! Serial (SN) and submission numbers (SNN).
//!
//! # About serial and submission numbers
//!
//! The **serial number** (SN) of a pass uniquely identifies it among all other passes submitted to a context,
//! regardless of the queue it was submitted on. SN 0 is considered invalid, and thus passes start at SN 1.
//!
//! The **submission number** (SNN) of a pass is composed of the SN plus the *queue index*,
//! which identifies the queue to which the pass has been or is going to be submitted.
//! They are written in the form `Q:SN` (e.g. `0:47` for queue #0, SN 47, `2:51` for queue #2, SN 51).
//! There cannot be two SNNs with the same SN but different queue indices (`0:50` and `1:50` is impossible).
//!
//! # Queue timelines
//!
//! Each queue has a timeline semaphore, which holds a monotonically increasing value that describes
//! the progression of passes submitted to the queue: when the timeline of queue Q reaches a value X,
//! all passes with SN <= X **that were submitted to Q** are guaranteed to have completed execution.
//!
//! For example, we can wait on timeline 0 for the value 3 to ensure that passes `0:1` and `0:3` have finished.
//! However, this wouldn't guarantee anything for pass `1:2`, submitted on a different queue.
//!
//! Timelines are more convenient than binary semaphores:
//! * we have a lot less semaphores to keep track of (one per queue instead of one per pass), and they are alive for the whole application.
//! * it's trivially easy to check if a pass with a given SNN (`Q:SN`) has finished: just get the value of `Q`'s timeline and check that it is greater than or equal to `SN`.
//!
//! We often use the phrase *waiting on an SNN* to signify waiting for the pass with that SNN on
//! its corresponding timeline semaphore. For instance, waiting on SNN `1:120` means waiting for the
//! value 120 to be signalled on the timeline semaphore of queue 1. When that value is reached, we can
//! be certain that pass SN 120 has finished executing.
//!
use std::cmp::Ordering;
use std::fmt;
use std::ops::{Deref, DerefMut};
use crate::MAX_QUEUES;

/// A submission number.
///
/// Combines the serial number of a pass and the queue it was submitted on.
/// See module-level documentation for more information.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct SubmissionNumber(u64);

impl SubmissionNumber {
    /// Creates a new submission number from a queue index and a serial.
    pub fn new(queue_index: usize, serial: u64) -> SubmissionNumber {
        assert!(queue_index < 4);
        assert!(serial < 1u64 << 62);
        SubmissionNumber(((queue_index as u64) << 62) | serial)
    }

    /// The queue that the pass is submitted on.
    pub const fn queue(&self) -> usize {
        (self.0 >> 62) as usize
    }

    /// The serial number of the pass.
    pub const fn serial(&self) -> u64 {
        self.0 & ((1 << 62) - 1)
    }

    /// Whether this submission number is valid.
    pub const fn is_valid(&self) -> bool {
        self.serial() != 0
    }
}

impl fmt::Debug for SubmissionNumber {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}:{}", self.queue(), self.serial())
    }
}

/// A set of serial numbers, one for each queue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct QueueSerialNumbers(pub [u64; MAX_QUEUES]);

impl QueueSerialNumbers {
    //
    pub const fn new() -> QueueSerialNumbers {
        QueueSerialNumbers([0; MAX_QUEUES])
    }

    // TODO better name?
    /*pub const fn has_nonzero_serial(&self) -> bool {
        let mut i = 0;
        while i < MAX_QUEUES {
            if self.0[i] != 0 {
                return true;
            }
            i += 1;
        }
        false
    }*/

    pub fn from_submission_number(snn: SubmissionNumber) -> QueueSerialNumbers {
        Self::from_queue_serial(snn.queue(), snn.serial())
    }

    pub fn from_queue_serial(queue: usize, serial: u64) -> QueueSerialNumbers {
        let mut s = Self::new();
        s[queue] = serial;
        s
    }

    pub fn serial(&self, queue: usize) -> u64 {
        self.0[queue]
    }

    pub fn join(&self, other: QueueSerialNumbers) -> QueueSerialNumbers {
        let mut r = *self;
        r.join_assign(other);
        r
    }

    pub fn join_assign(&mut self, other: QueueSerialNumbers) {
        for i in 0..MAX_QUEUES {
            self[i] = self[i].max(other[i]);
        }
    }

    pub fn join_serial(&self, snn: SubmissionNumber) -> QueueSerialNumbers {
        let mut r = *self;
        r[snn.queue()] = r[snn.queue()].max(snn.serial());
        r
    }

    pub fn iter(&self) -> impl Iterator<Item = &'_ u64> {
        self.0.iter()
    }

    pub(crate) fn is_single_source_same_queue_and_frame(
        &self,
        queue: usize,
        frame_base_serial: u64,
    ) -> bool {
        self.iter().enumerate().all(|(i, &sn)| {
            (i != queue && sn == 0) || (i == queue && (sn == 0 || sn > frame_base_serial))
        })
    }

    //pub fn iter_mut(&mut self) -> impl Iterator<Item = &'_ mut u64> {
    //    self.0.iter_mut()
    //}
}

impl Deref for QueueSerialNumbers {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for QueueSerialNumbers {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PartialOrd for QueueSerialNumbers {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        let before = self.0.iter().zip(other.0.iter()).all(|(&a, &b)| a <= b);

        let after = self.0.iter().zip(other.0.iter()).all(|(&a, &b)| a >= b);

        match (before, after) {
            (true, false) => Some(Ordering::Less),
            (false, true) => Some(Ordering::Greater),
            (true, true) => Some(Ordering::Equal),
            (false, false) => None,
        }
    }
}


/// A number that uniquely identifies a frame.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct FrameNumber(pub u64);