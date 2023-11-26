use crate::device::MAX_QUEUES;
use std::{
    cmp::Ordering,
    ops::{Deref, DerefMut},
};

/// A set of timeline values to wait for, one for each queue.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
#[repr(transparent)]
pub struct TimelineValues(pub [u64; MAX_QUEUES]);

impl TimelineValues {
    pub const fn new() -> TimelineValues {
        TimelineValues([0; MAX_QUEUES])
    }

    pub fn from_queue_serial(queue: usize, value: u64) -> TimelineValues {
        let mut s = Self::new();
        s[queue] = value;
        s
    }

    pub fn serial(&self, queue: usize) -> u64 {
        self.0[queue]
    }
}

impl Deref for TimelineValues {
    type Target = [u64];

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for TimelineValues {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl PartialOrd for TimelineValues {
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

/*
/// A number that uniquely identifies a frame.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct FrameNumber(pub u64);
*/
