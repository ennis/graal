use ash::vk::Handle;
use std::{mem, fmt};
use std::ops::Deref;

/// A wrapper around a vulkan handle with unique semantics.
#[repr(transparent)]
pub(crate) struct UniqueHandle<T: Handle + Copy>(T);

impl<T: Handle + Copy> Drop for UniqueHandle<T> {
    fn drop(&mut self) {
        if self.0.as_raw() != 0 {
            panic!("non-null UniqueHandle was dropped")
        }
    }
}

impl <T: Handle + Copy + fmt::Debug> fmt::Debug for UniqueHandle<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}

impl<T: Handle + Copy> Default for UniqueHandle<T> {
    fn default() -> Self {
        UniqueHandle::null()
    }
}

impl<T: Handle + Copy> UniqueHandle<T> {
    /// Returns the null handle.
    pub fn null() -> UniqueHandle<T> {
        UniqueHandle(Handle::from_raw(0))
    }

    pub fn new(inner: T) -> UniqueHandle<T> {
        UniqueHandle(inner)
    }

    /// Returns the handle itself.
    pub fn get_inner(&self) -> T {
        self.0
    }

    /// Releases the handle.
    pub fn into_inner(mut self) -> T {
        self.take()
    }

    /// Releases the handle.
    pub fn take(&mut self) -> T {
        mem::replace(&mut self.0, T::from_raw(0))
    }

    pub fn is_null(&self) -> bool { self.0.as_raw() == 0 }
}

pub(crate) struct UniqueHandleVec<T: Handle + Copy>(Vec<T>);

impl<T: Handle + Copy> UniqueHandleVec<T> {
    pub fn new() -> UniqueHandleVec<T> {
        UniqueHandleVec(Vec::new())
    }

    ///
    pub fn is_empty(&self) -> bool {
        return self.0.is_empty();
    }

    ///
    pub fn push(&mut self, mut handle: UniqueHandle<T>) -> T {
        self.0.push(handle.get_inner());
        handle.into_inner()
    }

    ///
    pub fn remove(&mut self, index: usize) -> UniqueHandle<T> {
        UniqueHandle::new(self.0.remove(index))
    }

    ///
    pub fn swap_remove(&mut self, index: usize) -> UniqueHandle<T> {
        UniqueHandle::new(self.0.swap_remove(index))
    }

    ///
    pub fn into_inner(self) -> Vec<T> {
        self.0
    }
}

impl<T: Handle + Copy> Deref for UniqueHandleVec<T> {
    type Target = [T];

    fn deref(&self) -> &[T] {
        &self.0
    }
}
