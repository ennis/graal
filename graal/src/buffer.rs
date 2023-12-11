//! Buffers
use std::{ffi::c_void, ptr::NonNull};

use crate::{
    device::{Device, RefCounted},
    vk, BufferId,
};

/// Wrapper around a Vulkan buffer.
#[derive(Debug)]
pub struct BufferAny {
    device: Device,
    id: RefCounted<BufferId>,
    handle: vk::Buffer,
    size: usize,
    usage: vk::BufferUsageFlags,
    mapped_ptr: Option<NonNull<c_void>>,
}

impl BufferAny {
    pub(super) fn new(
        device: Device,
        id: RefCounted<BufferId>,
        handle: vk::Buffer,
        size: usize,
        usage: vk::BufferUsageFlags,
        mapped_ptr: Option<NonNull<c_void>>,
    ) -> Self {
        Self {
            device,
            id,
            handle,
            size,
            usage,
            mapped_ptr,
        }
    }

    pub fn id(&self) -> RefCounted<BufferId> {
        self.id.clone()
    }

    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> usize {
        self.size
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.usage
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> vk::Buffer {
        self.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut u8> {
        self.mapped_ptr.map(|ptr| ptr.as_ptr() as *mut u8)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct BufferRangeAny<'a> {
    pub buffer: &'a BufferAny,
    pub offset: usize,
    pub size: usize,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Typed buffers.
pub struct Buffer<T> {
    any: BufferAny,
    _marker: std::marker::PhantomData<T>,
}

#[derive(Copy, Clone)]
pub struct BufferRange<'a, T> {
    pub buffer: &'a Buffer<T>,
    pub offset: usize,
    pub size: usize,
}
