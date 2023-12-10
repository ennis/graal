//! Buffers
use graal::{device::Device, vk};
use std::rc::Rc;

pub use graal::device::BufferHandle;

/// Wrapper around a Vulkan buffer.
#[derive(Debug)]
pub struct BufferAny {
    device: Device,
    handle: BufferHandle,
    size: usize,
    usage: vk::BufferUsageFlags,
}

impl BufferAny {
    pub(super) fn new(device: Device, handle: BufferHandle, size: usize, usage: vk::BufferUsageFlags) -> Self {
        Self {
            device,
            handle,
            size,
            usage,
        }
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
    pub fn handle(&self) -> BufferHandle {
        self.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut u8> {
        self.handle.mapped_ptr.map(|ptr| ptr.as_ptr() as *mut u8)
    }
}

impl Drop for BufferAny {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: we own the buffer and it is valid
            self.device.destroy_buffer(self.handle.id);
        }
    }
}

pub struct BufferRangeAny<'a> {
    pub buffer: &'a BufferAny,
    pub offset: usize,
    pub size: usize,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Typed buffers.
#[derive(Clone)]
pub struct Buffer<T> {
    any: BufferAny,
    _marker: std::marker::PhantomData<T>,
}

pub struct BufferRange<'a, T> {
    pub buffer: &'a Buffer<T>,
    pub offset: usize,
    pub size: usize,
}
