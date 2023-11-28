//! Buffers
use graal::{device::Device, vk};
use std::rc::Rc;

pub use graal::device::BufferHandle;

#[derive(Debug)]
struct BufferInner {
    device: Rc<Device>,
    size: usize,
    handle: BufferHandle,
    usage: vk::BufferUsageFlags,
}

/// Wrapper around a Vulkan buffer.
#[derive(Clone, Debug)]
pub struct BufferAny(Rc<BufferInner>);

impl BufferAny {
    /// Returns the size of the buffer in bytes.
    pub fn byte_size(&self) -> usize {
        self.0.size
    }

    /// Returns the usage flags of the buffer.
    pub fn usage(&self) -> vk::BufferUsageFlags {
        self.0.usage
    }

    /// Returns the buffer handle.
    pub fn handle(&self) -> BufferHandle {
        self.0.handle
    }

    /// Returns the device on which the buffer was created.
    pub fn device(&self) -> &Rc<Device> {
        &self.0.device
    }

    /// If the buffer is mapped in host memory, returns a pointer to the mapped memory.
    pub fn mapped_data(&self) -> Option<*mut u8> {
        self.0.handle.mapped_ptr.map(|ptr| ptr.as_ptr() as *mut u8)
    }
}

impl Drop for BufferInner {
    fn drop(&mut self) {
        unsafe {
            // SAFETY: we own the buffer and it is valid
            self.device.destroy_buffer(self.handle.id);
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Typed buffers.
#[derive(Clone)]
pub struct Buffer<T> {
    any: BufferAny,
    _marker: std::marker::PhantomData<T>,
}
