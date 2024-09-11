use crate::{Buffer, BufferUsage, Device, MemoryLocation};
use std::{mem, ptr, slice};

pub trait DeviceExt {
    fn upload<T: Copy>(&self, usage: BufferUsage, data: &T) -> Buffer<T>;
    fn create_array_buffer<T: Copy>(
        &self,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> Buffer<[T]>;
    fn upload_array_buffer<T: Copy>(&self, usage: BufferUsage, data: &[T]) -> Buffer<[T]>;
}

impl DeviceExt for Device {
    fn create_array_buffer<T: Copy>(
        &self,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> Buffer<[T]> {
        let buffer = self.create_buffer(usage, memory_location, (mem::size_of::<T>() * len) as u64);
        Buffer::new(buffer)
    }

    fn upload_array_buffer<T: Copy>(&self, usage: BufferUsage, data: &[T]) -> Buffer<[T]> {
        let buffer = self.create_array_buffer(usage, MemoryLocation::CpuToGpu, data.len());
        unsafe {
            // copy data to mapped buffer
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer
                    .as_mut_ptr(),
                data.len(),
            );
        }
        buffer
    }

    fn upload<T: Copy>(&self, usage: BufferUsage, data: &T) -> Buffer<T> {
        let buffer = self.upload_array_buffer(usage, slice::from_ref(data));
        Buffer::new(buffer.untyped)
    }
}
