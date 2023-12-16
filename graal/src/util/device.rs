use crate::{BufferUsage, Device, MemoryLocation, TypedBuffer};
use std::{mem, ptr};

pub trait DeviceExt {
    fn create_array_buffer<T>(
        &self,
        name: &str,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> TypedBuffer<[T]>;

    fn upload_array_buffer<T>(&self, name: &str, usage: BufferUsage, data: &[T]) -> TypedBuffer<[T]>;
}

impl DeviceExt for Device {
    fn create_array_buffer<T>(
        &self,
        name: &str,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        len: usize,
    ) -> TypedBuffer<[T]> {
        let buffer = self.create_buffer(name, usage, memory_location, (mem::size_of::<T>() * len) as u64);
        TypedBuffer::new(buffer)
    }

    fn upload_array_buffer<T>(&self, name: &str, usage: BufferUsage, data: &[T]) -> TypedBuffer<[T]> {
        let buffer = self.create_array_buffer(name, usage, MemoryLocation::CpuToGpu, data.len());
        unsafe {
            // copy data to mapped buffer
            ptr::copy_nonoverlapping(
                data.as_ptr(),
                buffer
                    .mapped_data()
                    .expect("buffer should have been mapped in host memory"),
                data.len(),
            );
        }
        buffer
    }
}
