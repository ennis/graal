use std::{
    ffi::c_void,
    mem,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
};

use crate::{
    map_texture_usage_to_layout, tracker::Tracker, vk, ArgumentKind, Arguments, Attachments, Buffer, Device, Format,
    Image, ImageView, RefCounted, ResourceId, ResourceUse, VertexInput,
};
use crate::tracker::emit_pipeline_barrier;

mod blit;
mod render;

pub use blit::BlitCommandEncoder;
pub use render::RenderEncoder;

////////////////////////////////////////////////////////////////////////////////////////////////////

enum DescriptorWrite {
    Image {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        image_view: vk::ImageView,
        format: Format,
        use_: ResourceUse,
    },
    Buffer {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        buffer: vk::Buffer,
        use_: ResourceUse,
        offset: u64,
        size: u64,
    },
    Sampler {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        sampler: vk::Sampler,
    },
}

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

unsafe fn do_cmd_push_descriptor_sets(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    pipeline_layout: vk::PipelineLayout,
    set: u32,
    desc_writes: &[DescriptorWrite],
) {
    let mut descriptors = Vec::with_capacity(desc_writes.len());
    let mut descriptor_writes = Vec::with_capacity(desc_writes.len());

    for dw in desc_writes {
        match *dw {
            DescriptorWrite::Buffer {
                binding,
                descriptor_type,
                buffer,
                offset,
                size,
                use_,
            } => {
                descriptors.push(DescriptorBufferOrImage {
                    buffer: vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: size,
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    // ignored for push descriptors
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_buffer_info: &descriptors.last().unwrap().buffer,
                    ..Default::default()
                });
            }
            DescriptorWrite::Image {
                binding,
                descriptor_type,
                image_view,
                use_,
                format,
            } => {
                let image_layout = map_texture_usage_to_layout(use_, format);
                descriptors.push(DescriptorBufferOrImage {
                    image: vk::DescriptorImageInfo {
                        sampler: Default::default(),
                        image_view,
                        image_layout,
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    // ignored for push descriptors
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_image_info: &descriptors.last().unwrap().image,
                    ..Default::default()
                });
            }
            DescriptorWrite::Sampler {
                sampler,
                binding,
                descriptor_type,
            } => {
                descriptors.push(DescriptorBufferOrImage {
                    image: vk::DescriptorImageInfo {
                        sampler,
                        image_view: Default::default(),
                        image_layout: Default::default(),
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_image_info: &descriptors.last().unwrap().image,
                    ..Default::default()
                });
            }
        }
    }

    // TODO inline uniforms
    unsafe {
        device.khr_push_descriptor().cmd_push_descriptor_set(
            command_buffer,
            bind_point,
            pipeline_layout,
            set,
            &descriptor_writes,
        );
    }
}

fn do_cmd_push_constants(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    pipeline_layout: vk::PipelineLayout,
    data: &[MaybeUninit<u8>],
) {
    let size = data.len();
    // Minimum push constant size guaranteed by Vulkan is 128 bytes.
    assert!(size <= 128, "push constant size must be <= 128 bytes");
    assert!(size % 4 == 0, "push constant size must be a multiple of 4 bytes");

    // None of the relevant drivers on desktop care about the actual stages,
    // only if it's graphics, compute, or ray tracing.
    let stages = match bind_point {
        vk::PipelineBindPoint::GRAPHICS => {
            vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
        }
        vk::PipelineBindPoint::COMPUTE => vk::ShaderStageFlags::COMPUTE,
        _ => panic!("unsupported bind point"),
    };

    // Use the raw function pointer because the wrapper takes a &[u8] slice which we can't
    // get from a &[T] slice (unless we depend on `bytemuck` and require T: Pod, which is more trouble than its worth).
    unsafe {
        (device.deref().fp_v1_0().cmd_push_constants)(
            command_buffer,
            pipeline_layout,
            stages,
            0,
            data.len() as u32,
            data.as_ptr() as *const c_void,
        );
    }
}

/// Common
struct EncoderBase<'a> {
    command_buffer: &'a mut CommandBuffer,
    push_constant_data: Vec<MaybeUninit<u8>>,
    descriptor_writes: Vec<DescriptorWrite>,
    tracker: Tracker,
}

impl<'a> EncoderBase<'a> {
    fn new(command_buffer: &'a mut CommandBuffer) -> EncoderBase<'a> {
        EncoderBase {
            command_buffer,
            push_constant_data: vec![],
            descriptor_writes: vec![],
            tracker: Tracker::new(),
        }
    }

    unsafe fn push_constants_raw(&mut self, len: usize, data: *const c_void) -> u32 {
        assert!(len <= 128, "push constant size must be <= 128 bytes");
        assert!(len % 4 == 0, "push constant size must be a multiple of 4 bytes");
        let offset = self.push_constant_data.len();
        self.push_constant_data.resize(offset + len, MaybeUninit::uninit());
        let slice = &mut self.push_constant_data[offset..];
        ptr::copy_nonoverlapping(data as *const u8, (*slice.as_mut_ptr()).as_mut_ptr(), len);
        offset as u32
    }

    unsafe fn push_constants<P>(&mut self, data: &P) -> u32
    where
        P: Copy + ?Sized,
    {
        self.push_constants_raw(mem::size_of_val(data), data as *const _ as *const c_void)
    }

    /// Binds an argument block to the pipeline at the specified set.
    ///
    /// NOTE: this currently uses push descriptors.
    unsafe fn record_arguments<A: Arguments>(&mut self, arguments: &A) -> (u32, u32) {
        let offset = self.descriptor_writes.len() as u32;
        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image { image_view, use_ } => {
                    self.use_image_view(image_view, use_);
                    self.descriptor_writes.push(DescriptorWrite::Image {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        image_view: image_view.handle(),
                        format: image_view.format(),
                        use_,
                    });
                }
                ArgumentKind::Buffer {
                    buffer,
                    use_,
                    offset,
                    size,
                } => {
                    self.use_buffer(buffer, use_);
                    self.descriptor_writes.push(DescriptorWrite::Buffer {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        buffer: buffer.handle(),
                        use_,
                        offset,
                        size,
                    });
                }
                ArgumentKind::Sampler { sampler } => {
                    self.descriptor_writes.push(DescriptorWrite::Sampler {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        sampler: sampler.handle(),
                    });
                }
            }
        }
        let count = self.descriptor_writes.len() as u32 - offset;
        (offset, count)
    }

    pub fn use_image(&mut self, image: &Image, state: ResourceUse) {
        self.command_buffer.refs.push(image.id().map(Into::into));
        self.tracker.use_image(image, state, false).expect("usage conflict");
    }

    pub fn use_image_view(&mut self, image_view: &ImageView, state: ResourceUse) {
        self.command_buffer.refs.push(image_view.parent_image.clone().map(Into::into));
        self.tracker
            .use_image_view(image_view, state, false)
            .expect("usage conflict");
    }

    pub fn use_buffer(&mut self, buffer: &Buffer, state: ResourceUse) {
        self.command_buffer.refs.push(buffer.id().map(Into::into));
        self.tracker.use_buffer(buffer, state, false).expect("usage conflict");
    }

    pub(super) fn flush_pipeline_barriers(&mut self) {
        let pb = self.command_buffer.tracker.merge(&self.tracker);
        if !pb.is_empty() {
            unsafe {
                emit_pipeline_barrier(self.command_buffer.device(), self.command_buffer.command_buffer, &pb);
            }
        }
        self.tracker.clear()
    }
}

/// Command buffers
pub struct CommandBuffer {
    device: Device,
    pub(super) refs: Vec<RefCounted<ResourceId>>,
    pub(super) command_buffer: vk::CommandBuffer,
    pub(super) tracker: Tracker,
}

impl CommandBuffer {
    pub(super) fn new(device: &Device, command_buffer: vk::CommandBuffer) -> CommandBuffer {
        Self {
            device: device.clone(),
            refs: vec![],
            command_buffer,
            tracker: Tracker::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}
