use crate::{
    argument::{ArgumentKind, Arguments, PushConstants, StaticPushConstants},
    device::Device,
    shader::GraphicsPipeline,
    vertex::{VertexBufferDescriptor, VertexInput},
    vk,
};
use std::{ffi::c_void, mem, ops::Deref, ptr};

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    device: &'a mut Device,
    submission: &'a mut graal::queue::Submission,
    command_buffer: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    bind_point: vk::PipelineBindPoint,
}

impl<'a> RenderEncoder<'a> {
    // SAFETY: TBD
    pub unsafe fn bind_graphics_pipeline(&self, pipeline: GraphicsPipeline) {
        self.device.device().cmd_bind_pipeline(
            self.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline(),
        );
    }

    /// Binds vertex buffers to the pipeline.
    pub fn bind_vertex_buffers<VI: VertexInput>(&self, vertex_input: &VI) {
        let buffer_count = vertex_input.vertex_buffers().count();
        let mut buffers = Vec::with_capacity(buffer_count);
        let mut offsets = Vec::with_capacity(buffer_count);
        let mut sizes = Vec::with_capacity(buffer_count);
        let mut strides = Vec::with_capacity(buffer_count);

        for vertex_buffer in vertex_input.vertex_buffers() {
            self.submission.use_buffer(
                vertex_buffer.buffer_range.buffer.handle().id,
                graal::queue::ResourceState::VERTEX_BUFFER,
            );
            buffers.push(vertex_buffer.buffer_range.buffer.handle().vk);
            offsets.push(vertex_buffer.buffer_range.offset as vk::DeviceSize);
            sizes.push(vertex_buffer.buffer_range.size as vk::DeviceSize);
            strides.push(vertex_buffer.stride as vk::DeviceSize);
        }

        unsafe {
            self.device.device().cmd_bind_vertex_buffers2(
                self.command_buffer,
                0,
                &buffers,
                &offsets,
                Some(&sizes),
                Some(&strides),
            );
        }
    }

    /// Binds a vertex buffer.
    pub fn bind_vertex_buffer(&self, vertex_buffer: &VertexBufferDescriptor) {
        self.submission.use_buffer(
            vertex_buffer.buffer_range.buffer.handle().id,
            graal::queue::ResourceState::VERTEX_BUFFER,
        );
        unsafe {
            self.device.device().cmd_bind_vertex_buffers2(
                self.command_buffer,
                vertex_buffer.binding,
                &[vertex_buffer.buffer_range.buffer.handle().vk],
                &[vertex_buffer.buffer_range.offset as vk::DeviceSize],
                Some(&[vertex_buffer.buffer_range.size as vk::DeviceSize]),
                Some(&[vertex_buffer.stride as vk::DeviceSize]),
            );
        }
    }

    /// Binds push constants.
    ///
    /// Usually this is called through a type-safe wrapper.
    pub unsafe fn bind_push_constants(&self, stages: vk::ShaderStageFlags, size: usize, data: *const c_void) {
        // Use the raw function pointer because the wrapper takes a &[u8] slice which we can't
        // get from a &[T] slice (unless we depend on `bytemuck` and require T: Pod, which is more trouble than its worth).
        (self.device.device().deref().fp_v1_0().cmd_push_constants)(
            self.command_buffer,
            self.pipeline_layout,
            stages,
            0,
            size as u32,
            data,
        );
    }

    /// Binds an argument block to the pipeline at the specified set.
    ///
    /// NOTE: this currently uses push descriptors.
    pub unsafe fn bind_arguments<A: Arguments>(&self, set: u32, arguments: &A) {
        enum DescriptorInfo {
            Image(vk::DescriptorImageInfo),
            Buffer(vk::DescriptorBufferInfo),
        }

        impl DescriptorInfo {
            fn as_image_ptr(&self) -> *const vk::DescriptorImageInfo {
                match self {
                    DescriptorInfo::Image(info) => info as *const _,
                    _ => panic!(),
                }
            }
            fn as_buffer_ptr(&self) -> *const vk::DescriptorBufferInfo {
                match self {
                    DescriptorInfo::Buffer(info) => info as *const _,
                    _ => panic!(),
                }
            }
        }

        // +1 for the inline data descriptor
        let descriptor_count = arguments.descriptors().count() + 1;
        let inline_data = arguments.inline_data();

        // NOTE: it's crucial that this vector never reallocates since `descriptor_writes` contains pointers to its elements
        let mut descriptor_infos = Vec::with_capacity(descriptor_count);
        let mut descriptor_writes = Vec::with_capacity(descriptor_count);

        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image {
                    image,
                    image_view,
                    resource_state,
                } => {
                    self.submission.use_image(image.id, resource_state);
                    descriptor_infos.push(DescriptorInfo::Image(vk::DescriptorImageInfo {
                        sampler: Default::default(),
                        image_view,
                        image_layout: resource_state.layout,
                    }));
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        // ignored for push descriptors
                        dst_set: Default::default(),
                        dst_binding: arg.binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: arg.descriptor_type,
                        p_image_info: descriptor_infos.last().unwrap().as_image_ptr(),
                        p_buffer_info: ptr::null(),
                        p_texel_buffer_view: ptr::null(),
                        ..Default::default()
                    });
                }
                ArgumentKind::Buffer {
                    buffer,
                    resource_state,
                    offset,
                    size,
                } => {
                    self.submission.use_buffer(buffer.id, resource_state);
                    descriptor_infos.push(DescriptorInfo::Buffer(vk::DescriptorBufferInfo {
                        buffer: buffer.handle(),
                        offset: offset as vk::DeviceSize,
                        range: size as vk::DeviceSize,
                    }));
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        // ignored for push descriptors
                        dst_set: Default::default(),
                        dst_binding: arg.binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: arg.descriptor_type,
                        p_image_info: ptr::null(),
                        p_buffer_info: descriptor_infos.last().unwrap().as_buffer_ptr(),
                        p_texel_buffer_view: ptr::null(),
                        ..Default::default()
                    });
                }
            }
        }

        let write_inline_uniform_block;

        // handle inline data
        let inline_data_size = mem::size_of_val(inline_data.as_ref());
        if inline_data_size != 0 {
            write_inline_uniform_block = vk::WriteDescriptorSetInlineUniformBlock {
                data_size: inline_data_size as u32,
                p_data: inline_data.as_ref() as *const A::InlineData as *const c_void,
                ..Default::default()
            };

            descriptor_writes.push(vk::WriteDescriptorSet {
                // ignored for push descriptors
                p_next: &write_inline_uniform_block as *const _ as *const _,
                dst_set: Default::default(),
                dst_binding: descriptor_count as u32 - 1,
                dst_array_element: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::INLINE_UNIFORM_BLOCK_EXT,
                p_image_info: ptr::null(),
                p_buffer_info: descriptor_infos.last().unwrap().as_buffer_ptr(),
                p_texel_buffer_view: ptr::null(),
                ..Default::default()
            });
        }

        unsafe {
            self.device.device().khr_push_descriptor().cmd_push_descriptor_set(
                self.command_buffer,
                self.bind_point,
                self.pipeline_layout,
                set,
                &descriptor_writes,
            );
        }
    }
}
