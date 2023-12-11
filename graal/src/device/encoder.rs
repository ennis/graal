use std::{
    ffi::c_void,
    mem,
    ops::{Deref, DerefMut},
    ptr,
};

use crate::{
    argument::{ArgumentKind, Arguments},
    device::Device,
    pipeline::GraphicsPipeline,
    resource_state::ResourceState,
    vertex::{VertexBufferDescriptor, VertexInput},
    vk, BufferAny, CommandBuffer, ImageAny, ImageSubresourceLayers, Rect3D,
};

/// Common
pub struct Encoder<'a> {
    command_buffer: &'a mut CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
    bind_point: vk::PipelineBindPoint,
}

impl<'a> Encoder<'a> {
    pub fn device(&self) -> &Device {
        self.command_buffer.device()
    }
}

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    base: Encoder<'a>,
}

impl<'a> Deref for RenderEncoder<'a> {
    type Target = Encoder<'a>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'a> DerefMut for RenderEncoder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}

impl<'a> Encoder<'a> {
    /// Binds an argument block to the pipeline at the specified set.
    ///
    /// NOTE: this currently uses push descriptors.
    pub unsafe fn bind_arguments<A: Arguments>(&mut self, set: u32, arguments: &A) {
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
        let descriptor_count = arguments.arguments().count() + 1;
        let inline_data = arguments.inline_data();

        // NOTE: it's crucial that this vector never reallocates since `descriptor_writes` contains pointers to its elements
        let mut descriptor_infos = Vec::with_capacity(descriptor_count);
        let mut descriptor_writes = Vec::with_capacity(descriptor_count);

        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image {
                    image_view,
                    resource_state,
                } => {
                    self.command_buffer.use_image_view(image_view, resource_state);
                    descriptor_infos.push(DescriptorInfo::Image(vk::DescriptorImageInfo {
                        sampler: Default::default(),
                        image_view: image_view.handle(),
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
                    self.command_buffer.use_buffer(buffer, resource_state);
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
                ArgumentKind::Sampler { sampler } => {
                    descriptor_infos.push(DescriptorInfo::Image(vk::DescriptorImageInfo {
                        sampler: sampler.handle(),
                        image_view: Default::default(),
                        image_layout: Default::default(),
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
            self.command_buffer
                .device()
                .khr_push_descriptor()
                .cmd_push_descriptor_set(
                    self.command_buffer.command_buffer,
                    self.bind_point,
                    self.pipeline_layout,
                    set,
                    &descriptor_writes,
                );
        }
    }
}

impl<'a> RenderEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer) -> Self {
        Self {
            base: Encoder {
                command_buffer,
                pipeline_layout: vk::PipelineLayout::null(),
                bind_point: vk::PipelineBindPoint::GRAPHICS,
            },
        }
    }

    // SAFETY: TBD
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        self.command_buffer.device().cmd_bind_pipeline(
            self.command_buffer.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline(),
        );
    }

    /// Binds vertex buffers to the pipeline.
    pub fn bind_vertex_buffers<VI: VertexInput>(&mut self, vertex_input: &VI) {
        let buffer_count = vertex_input.vertex_buffers().count();
        let mut buffers = Vec::with_capacity(buffer_count);
        let mut offsets = Vec::with_capacity(buffer_count);
        let mut sizes = Vec::with_capacity(buffer_count);
        let mut strides = Vec::with_capacity(buffer_count);

        for vertex_buffer in vertex_input.vertex_buffers() {
            self.command_buffer
                .use_buffer(vertex_buffer.buffer_range.buffer, ResourceState::VERTEX_BUFFER);
            buffers.push(vertex_buffer.buffer_range.buffer.handle());
            offsets.push(vertex_buffer.buffer_range.offset as vk::DeviceSize);
            sizes.push(vertex_buffer.buffer_range.size as vk::DeviceSize);
            strides.push(vertex_buffer.stride as vk::DeviceSize);
        }

        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_bind_vertex_buffers2(
                self.command_buffer.command_buffer,
                0,
                &buffers,
                &offsets,
                Some(&sizes),
                Some(&strides),
            );
        }
    }

    /// Binds a vertex buffer.
    pub fn bind_vertex_buffer(&mut self, vertex_buffer: &VertexBufferDescriptor) {
        self.command_buffer
            .use_buffer(vertex_buffer.buffer_range.buffer, ResourceState::VERTEX_BUFFER);
        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_bind_vertex_buffers2(
                self.command_buffer.command_buffer,
                vertex_buffer.binding,
                &[vertex_buffer.buffer_range.buffer.handle()],
                &[vertex_buffer.buffer_range.offset as vk::DeviceSize],
                Some(&[vertex_buffer.buffer_range.size as vk::DeviceSize]),
                Some(&[vertex_buffer.stride as vk::DeviceSize]),
            );
        }
    }

    /// Binds push constants.
    ///
    /// Usually this is called through a type-safe wrapper.
    pub unsafe fn bind_push_constants(&mut self, stages: vk::ShaderStageFlags, size: usize, data: *const c_void) {
        // Use the raw function pointer because the wrapper takes a &[u8] slice which we can't
        // get from a &[T] slice (unless we depend on `bytemuck` and require T: Pod, which is more trouble than its worth).
        (self.command_buffer.device().deref().fp_v1_0().cmd_push_constants)(
            self.command_buffer.command_buffer,
            self.pipeline_layout,
            stages,
            0,
            size as u32,
            data,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Copy, Clone, Debug)]
pub struct ImageCopyBuffer<'a> {
    pub buffer: &'a BufferAny,
    pub layout: ImageDataLayout,
}

#[derive(Copy, Clone, Debug)]
pub struct ImageDataLayout {
    pub offset: u64,
    /// In texels.
    pub row_length: Option<u32>,
    /// In lines.
    pub image_height: Option<u32>,
}

#[derive(Copy, Clone, Debug)]
pub struct ImageCopyView<'a> {
    pub image: &'a ImageAny,
    pub mip_level: u32,
    pub origin: vk::Offset3D,
    pub aspect: vk::ImageAspectFlags,
}

pub struct BlitCommandEncoder<'a> {
    base: Encoder<'a>,
}

impl<'a> BlitCommandEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer) -> Self {
        Self {
            base: Encoder {
                command_buffer,
                pipeline_layout: vk::PipelineLayout::null(),
                bind_point: vk::PipelineBindPoint::COMPUTE,
            },
        }
    }

    pub unsafe fn copy_image_to_image(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.base
            .command_buffer
            .use_image(source.image, ResourceState::TRANSFER_SRC);
        self.base
            .command_buffer
            .use_image(destination.image, ResourceState::TRANSFER_DST);

        let regions = [vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: source.aspect,
                mip_level: source.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: source.origin,
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect,
                mip_level: destination.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: destination.origin,
            extent: copy_size,
        }];

        self.base.command_buffer.flush_barriers();
        let device = self.base.command_buffer.device().raw();
        unsafe {
            device.cmd_copy_image(
                self.base.command_buffer.command_buffer,
                source.image.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination.image.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    pub unsafe fn copy_buffer_to_image(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.base
            .command_buffer
            .use_buffer(source.buffer, ResourceState::TRANSFER_SRC);
        self.base
            .command_buffer
            .use_image(destination.image, ResourceState::TRANSFER_DST);

        let regions = [vk::BufferImageCopy {
            buffer_offset: source.layout.offset,
            buffer_row_length: source.layout.row_length.unwrap_or(0),
            buffer_image_height: source.layout.image_height.unwrap_or(0),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect,
                mip_level: destination.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: destination.origin,
            image_extent: copy_size,
        }];

        self.base.command_buffer.flush_barriers();
        let device = self.base.command_buffer.device().raw();
        unsafe {
            device.cmd_copy_buffer_to_image(
                self.base.command_buffer.command_buffer,
                source.buffer.handle(),
                destination.image.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    pub fn blit_image(
        &mut self,
        src: &ImageAny,
        src_subresource: ImageSubresourceLayers,
        src_region: Rect3D,
        dst: &ImageAny,
        dst_subresource: ImageSubresourceLayers,
        dst_region: Rect3D,
        filter: vk::Filter,
    ) {
        self.base.command_buffer.use_image(src, ResourceState::TRANSFER_SRC);
        self.base.command_buffer.use_image(dst, ResourceState::TRANSFER_DST);

        let blits = [vk::ImageBlit {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: src_subresource.aspect_mask,
                mip_level: src_subresource.mip_level,
                base_array_layer: src_subresource.base_array_layer,
                layer_count: src_subresource.layer_count,
            },
            src_offsets: [
                vk::Offset3D {
                    x: src_region.min.x,
                    y: src_region.min.y,
                    z: src_region.min.z,
                },
                vk::Offset3D {
                    x: src_region.max.x,
                    y: src_region.max.y,
                    z: src_region.max.z,
                },
            ],
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: dst_subresource.aspect_mask,
                mip_level: dst_subresource.mip_level,
                base_array_layer: dst_subresource.base_array_layer,
                layer_count: dst_subresource.layer_count,
            },
            dst_offsets: [
                vk::Offset3D {
                    x: dst_region.min.x,
                    y: dst_region.min.y,
                    z: dst_region.min.z,
                },
                vk::Offset3D {
                    x: dst_region.max.x,
                    y: dst_region.max.y,
                    z: dst_region.max.z,
                },
            ],
        }];

        self.base.command_buffer.flush_barriers();
        let device = self.base.command_buffer.device().raw();
        unsafe {
            device.cmd_blit_image(
                self.base.command_buffer.command_buffer,
                src.handle(),
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blits,
                filter,
            );
        }
    }
}
