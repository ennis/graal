use drop_bomb::DropBomb;
use std::{
    ffi::c_void,
    mem,
    ops::{Deref, DerefMut, Range},
    ptr,
};

use crate::{
    vk, ArgumentKind, Arguments, Buffer, BufferRangeAny, ClearColorValue, ColorBlendEquation, CommandBuffer,
    ConservativeRasterizationMode, Device, GraphicsPipeline, Image, ImageCopyBuffer, ImageCopyView,
    ImageSubresourceLayers, IndexType, PipelineBindPoint, PrimitiveTopology, Rect2D, Rect3D, ResourceState,
    VertexBufferDescriptor, VertexInput,
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
    width: u32,
    height: u32,
    bomb: DropBomb,
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

    // TODO typed version
    pub fn fill_buffer(&mut self, buffer: &BufferRangeAny, data: u32) {
        self.command_buffer
            .use_buffer(buffer.buffer, ResourceState::TRANSFER_DST);
        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_fill_buffer(
                self.command_buffer.command_buffer,
                buffer.buffer.handle(),
                buffer.offset,
                buffer.size,
                data,
            );
        }
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        self.command_buffer.use_image(image, ResourceState::TRANSFER_DST);
        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_clear_color_image(
                self.command_buffer.command_buffer,
                image.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color_value.into(),
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                }],
            );
        }
    }
}

impl<'a> RenderEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer, width: u32, height: u32) -> Self {
        Self {
            base: Encoder {
                command_buffer,
                pipeline_layout: vk::PipelineLayout::null(),
                bind_point: vk::PipelineBindPoint::GRAPHICS,
            },
            width,
            height,
            bomb: DropBomb::new("RenderEncoder should be finished with `.finish()`"),
        }
    }

    // SAFETY: TBD
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        self.command_buffer.device().cmd_bind_pipeline(
            self.command_buffer.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            pipeline.pipeline(),
        );
        self.pipeline_layout = pipeline.pipeline_layout;
    }

    /// Binds vertex buffers to the pipeline.
    pub fn bind_vertex_buffers<VI: VertexInput>(&mut self, vertex_input: &VI) {
        let buffer_count = vertex_input.vertex_buffers().count();
        let mut buffers = Vec::with_capacity(buffer_count);
        let mut offsets = Vec::with_capacity(buffer_count);
        //let mut sizes = Vec::with_capacity(buffer_count);
        //let mut strides = Vec::with_capacity(buffer_count);

        for vertex_buffer in vertex_input.vertex_buffers() {
            self.command_buffer
                .use_buffer(vertex_buffer.buffer_range.buffer, ResourceState::VERTEX_BUFFER);
            buffers.push(vertex_buffer.buffer_range.buffer.handle());
            offsets.push(vertex_buffer.buffer_range.offset as vk::DeviceSize);
            //sizes.push(vertex_buffer.buffer_range.size as vk::DeviceSize);
            //strides.push(vertex_buffer.stride as vk::DeviceSize);
        }

        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_bind_vertex_buffers2(
                self.command_buffer.command_buffer,
                0,
                &buffers,
                &offsets,
                None,
                None,
            );
        }
    }

    /// Binds a vertex buffer.
    ///
    /// FIXME: we shouldn't be able to set the stride here, it should be part of the pipeline state.
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
                None,
                None,
            );
        }
    }

    pub fn bind_index_buffer(&mut self, index_type: IndexType, index_buffer: BufferRangeAny) {
        self.command_buffer
            .use_buffer(index_buffer.buffer, ResourceState::INDEX_BUFFER);
        self.command_buffer.flush_barriers();
        unsafe {
            self.command_buffer.device().cmd_bind_index_buffer(
                self.command_buffer.command_buffer,
                index_buffer.buffer.handle(),
                index_buffer.offset as vk::DeviceSize,
                index_type.into(),
            );
        }
    }

    /// Binds push constants.
    ///
    /// Usually this is called through a type-safe wrapper.
    pub unsafe fn bind_push_constants_raw(&mut self, bind_point: PipelineBindPoint, size: usize, data: *const c_void) {
        // Minimum push constant size guaranteed by Vulkan is 128 bytes.
        assert!(size <= 128, "push constant size must be <= 128 bytes");
        assert!(size % 4 == 0, "push constant size must be a multiple of 4 bytes");

        // None of the relevant drivers on desktop care about the actual stages,
        // only if it's graphics, compute, or ray tracing.
        let stages = match bind_point {
            PipelineBindPoint::Graphics => {
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
            }
            PipelineBindPoint::Compute => vk::ShaderStageFlags::COMPUTE,
            //_ => panic!("unsupported bind point"),
        };

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

    /// Binds push constants.
    pub fn bind_push_constants<P>(&mut self, bind_point: PipelineBindPoint, data: &P)
    where
        P: Copy + ?Sized,
    {
        unsafe {
            self.bind_push_constants_raw(bind_point, mem::size_of_val(data), data as *const _ as *const c_void);
        }
    }

    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        unsafe {
            self.command_buffer
                .device()
                .cmd_set_primitive_topology(self.command_buffer.command_buffer, topology.into());
        }
    }

    /*pub fn set_conservative_rasterization_mode(&mut self, mode: ConservativeRasterizationMode) {
        unsafe {
            self.command_buffer
                .device()
                .ext_extended_dynamic_state3()
                .cmd_set_conservative_rasterization_mode(self.command_buffer.command_buffer, mode.into());
        }
    }*/

    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        unsafe {
            self.command_buffer.device().cmd_set_viewport(
                self.command_buffer.command_buffer,
                0,
                &[vk::Viewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                }],
            );
        }
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        unsafe {
            self.command_buffer.device().cmd_set_scissor(
                self.command_buffer.command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x, y },
                    extent: vk::Extent2D { width, height },
                }],
            );
        }
    }

    pub fn set_color_blend_enable(&mut self, color_attachment: usize, enable: bool) {
        unsafe {
            let cb = self.command_buffer.command_buffer;
            let pfn = self.command_buffer.device().ext_extended_dynamic_state3();
            pfn.cmd_set_color_blend_enable(cb, color_attachment as u32, &[enable.into()]);
        }
    }

    pub fn set_color_blend_equation(&mut self, color_attachment: usize, color_blend_equation: &ColorBlendEquation) {
        unsafe {
            let cb = self.command_buffer.command_buffer;
            let pfn = self.command_buffer.device().ext_extended_dynamic_state3();
            let i = color_attachment as u32;
            pfn.cmd_set_color_blend_equation(
                cb,
                i,
                &[vk::ColorBlendEquationEXT {
                    src_color_blend_factor: color_blend_equation.src_color_blend_factor.to_vk_blend_factor(),
                    dst_color_blend_factor: color_blend_equation.dst_color_blend_factor.to_vk_blend_factor(),
                    color_blend_op: color_blend_equation.color_blend_op.to_vk_blend_op(),
                    src_alpha_blend_factor: color_blend_equation.src_alpha_blend_factor.to_vk_blend_factor(),
                    dst_alpha_blend_factor: color_blend_equation.dst_alpha_blend_factor.to_vk_blend_factor(),
                    alpha_blend_op: color_blend_equation.alpha_blend_op.to_vk_blend_op(),
                }],
            );
        }
    }

    pub fn clear_color(&mut self, attachment: u32, color: ClearColorValue) {
        self.clear_color_rect(attachment, color, Rect2D::from_xywh(0, 0, self.width, self.height));
    }

    pub fn clear_depth(&mut self, depth: f32) {
        self.clear_depth_rect(depth, Rect2D::from_xywh(0, 0, self.width, self.height));
    }

    pub fn clear_color_rect(&mut self, attachment: u32, color: ClearColorValue, rect: Rect2D) {
        unsafe {
            self.command_buffer.device().cmd_clear_attachments(
                self.command_buffer.command_buffer,
                &[vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    color_attachment: attachment,
                    clear_value: vk::ClearValue { color: color.into() },
                }],
                &[vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D {
                            x: rect.min.x,
                            y: rect.min.y,
                        },
                        extent: vk::Extent2D {
                            width: rect.width(),
                            height: rect.height(),
                        },
                    },
                }],
            );
        }
    }

    pub fn clear_depth_rect(&mut self, depth: f32, rect: Rect2D) {
        unsafe {
            self.command_buffer.device().cmd_clear_attachments(
                self.command_buffer.command_buffer,
                &[vk::ClearAttachment {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    color_attachment: 0,
                    clear_value: vk::ClearValue {
                        depth_stencil: vk::ClearDepthStencilValue { depth, stencil: 0 },
                    },
                }],
                &[vk::ClearRect {
                    base_array_layer: 0,
                    layer_count: 1,
                    rect: vk::Rect2D {
                        offset: vk::Offset2D {
                            x: rect.min.x,
                            y: rect.min.y,
                        },
                        extent: vk::Extent2D {
                            width: rect.width(),
                            height: rect.height(),
                        },
                    },
                }],
            );
        }
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        unsafe {
            self.command_buffer.device().cmd_draw(
                self.command_buffer.command_buffer,
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            );
        }
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        unsafe {
            self.command_buffer.device().cmd_draw_indexed(
                self.command_buffer.command_buffer,
                indices.len() as u32,
                instances.len() as u32,
                indices.start,
                base_vertex,
                instances.start,
            );
        }
    }

    pub fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        unsafe {
            self.command_buffer.device().ext_mesh_shader().cmd_draw_mesh_tasks(
                self.command_buffer.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    pub fn finish(mut self) {
        self.bomb.defuse();
        self.command_buffer.end_rendering();
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct BlitCommandEncoder<'a> {
    base: Encoder<'a>,
    bomb: DropBomb,
}

impl<'a> BlitCommandEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer) -> Self {
        Self {
            base: Encoder {
                command_buffer,
                pipeline_layout: vk::PipelineLayout::null(),
                bind_point: vk::PipelineBindPoint::COMPUTE,
            },
            bomb: DropBomb::new("BlitCommandEncoder should be finished with `.finish()`"),
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

    // TODO the call-site verbosity of this method is ridiculous, fix that
    pub fn blit_image(
        &mut self,
        src: &Image,
        src_subresource: ImageSubresourceLayers,
        src_region: Rect3D,
        dst: &Image,
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

    pub fn finish(mut self) {
        self.bomb.defuse()
    }
}

impl<'a> Deref for BlitCommandEncoder<'a> {
    type Target = Encoder<'a>;

    fn deref(&self) -> &Self::Target {
        &self.base
    }
}

impl<'a> DerefMut for BlitCommandEncoder<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.base
    }
}
