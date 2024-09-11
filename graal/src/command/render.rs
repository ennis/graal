//! Render command encoders
use std::{mem, mem::MaybeUninit, ops::Range, ptr, slice};

use ash::vk;

use crate::{
    is_depth_and_stencil_format, Barrier, BufferRangeUntyped, ClearColorValue, ColorAttachment, CommandStream,
    DepthStencilAttachment, Descriptor, Device, GpuResource, GraphicsPipeline, Rect2D,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    stream: &'a mut CommandStream,
    command_buffer: vk::CommandBuffer,
    render_area: vk::Rect2D,
    pipeline_layout: vk::PipelineLayout,
}

impl<'a> RenderEncoder<'a> {
    pub fn device(&self) -> &Device {
        self.stream.device()
    }

    /*pub fn use_image(&mut self, image: &Image, access: ImageAccess) {
        let image_id = image.id();
        if let Some(entry) = self.used_images.get_mut(&image_id) {
            if entry.1 != access {
                panic!(
                    "Image {:?} is already used in render pass with conflicting access {:?} (new access: {:?})",
                    image_id, entry.1, access
                );
            }
        } else {
            self.used_images.insert(image_id, (image.clone(), access));
        }
    }*/

    /*pub fn use_buffer(&mut self, buffer: &BufferUntyped, access: BufferAccess) {
        let buffer_id = buffer.id();
        if let Some(entry) = self.used_buffers.get_mut(&buffer_id) {
            if entry.1 != access {
                panic!(
                    "Buffer {:?} is already used in render pass with conflicting access {:?} (new access: {:?})",
                    buffer_id, entry.1, access
                );
            }
        } else {
            self.used_buffers.insert(buffer_id, (buffer.clone(), access));
        }
    }*/

    /*pub fn use_image_view(&mut self, image_view: &ImageView, state: ImageAccess) {
        let image_view_id = image_view.id();
        self.use_image(image_view.image(), state);
        self.used_image_views.insert(image_view_id, image_view.clone());
    }*/

    /// Marks the resource as being in use by the current submission.
    ///
    /// This will prevent the resource from being destroyed until the current submission is
    /// either complete or cancelled.
    pub fn reference_resource<R: GpuResource>(&mut self, resource: &R) {
        self.stream.reference_resource(resource);
    }

    /// Binds a descriptor set (`vkCmdBindDescriptorSets`).
    ///
    /// # Safety
    ///
    /// The caller is responsible for ensuring that the descriptor set is compatible with the
    /// currently bound pipeline, and that the descriptor set is not destroyed while it is still
    /// in use by the GPU.
    pub unsafe fn bind_descriptor_set(&mut self, index: u32, set: vk::DescriptorSet) {
        self.stream.device.cmd_bind_descriptor_sets(
            self.command_buffer,
            vk::PipelineBindPoint::GRAPHICS,
            self.pipeline_layout,
            index,
            &[set],
            &[],
        )
    }

    /// Specifies descriptors for subsequent draw calls with `vkCmdPushDescriptorSetKHR`.
    pub fn push_descriptors(&mut self, set: u32, bindings: &[(u32, Descriptor)]) {
        assert!(
            self.pipeline_layout != vk::PipelineLayout::null(),
            "encoder must have a pipeline bound before binding arguments"
        );

        unsafe {
            self.stream.do_cmd_push_descriptor_set(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                set,
                bindings,
            );
        }
    }

    /// Binds a graphics pipeline.
    ///
    /// Calling this function invalidates all descriptor & push constant state set by previous calls
    /// to `push_descriptors`, `bind_descriptor_set`, and `push_constants`.
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        // Note about pipeline compatibility:
        //
        // Calling CmdBindPipeline doesn't really invalidate descriptor sets or push constants,
        // but they are only valid for this pipeline if its layout is "compatible" with the layout
        // used previously.
        // There is a notion of "partial compatibility", in which the first N descriptor set bindings
        // stay valid if the pipeline layouts have the same N first descriptor set layouts.
        // However, partial compatibility requires that layouts have the *same push constants ranges*
        // which is far too restrictive for our use cases
        // (bindless, with pass-specific parameters in push constants).
        //
        // So, don't bother with this insanity and rebind everything between pipeline changes.
        // Hopefully vkCmdBindDescriptorSets is cheap enough. I'm pretty sure it doesn't do much
        // if the sets are already bound
        // (for reference, see https://gitlab.freedesktop.org/mesa/mesa/-/blob/main/src/nouveau/vulkan/nvk_cmd_buffer.c?ref_type=heads#L648)

        // SAFETY: TBD
        // TODO: there's no way to ensure that the pipeline lives long enough
        unsafe {
            self.stream.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                pipeline.pipeline,
            );
            if pipeline.bindless {
                self.stream.bind_bindless_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::GRAPHICS,
                    pipeline.pipeline_layout,
                );
            }
            self.pipeline_layout = pipeline.pipeline_layout;
            // TODO strong ref to pipeline
        }
    }

    /// Binds a vertex buffer.
    ///
    /// # Arguments
    /// * `binding` vertex buffer binding index
    /// * `buffer_range` vertex buffer range
    /// * `stride` size in bytes between vertices in the buffer
    pub fn bind_vertex_buffer(&mut self, binding: u32, buffer_range: BufferRangeUntyped) {
        self.reference_resource(&buffer_range.buffer);
        unsafe {
            self.stream.device.cmd_bind_vertex_buffers2(
                self.command_buffer,
                binding,
                &[buffer_range.buffer.handle],
                &[buffer_range.offset as vk::DeviceSize],
                None,
                None,
            );
        }
    }

    /// Binds an index buffer.
    ///
    /// # Arguments
    /// * `index_type` type of indices in the index buffer
    /// * `index_buffer` index buffer range
    pub fn bind_index_buffer(&mut self, index_type: vk::IndexType, index_buffer: BufferRangeUntyped) {
        self.reference_resource(&index_buffer.buffer);
        unsafe {
            self.stream.device.cmd_bind_index_buffer(
                self.command_buffer,
                index_buffer.buffer.handle,
                index_buffer.offset as vk::DeviceSize,
                index_type.into(),
            );
        }
    }

    /// Binds push constants.
    ///
    /// Push constants stay valid until the bound pipeline is changed.
    pub fn push_constants<P>(&mut self, data: &P)
    where
        P: Copy,
    {
        unsafe {
            self.stream.do_cmd_push_constants(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                slice::from_raw_parts(data as *const P as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }
    /// Binds push constants.
    ///
    /// Push constants stay valid until the bound pipeline is changed.

    pub fn push_constants_slice(&mut self, data: &[u8]) {
        unsafe {
            self.stream.do_cmd_push_constants(
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                slice::from_raw_parts(data.as_ptr() as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }

    /// Sets the primitive topology.
    pub fn set_primitive_topology(&mut self, topology: vk::PrimitiveTopology) {
        unsafe {
            self.stream
                .device
                .cmd_set_primitive_topology(self.command_buffer, topology.into());
        }
    }

    /// Sets the viewport.
    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        unsafe {
            self.stream.device.cmd_set_viewport(
                self.command_buffer,
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

    pub fn set_viewport_to_render_area(&mut self) {
        self.set_viewport(
            self.render_area.offset.x as f32,
            self.render_area.offset.y as f32,
            self.render_area.extent.width as f32,
            self.render_area.extent.height as f32,
            0.0,
            1.0,
        );
    }

    /// Sets the scissor rectangle.
    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        unsafe {
            self.stream.device.cmd_set_scissor(
                self.command_buffer,
                0,
                &[vk::Rect2D {
                    offset: vk::Offset2D { x, y },
                    extent: vk::Extent2D { width, height },
                }],
            );
        }
    }

    pub fn set_scissor_to_render_area(&mut self) {
        self.set_scissor(
            self.render_area.offset.x,
            self.render_area.offset.y,
            self.render_area.extent.width,
            self.render_area.extent.height,
        );
    }

    pub fn clear_color(&mut self, attachment: u32, color: ClearColorValue) {
        self.clear_color_rect(
            attachment,
            color,
            Rect2D::from_xywh(0, 0, self.render_area.extent.width, self.render_area.extent.height),
        );
    }

    pub fn clear_depth(&mut self, depth: f32) {
        self.clear_depth_rect(
            depth,
            Rect2D::from_xywh(0, 0, self.render_area.extent.width, self.render_area.extent.height),
        );
    }

    pub fn clear_color_rect(&mut self, attachment: u32, color: ClearColorValue, rect: Rect2D) {
        unsafe {
            self.stream.device.cmd_clear_attachments(
                self.command_buffer,
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
            self.stream.device.cmd_clear_attachments(
                self.command_buffer,
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
            self.stream.device.cmd_draw(
                self.command_buffer,
                vertices.len() as u32,
                instances.len() as u32,
                vertices.start,
                instances.start,
            );
        }
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        unsafe {
            self.stream.device.cmd_draw_indexed(
                self.command_buffer,
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
            self.stream.device.ext_mesh_shader().cmd_draw_mesh_tasks(
                self.command_buffer,
                group_count_x,
                group_count_y,
                group_count_z,
            );
        }
    }

    pub fn finish(self) {
        // Nothing to do. Drop impl does the work (and calls `do_finish`).
    }

    fn do_finish(&mut self) {
        unsafe {
            self.stream.device.cmd_end_rendering(self.command_buffer);
        }
    }
}

impl<'a> Drop for RenderEncoder<'a> {
    fn drop(&mut self) {
        self.do_finish();
    }
}

/// Parameters of `CommandStream::begin_rendering`
pub struct RenderPassInfo<'a> {
    /// The color attachments to use for the render pass.
    pub color_attachments: &'a [ColorAttachment<'a>],
    /// The depth/stencil attachment to use for the render pass.
    pub depth_stencil_attachment: Option<DepthStencilAttachment<'a>>,
}

impl CommandStream {
    /// Starts a rendering pass.
    ///
    /// The render area is set to cover the entire size of the attachments.
    /// The initial viewport and scissor rects are set to cover the entire render area.
    ///
    /// # Arguments
    ///
    /// * `attachments` - The attachments to use for the render pass
    pub fn begin_rendering(&mut self, desc: RenderPassInfo) -> RenderEncoder {
        // determine render area
        let render_area = {
            // FIXME validate that all attachments have the same size
            // FIXME validate that all images are 2D
            let extent;
            if let Some(color) = desc.color_attachments.first() {
                extent = color.image_view.size();
            } else if let Some(ref depth) = desc.depth_stencil_attachment {
                extent = depth.image_view.size();
            } else {
                panic!("render_area must be specified if no attachments are specified");
            }

            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            }
        };

        // Begin render pass
        let color_attachment_infos: Vec<_> = desc
            .color_attachments
            .iter()
            .map(|a| {
                vk::RenderingAttachmentInfo {
                    image_view: a.image_view.handle,
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: if a.clear_value.is_some() {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    },
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue {
                        color: a.get_vk_clear_color_value(),
                    },
                    // TODO multisampling resolve
                    ..Default::default()
                }
            })
            .collect();

        let depth_attachment;
        let stencil_attachment;
        let p_depth_attachment;
        let p_stencil_attachment;
        if let Some(ref depth) = desc.depth_stencil_attachment {
            depth_attachment = vk::RenderingAttachmentInfo {
                image_view: depth.image_view.handle,
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: if depth.depth_clear_value.is_some() {
                    vk::AttachmentLoadOp::CLEAR
                } else {
                    vk::AttachmentLoadOp::LOAD
                },
                store_op: vk::AttachmentStoreOp::STORE,
                clear_value: vk::ClearValue {
                    depth_stencil: depth.get_vk_clear_depth_stencil_value(),
                },
                // TODO multisampling resolve
                ..Default::default()
            };
            p_depth_attachment = &depth_attachment as *const _;

            if is_depth_and_stencil_format(depth.image_view.format) {
                stencil_attachment = vk::RenderingAttachmentInfo {
                    image_view: depth.image_view.handle,
                    // TODO different layouts
                    image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: if depth.stencil_clear_value.is_some() {
                        vk::AttachmentLoadOp::CLEAR
                    } else {
                        vk::AttachmentLoadOp::LOAD
                    },
                    store_op: vk::AttachmentStoreOp::STORE,
                    clear_value: vk::ClearValue {
                        depth_stencil: depth.get_vk_clear_depth_stencil_value(),
                    },
                    // TODO multisampling resolve
                    ..Default::default()
                };
                p_stencil_attachment = &stencil_attachment as *const _;
            } else {
                p_stencil_attachment = ptr::null();
            }
        } else {
            p_depth_attachment = ptr::null();
            p_stencil_attachment = ptr::null();
        };

        // Register resource uses.
        // We could also do that after encoding the pass.
        // It doesn't matter much except we can report usage conflicts earlier.
        let mut barrier = Barrier::new();
        for color in desc.color_attachments.iter() {
            self.reference_resource(color.image_view);
            barrier = barrier.color_attachment_write(&color.image_view.image());
        }
        if let Some(ref depth) = desc.depth_stencil_attachment {
            // TODO we don't know whether the depth attachment will be written to
            self.reference_resource(depth.image_view);
            barrier = barrier.depth_stencil_attachment_write(&depth.image_view.image());
        }

        self.barrier(barrier);

        let rendering_info = vk::RenderingInfo {
            flags: Default::default(),
            render_area,
            layer_count: 1, // TODO?
            view_mask: 0,
            color_attachment_count: color_attachment_infos.len() as u32,
            p_color_attachments: color_attachment_infos.as_ptr(),
            p_depth_attachment,
            p_stencil_attachment,
            ..Default::default()
        };

        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_begin_rendering(command_buffer, &rendering_info);
        }

        let mut encoder = RenderEncoder {
            stream: self,
            command_buffer,
            render_area,
            pipeline_layout: Default::default(),
        };

        encoder.set_viewport(
            0.0,
            0.0,
            render_area.extent.width as f32,
            render_area.extent.height as f32,
            0.0,
            1.0,
        );
        encoder.set_scissor(0, 0, render_area.extent.width, render_area.extent.height);

        encoder
    }
}
