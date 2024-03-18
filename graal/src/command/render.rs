//! Render command encoders
use std::{mem, mem::MaybeUninit, ops::Range, ptr, slice, sync::Arc};

use ash::vk;
use fxhash::FxHashMap;

use crate::{
    command::{do_cmd_push_constants, do_cmd_push_descriptor_sets, DescriptorWrite},
    is_depth_and_stencil_format, ArgumentKind, Arguments, Attachments, BufferAccess, BufferId, BufferInner, BufferRangeUntyped,
    ClearColorValue, CommandStream, Device, GraphicsPipeline, ImageAccess, ImageId, ImageInner, ImageView, ImageViewId, ImageViewInner,
    IndexType, PrimitiveTopology, Rect2D, VertexBufferDescriptor,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

/*
struct ColorTarget {
    image_view: vk::ImageView,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    clear_value: Option<vk::ClearValue>,
}

struct DepthStencilTarget {
    image_view: vk::ImageView,
    format: vk::Format,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    // TODO stencil_load_op, stencil_store_op
    clear_value: Option<vk::ClearValue>,
}*/

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    stream: &'a mut CommandStream,
    command_buffer: vk::CommandBuffer,
    render_area: vk::Rect2D,
    pipeline_layout: vk::PipelineLayout,

    pub(crate) used_buffers: FxHashMap<BufferId, (Arc<BufferInner>, BufferAccess)>,
    pub(crate) used_images: FxHashMap<ImageId, (Arc<ImageInner>, ImageAccess)>,
    pub(crate) used_image_views: FxHashMap<ImageViewId, Arc<ImageViewInner>>,
}

impl<'a> RenderEncoder<'a> {
    pub fn device(&self) -> &Device {
        self.stream.device()
    }

    fn use_image(&mut self, image: &Arc<ImageInner>, access: ImageAccess) {
        if let Some(entry) = self.used_images.get_mut(&image.id) {
            if entry.1 != access {
                panic!(
                    "Image {:?} is already used in render pass with conflicting access {:?} (new access: {:?})",
                    image.id, entry.1, access
                );
            }
        } else {
            self.used_images.insert(image.id, (image.clone(), access));
        }
    }

    fn use_buffer(&mut self, buffer: &Arc<BufferInner>, access: BufferAccess) {
        if let Some(entry) = self.used_buffers.get_mut(&buffer.id) {
            if entry.1 != access {
                panic!(
                    "Buffer {:?} is already used in render pass with conflicting access {:?} (new access: {:?})",
                    buffer.id, entry.1, access
                );
            }
        } else {
            self.used_buffers.insert(buffer.id, (buffer.clone(), access));
        }
    }

    fn use_image_view(&mut self, image_view: &ImageView, state: ImageAccess) {
        self.use_image(&image_view.inner.image, state);
        self.used_image_views.insert(image_view.inner.id, image_view.inner.clone());
    }

    pub fn bind_arguments<A: Arguments>(&mut self, set: u32, arguments: &A) {
        assert!(
            self.pipeline_layout != vk::PipelineLayout::null(),
            "encoder must have a pipeline bound before binding arguments"
        );

        let mut descriptor_writes = vec![];
        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image { image_view, access } => {
                    self.use_image_view(image_view, access);
                    descriptor_writes.push(DescriptorWrite::Image {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        image_view: image_view.handle(),
                        format: image_view.format(),
                        access,
                    });
                }
                ArgumentKind::Buffer {
                    buffer,
                    access,
                    offset,
                    size,
                } => {
                    self.use_buffer(&buffer.inner, access);
                    descriptor_writes.push(DescriptorWrite::Buffer {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        buffer: buffer.handle(),
                        access,
                        offset,
                        size,
                    });
                }
                ArgumentKind::Sampler { sampler } => {
                    descriptor_writes.push(DescriptorWrite::Sampler {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        sampler: sampler.handle(),
                    });
                }
            }
        }

        let device = self.stream.device();

        unsafe {
            do_cmd_push_descriptor_sets(
                device,
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                set,
                descriptor_writes.as_slice(),
            );
        }
    }

    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        // SAFETY: TBD
        unsafe {
            self.stream
                .device
                .cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline);
            self.pipeline_layout = pipeline.pipeline_layout;
            // TODO strong ref to pipeline
        }
    }

    /// Binds a vertex buffer.
    pub fn bind_vertex_buffer(&mut self, vertex_buffer: &VertexBufferDescriptor) {
        self.use_buffer(&vertex_buffer.buffer_range.buffer.inner, BufferAccess::VERTEX);
        unsafe {
            self.stream.device.cmd_bind_vertex_buffers2(
                self.command_buffer,
                vertex_buffer.binding,
                &[vertex_buffer.buffer_range.buffer.handle],
                &[vertex_buffer.buffer_range.offset as vk::DeviceSize],
                None, // FIXME vertex buffer strides
                None, // FIXME vertex buffer sizes
            );
        }
    }

    // TODO typed version
    pub fn bind_index_buffer(&mut self, index_type: IndexType, index_buffer: BufferRangeUntyped) {
        self.use_buffer(&index_buffer.buffer.inner, BufferAccess::INDEX);
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
    pub fn bind_push_constants<P>(&mut self, data: &P)
    where
        P: Copy + ?Sized,
    {
        unsafe {
            do_cmd_push_constants(
                self.stream.device(),
                self.command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                slice::from_raw_parts(data as *const P as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }

    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        unsafe {
            self.stream.device.cmd_set_primitive_topology(self.command_buffer, topology.into());
        }
    }

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
            self.stream
                .device
                .ext_mesh_shader()
                .cmd_draw_mesh_tasks(self.command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn finish(self) {
        // Nothing to do. Drop impl does the work (and calls `do_finish`).
    }

    fn do_finish(&mut self) {
        for (_, (buffer, access)) in self.used_buffers.drain() {
            self.stream.use_buffer(&buffer, access);
        }

        for (_, (image, access)) in self.used_images.drain() {
            self.stream.use_image(&image, access);
        }

        for (_, image_view) in self.used_image_views.drain() {
            self.stream.tracked_image_views.insert(image_view.id, image_view);
        }

        self.stream.flush_barriers();
        self.stream.close_command_buffer();

        unsafe {
            self.stream.device.cmd_end_rendering(self.command_buffer);
            self.stream.set_current_command_buffer(self.command_buffer);
        }
    }
}

impl<'a> Drop for RenderEncoder<'a> {
    fn drop(&mut self) {
        self.do_finish();
    }
}

impl CommandStream {
    /// Start a rendering pass
    ///
    /// # Arguments
    ///
    /// * `attachments` - The attachments to use for the render pass
    /// * `render_area` - The area to render to. If `None`, the entire area of the attached images is rendered to.
    pub fn begin_rendering<A: Attachments>(&mut self, attachments: &A) -> RenderEncoder {
        // collect attachments
        let color_attachments: Vec<_> = attachments.color_attachments().collect();
        let depth_stencil_attachment = attachments.depth_stencil_attachment();

        // determine render area
        let render_area = {
            // FIXME validate that all attachments have the same size
            // FIXME validate that all images are 2D
            let extent = color_attachments
                .first()
                .or(depth_stencil_attachment.as_ref())
                .expect("render_area must be specified if no attachments are specified")
                .image_view
                .size();
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            }
        };

        // Begin render pass
        let mut color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                vk::RenderingAttachmentInfo {
                    image_view: a.image_view.handle,
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: a.load_op,
                    store_op: a.store_op,
                    clear_value: a.clear_value.unwrap_or_default(),
                    // TODO multisampling resolve
                    ..Default::default()
                }
            })
            .collect();

        let depth_attachment;
        let stencil_attachment;
        let p_depth_attachment;
        let p_stencil_attachment;
        if let Some(ref depth) = depth_stencil_attachment {
            depth_attachment = vk::RenderingAttachmentInfo {
                image_view: depth.image_view.handle,
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: depth.load_op,
                store_op: depth.store_op,
                clear_value: depth.clear_value.unwrap_or_default(),
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
                    load_op: depth.load_op, // FIXME: different load/store ops!
                    store_op: depth.store_op,
                    clear_value: depth.clear_value.unwrap_or_default(),
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

        let command_buffer = self.create_command_buffer_raw();
        unsafe {
            self.device.cmd_begin_rendering(command_buffer, &rendering_info);
        }

        let mut encoder = RenderEncoder {
            stream: self,
            command_buffer,
            render_area,
            /*color_targets: color_attachments
                .iter()
                .map(|target| ColorTarget {
                    image_view: target.image_view.handle(),
                    load_op: target.load_op,
                    store_op: target.store_op,
                    clear_value: target.clear_value,
                })
                .collect(),
            depth_stencil_target: depth_stencil_attachment.as_ref().map(|attachment| DepthStencilTarget {
                image_view: attachment.image_view.handle(),
                format: attachment.image_view.format,
                load_op: attachment.load_op,
                store_op: attachment.store_op,
                clear_value: attachment.clear_value,
            }),*/
            pipeline_layout: Default::default(),
            used_buffers: Default::default(),
            used_images: Default::default(),
            used_image_views: Default::default(),
        };

        // Register resource uses.
        // We could also do that after encoding the pass.
        // It doesn't matter much except we can report usage conflicts earlier.
        for color in color_attachments.iter() {
            encoder.use_image_view(color.image_view, ImageAccess::COLOR_TARGET);
        }
        if let Some(ref depth) = depth_stencil_attachment {
            // TODO we don't know whether the depth attachment will be written to
            encoder.use_image_view(depth.image_view, ImageAccess::DEPTH_STENCIL_READ | ImageAccess::DEPTH_STENCIL_WRITE);
        }

        encoder
    }
}
