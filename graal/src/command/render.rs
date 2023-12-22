//! Render command encoders
use std::ffi::c_void;
use std::mem::MaybeUninit;
use std::ops::{Deref, Range};
use std::{mem, ptr};
use ash::vk;
use ash::vk::{PipelineBindPoint, PipelineLayout};
use crate::{ArgumentKind, Arguments, Attachments, Buffer, BufferId, BufferRangeAny, ClearColorValue, Device, GraphicsPipeline, Image, ImageDataLayout, ImageSubresourceLayers, ImageView, IndexType, is_depth_and_stencil_format, PrimitiveTopology, Rect2D, Rect3D, RefCounted, ResourceUse, VertexBufferDescriptor};
use crate::command::{CommandBuffer, do_cmd_push_constants, do_cmd_push_descriptor_sets, EncoderBase};


enum RenderCommand {
    BindArguments {
        set: u32,
        /// Offset in argument array
        offset: u32,
        /// Number of arguments
        count: u32,
    },
    BindPipeline {
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
    },
    BindVertexBuffer {
        binding: u32,
        buffer: RefCounted<BufferId>,
        handle: vk::Buffer,
        offset: u64,
        size: u64,
        stride: u32,
    },
    BindIndexBuffer {
        buffer: RefCounted<BufferId>,
        handle: vk::Buffer,
        offset: u64,
        index_type: IndexType,
    },
    BindPushConstants {
        /// Offset in encoder buffer (in bytes)
        offset: u32,
        /// Size in bytes
        size: u32,
    },
    ClearColorRect {
        attachment: u32,
        color: ClearColorValue,
        rect: Rect2D,
    },
    ClearDepthRect {
        depth: f32,
        rect: Rect2D,
    },
    Draw {
        vertices: Range<u32>,
        instances: Range<u32>,
    },
    DrawIndexed {
        indices: Range<u32>,
        base_vertex: i32,
        instances: Range<u32>,
    },
    DrawMeshTasks {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
    SetViewport {
        x: f32,
        y: f32,
        width: f32,
        height: f32,
        min_depth: f32,
        max_depth: f32,
    },
    SetScissor {
        x: i32,
        y: i32,
        width: u32,
        height: u32,
    },
    SetPrimitiveTopology(PrimitiveTopology),
}
////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

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
}

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct RenderEncoder<'a> {
    base: EncoderBase<'a>,
    render_area: vk::Rect2D,
    color_targets: Vec<ColorTarget>,
    depth_stencil_target: Option<DepthStencilTarget>,
    commands: Vec<RenderCommand>,
}

impl<'a> RenderEncoder<'a> {
    /*pub(super) fn new(command_buffer: &'a mut CommandBuffer, width: u32, height: u32) -> Self {
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
    }*/

    pub fn device(&self) -> &Device {
        self.base.command_buffer.device()
    }

    pub fn bind_arguments<A: Arguments>(&mut self, set: u32, arguments: &A) {
        unsafe {
            let (offset,count) = self.base.record_arguments(arguments);
            self.commands.push(RenderCommand::BindArguments {
                set,
                offset,
                count,
            })
        }
    }

    // SAFETY: TBD
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        self.commands.push(RenderCommand::BindPipeline {
            pipeline: pipeline.pipeline,
            pipeline_layout: pipeline.pipeline_layout,
        })
    }

    /// Binds a vertex buffer.
    pub fn bind_vertex_buffer(&mut self, vertex_buffer: &VertexBufferDescriptor) {
        self.base.use_buffer(vertex_buffer.buffer_range.buffer, ResourceUse::VERTEX);
        self.commands.push(RenderCommand::BindVertexBuffer {
            binding: vertex_buffer.binding,
            buffer: vertex_buffer.buffer_range.buffer.id(),
            handle: vertex_buffer.buffer_range.buffer.handle(),
            offset: vertex_buffer.buffer_range.offset,
            size: vertex_buffer.buffer_range.size,
            stride: vertex_buffer.stride,
        });
    }

    // TODO typed version
    pub fn bind_index_buffer(&mut self, index_type: IndexType, index_buffer: BufferRangeAny) {
        self.base.use_buffer(index_buffer.buffer, ResourceUse::INDEX);
        self.commands.push(RenderCommand::BindIndexBuffer {
            buffer: index_buffer.buffer.id(),
            handle: index_buffer.buffer.handle(),
            offset: index_buffer.offset,
            index_type,
        });
    }

    /// Binds push constants.
    pub fn bind_push_constants<P>(&mut self, data: &P)
        where
            P: Copy + ?Sized,
    {
        let size = mem::size_of_val(data);
        let offset = unsafe { self.base.push_constants_raw(size, data as *const _ as *const c_void) };
        self.commands.push(RenderCommand::BindPushConstants {
            offset,
            size: size as u32,
        });
    }

    pub fn set_primitive_topology(&mut self, topology: PrimitiveTopology) {
        self.commands.push(RenderCommand::SetPrimitiveTopology(topology));
    }

    pub fn set_viewport(&mut self, x: f32, y: f32, width: f32, height: f32, min_depth: f32, max_depth: f32) {
        self.commands.push(RenderCommand::SetViewport {
            x,
            y,
            width,
            height,
            min_depth,
            max_depth,
        });
    }

    pub fn set_scissor(&mut self, x: i32, y: i32, width: u32, height: u32) {
        self.commands.push(RenderCommand::SetScissor { x, y, width, height });
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
        self.commands.push(RenderCommand::ClearColorRect {
            attachment,
            color,
            rect,
        });
    }

    pub fn clear_depth_rect(&mut self, depth: f32, rect: Rect2D) {
        self.commands.push(RenderCommand::ClearDepthRect { depth, rect });
    }

    pub fn draw(&mut self, vertices: Range<u32>, instances: Range<u32>) {
        self.commands.push(RenderCommand::Draw { vertices, instances });
    }

    pub fn draw_indexed(&mut self, indices: Range<u32>, base_vertex: i32, instances: Range<u32>) {
        self.commands.push(RenderCommand::DrawIndexed {
            indices,
            base_vertex,
            instances,
        });
    }

    pub fn draw_mesh_tasks(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.commands.push(RenderCommand::DrawMeshTasks {
            group_count_x,
            group_count_y,
            group_count_z,
        });
    }

    unsafe fn record_render_commands(&mut self) {

        self.base.flush_pipeline_barriers();

        let device = &self.base.command_buffer.device;
        let command_buffer = self.base.command_buffer.command_buffer;
        let mut current_pipeline_layout = vk::PipelineLayout::null();

        // Setup VkRenderingAttachmentInfos
        let mut color_attachment_infos: Vec<_> = self
            .color_targets
            .iter()
            .map(|a| {
                vk::RenderingAttachmentInfo {
                    image_view: a.image_view,
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
        if let Some(ref depth) = self.depth_stencil_target {
            depth_attachment = vk::RenderingAttachmentInfo {
                image_view: depth.image_view,
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

            if is_depth_and_stencil_format(depth.format) {
                stencil_attachment = vk::RenderingAttachmentInfo {
                    image_view: depth.image_view,
                    // TODO different layouts
                    image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: depth.load_op,     // FIXME: different load/store ops!
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
            render_area: self.render_area,
            layer_count: 1, // TODO?
            view_mask: 0,
            color_attachment_count: color_attachment_infos.len() as u32,
            p_color_attachments: color_attachment_infos.as_ptr(),
            p_depth_attachment,
            p_stencil_attachment,
            ..Default::default()
        };

        device.cmd_begin_rendering(command_buffer, &rendering_info);

        for command in self.commands.iter() {
            match *command {
                RenderCommand::BindPipeline {
                    pipeline,
                    pipeline_layout,
                } => {
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::GRAPHICS, pipeline);
                    current_pipeline_layout = pipeline_layout;
                }
                RenderCommand::BindArguments { set, offset, count } => {
                    do_cmd_push_descriptor_sets(
                        device,
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        current_pipeline_layout,
                        set,
                        &self.base.descriptor_writes[(offset as usize)..(offset as usize + count as usize)],
                    );
                }
                RenderCommand::BindVertexBuffer {
                    binding,
                    handle,
                    offset,
                    size,
                    stride,
                    ..
                } => {
                    device.cmd_bind_vertex_buffers2(
                        command_buffer,
                        binding,
                        &[handle],
                        &[offset as vk::DeviceSize],
                        None,
                        None,
                    );
                }
                RenderCommand::BindIndexBuffer {
                    handle,
                    offset,
                    index_type,
                    ..
                } => {
                    device.cmd_bind_index_buffer(command_buffer, handle, offset as vk::DeviceSize, index_type.into());
                }
                RenderCommand::BindPushConstants { offset, size } => {
                    do_cmd_push_constants(
                        device,
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        current_pipeline_layout,
                        &self.base.push_constant_data[(offset as usize)..(offset as usize + size as usize)],
                    );
                }
                RenderCommand::ClearColorRect {
                    color,
                    rect,
                    attachment,
                } => unsafe {
                    device.cmd_clear_attachments(
                        command_buffer,
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
                },
                RenderCommand::ClearDepthRect { rect, depth } => {
                    device.cmd_clear_attachments(
                        command_buffer,
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
                RenderCommand::Draw {
                    ref vertices,
                    ref instances,
                } => {
                    device.cmd_draw(
                        command_buffer,
                        vertices.len() as u32,
                        instances.len() as u32,
                        vertices.start,
                        instances.start,
                    );
                }
                RenderCommand::DrawIndexed {
                    ref indices,
                    base_vertex,
                    ref instances,
                } => {
                    device.cmd_draw_indexed(
                        command_buffer,
                        indices.len() as u32,
                        instances.len() as u32,
                        indices.start,
                        base_vertex,
                        instances.start,
                    );
                }
                RenderCommand::DrawMeshTasks {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => {
                    device.ext_mesh_shader().cmd_draw_mesh_tasks(
                        command_buffer,
                        group_count_x,
                        group_count_y,
                        group_count_z,
                    );
                }
                RenderCommand::SetViewport {
                    x,
                    y,
                    width,
                    height,
                    min_depth,
                    max_depth,
                } => {
                    device.cmd_set_viewport(
                        command_buffer,
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
                RenderCommand::SetScissor { x, y, width, height } => {
                    device.cmd_set_scissor(
                        command_buffer,
                        0,
                        &[vk::Rect2D {
                            offset: vk::Offset2D { x, y },
                            extent: vk::Extent2D { width, height },
                        }],
                    );
                }
                RenderCommand::SetPrimitiveTopology(topology) => {
                    device.cmd_set_primitive_topology(command_buffer, topology.into());
                }
            }
        }

        device.cmd_end_rendering(command_buffer);
    }
}

impl<'a> Drop for RenderEncoder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.record_render_commands();
        }
    }
}

impl CommandBuffer {
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

        // TODO: RenderEncoder::new
        let mut encoder = RenderEncoder {
            base: EncoderBase::new(self),
            render_area,
            color_targets: color_attachments
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
            }),
            commands: vec![],
        };

        // Register resource uses.
        // We could also do that after encoding the pass.
        // It doesn't matter much except we can report usage conflicts earlier.
        for color in color_attachments.iter() {
            encoder.base.use_image_view(color.image_view, ResourceUse::COLOR_TARGET);
        }
        if let Some(ref depth) = depth_stencil_attachment {
            // TODO we don't know whether the depth attachment will be written to
            encoder.base.use_image_view(
                depth.image_view,
                ResourceUse::DEPTH_STENCIL_READ | ResourceUse::DEPTH_STENCIL_WRITE,
            );
        }

        encoder
    }
}