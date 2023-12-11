use std::ptr;

use ash::prelude::VkResult;
use fxhash::FxHashMap;

use crate::{
    aspects_for_format,
    device::{
        encoder::{BlitCommandEncoder, RenderEncoder},
        ensure_memory_dependency, DependencyState, PipelineBarrierBuilder, RefCounted, ResourceHandle, ResourceUse,
    },
    vk, Attachments, BufferAny, CommandBuffer, Device, GroupId, ImageAny, ImageView, ResourceId, ResourceState,
};

impl CommandBuffer {
    pub(super) fn new(device: &Device, command_buffer: vk::CommandBuffer) -> CommandBuffer {
        Self {
            device: device.clone(),
            refs: vec![],
            command_buffer,
            initial_uses: FxHashMap::default(),
            final_states: FxHashMap::default(),
            group_uses: Vec::new(),
            barrier_builder: PipelineBarrierBuilder::default(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Makes sure that the given image is in the given layout, and emits the necessary barriers
    /// to transition it if needed.
    ///
    /// # Arguments
    ///
    /// * stage_mask: the pipeline stage at which the image will be used
    /// * access_mask: the access mask for the image (defines both the visiblity requirements and flushes)
    fn use_resource(&mut self, resource: RefCounted<ResourceId>, handle: ResourceHandle, use_: ResourceUse) {
        if !self.initial_uses.contains_key(&resource.value) {
            self.initial_uses.insert(resource.value, use_);
            self.final_states.insert(
                resource.value,
                DependencyState {
                    stages: use_.state.stages,
                    flush_mask: use_.state.access,
                    visible: vk::AccessFlags2::empty(),
                    layout: use_.state.layout,
                },
            );
        } else {
            let dep_state = self.final_states.get_mut(&resource.value).unwrap();
            ensure_memory_dependency(&mut self.barrier_builder, handle, dep_state, &use_);
        }
        self.refs.push(resource);
    }

    pub fn use_buffer(&mut self, buffer: &BufferAny, state: ResourceState) {
        self.use_resource(
            buffer.id().map(Into::into),
            buffer.handle().into(),
            ResourceUse {
                aspect: Default::default(),
                state,
            },
        );
    }

    pub fn use_image_view(&mut self, image_view: &ImageView, state: ResourceState) {
        self.use_resource(
            image_view.parent_id().map(Into::into),
            image_view.image_handle().into(),
            ResourceUse {
                aspect: aspects_for_format(image_view.original_format()),
                state,
            },
        );
    }

    pub fn use_image(&mut self, image: &ImageAny, state: ResourceState) {
        self.use_resource(
            image.id().map(Into::into),
            image.handle().into(),
            ResourceUse {
                aspect: aspects_for_format(image.format()),
                state,
            },
        );
    }

    pub fn use_group(&mut self, group: GroupId) {
        self.group_uses.push(group);
    }

    pub(super) fn flush_barriers(&mut self) {
        if !self.barrier_builder.is_empty() {
            let device = self.device.raw();
            unsafe {
                device.cmd_pipeline_barrier2(
                    self.command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: Default::default(),
                        memory_barrier_count: 0,
                        p_memory_barriers: ptr::null(),
                        buffer_memory_barrier_count: self.barrier_builder.buffer_barriers.len() as u32,
                        p_buffer_memory_barriers: self.barrier_builder.buffer_barriers.as_ptr(),
                        image_memory_barrier_count: self.barrier_builder.image_barriers.len() as u32,
                        p_image_memory_barriers: self.barrier_builder.image_barriers.as_ptr(),
                        ..Default::default()
                    },
                );
            }
            self.barrier_builder.clear();
        }
    }

    /// Encode a blit operation
    pub fn encode_blit(&mut self, encode_fn: impl FnOnce(&mut BlitCommandEncoder)) {
        let mut encoder = BlitCommandEncoder::new(self);
        encode_fn(&mut encoder);
    }

    /// Start a rendering pass
    ///
    /// # Arguments
    ///
    /// * `attachments` - The attachments to use for the render pass
    /// * `render_area` - The area to render to. If `None`, the entire area of the attached images is rendered to.
    pub fn render<A: Attachments>(
        &mut self,
        attachments: &A,
        render_area: Option<vk::Rect2D>,
        pass_fn: impl FnOnce(&mut RenderEncoder),
    ) -> VkResult<()> {
        // collect attachments
        let mut color_attachments: Vec<_> = attachments.color_attachments().collect();
        let mut depth_attachment = attachments.depth_attachment();
        let mut stencil_attachment = attachments.stencil_attachment();

        // determine render area
        let render_area = if let Some(render_area) = render_area {
            render_area
        } else {
            // FIXME validate that all attachments have the same size
            // FIXME validate that all images are 2D
            let extent = color_attachments
                .first()
                .or(depth_attachment.as_ref())
                .or(stencil_attachment.as_ref())
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

        // register resource uses
        // TODO layout should be configurable
        for color in color_attachments.iter() {
            self.use_image_view(color.image_view, ResourceState::COLOR_ATTACHMENT_OUTPUT);
        }
        if let Some(ref depth) = depth_attachment {
            self.use_image_view(depth.image_view, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }
        if let Some(ref stencil) = stencil_attachment {
            // Could be the same image as depth, but resource tracking should be OK with that
            self.use_image_view(stencil.image_view, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }

        // Setup VkRenderingAttachmentInfos
        let mut color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                vk::RenderingAttachmentInfo {
                    image_view: a.image_view.handle(),
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
        let depth_attachment_info = if let Some(ref depth) = depth_attachment {
            Some(vk::RenderingAttachmentInfo {
                image_view: depth.image_view.handle(),
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: depth.load_op,
                store_op: depth.store_op,
                clear_value: depth.clear_value.unwrap_or_default(),
                // TODO multisampling resolve
                ..Default::default()
            })
        } else {
            None
        };
        let stencil_attachment_info = if let Some(ref stencil) = stencil_attachment {
            Some(vk::RenderingAttachmentInfo {
                image_view: stencil.image_view.handle(),
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: stencil.load_op,
                store_op: stencil.store_op,
                clear_value: stencil.clear_value.unwrap_or_default(),
                // TODO multisampling resolve
                ..Default::default()
            })
        } else {
            None
        };

        let rendering_info = vk::RenderingInfo {
            flags: Default::default(),
            render_area,
            layer_count: 1, // TODO?
            view_mask: 0,
            color_attachment_count: color_attachment_infos.len() as u32,
            p_color_attachments: color_attachment_infos.as_ptr(),
            p_depth_attachment: depth_attachment_info
                .as_ref()
                .map(|a| a as *const _)
                .unwrap_or(ptr::null()),
            p_stencil_attachment: stencil_attachment_info
                .as_ref()
                .map(|a| a as *const _)
                .unwrap_or(ptr::null()),
            ..Default::default()
        };

        unsafe {
            self.device
                .raw()
                .cmd_begin_rendering(self.command_buffer, &rendering_info);
            let mut render_encoder = RenderEncoder::new(self);
            pass_fn(&mut render_encoder);
            self.device.raw().cmd_end_rendering(self.command_buffer);
        }
        Ok(())
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
