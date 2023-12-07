//! Wrapper around device and queues.
use crate::{
    attachments::Attachments,
    sampler::{Sampler, SamplerCreateInfo},
    vk,
};
use graal::queue::{ResourceState, Submission};
use std::{cell::RefCell, collections::HashMap, ptr};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to create device")]
    DeviceCreationFailed(#[from] graal::device::DeviceCreateError),
}

pub struct Device {
    device: graal::device::Device,
    queue: graal::queue::Queue,
    sampler_cache: RefCell<HashMap<SamplerCreateInfo, Sampler>>,
}

pub struct RenderPass<'a> {
    device: &'a mut Device,
    cb: vk::CommandBuffer,
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    pub fn new() -> Result<Device, Error> {
        // SAFETY: no surface is passed, so there are no requirements on the call.
        let (device, queue) = unsafe { graal::device::create_device_and_queue(None)? };

        Ok(Device {
            device,
            queue,
            sampler_cache: RefCell::new(HashMap::new()),
        })
    }

    pub fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.sampler_cache.borrow().get(info) {
            return sampler.clone();
        }

        todo!()
        /*let sampler = Sampler::new(self.device.clone(), info);
        self.sampler_cache.borrow_mut().insert(info.clone(), sampler.clone());
        sampler*/
    }

    /// Returns the underlying graal device.
    pub fn device(&self) -> &graal::device::Device {
        &self.device
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
        pass_fn: impl FnOnce(&mut RenderPass),
    ) {
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
                .image
                .extent();
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            }
        };

        let mut submission = Submission::new();

        // register resource uses
        // TODO layout should be configurable
        for color in color_attachments.iter() {
            submission.use_image(color.image.handle().id, ResourceState::COLOR_ATTACHMENT_OUTPUT);
        }
        if let Some(ref depth) = depth_attachment {
            submission.use_image(depth.image.handle().id, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }
        if let Some(ref stencil) = stencil_attachment {
            // Could be the same image as depth, but resource tracking should be OK with that
            submission.use_image(stencil.image.handle().id, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }

        // Setup VkRenderingAttachmentInfos
        let mut color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                // NOTE: the image view could be created at the same time as the Attachment struct
                // but actually no, because if both a depth and stencil attachment are specified, they
                // must be the same image view.

                let image_view = a.image.create_view(&a.view_info);
                vk::RenderingAttachmentInfo {
                    image_view,
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
            let image_view = depth.image.create_view(&depth.view_info);
            Some(vk::RenderingAttachmentInfo {
                image_view,
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
            let image_view = stencil.image.create_view(&stencil.view_info);
            Some(vk::RenderingAttachmentInfo {
                image_view,
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

        let cb = self.queue.allocate_command_buffer();
        unsafe {
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .expect("begin_command_buffer failed");
            self.device.cmd_begin_rendering(cb, &rendering_info);
            let mut render_pass = RenderPass { device: self, cb };
            pass_fn(&mut render_pass);
            self.device.cmd_end_rendering(cb);
            self.device.end_command_buffer(cb).expect("end_command_buffer failed");
        }
    }
}

// trait GraphicsPipeline for the object that owns the pipeline
// GraphicsPipeline::Interface for the object used to set the parameters once it's bound
// It's an associated type because there might be different methods depending on the pipeline type.
// E.g. some pipelines may not have vertex inputs, or no attachments

/*

 Philosophy: command buffer oriented.
 * Set arguments procedurally.
 * Stateful: draw commands take the last arguments
 * Set arguments once, then draw multiple times
 * More closely matches the actual commands emitted in the command buffer

 Declarative alternative:
 * Specify arguments all at once in a big struct
 * Issue: if the arguments haven't changed, it will potentially upload a lot of redundant data
 * It's "cleaner" in the sense that it's harder to misuse and more things are checked at compile time
   (e.g. it's impossible to forget to set an argument, this is checked statically).
 * It hides the underlying command buffer / binding model
 * However, it needs some kind of caching to avoid changing states when not necessary => complex, more impl work, possibly inefficient

 => Don't hide the underlying GPU binding model.
    We could abstract it away if we were building a compiler, because then we'd be able to analyze the code
    and determine how to bind things optimally. But we're not building a compiler (at least not yet...).

 device.render(attachments, |render_pass| {
    render_pass.bind_pipeline(&pipeline, |pctx| {
        pctx.set_scene_data(...);

        for object in scene.object() {
            pctx.set_vertex_buffers(...);
            pctx.set_object_data(...);
            pctx.draw(...);
        }

        pipeline.set_arguments_0(???);
        pipeline.set_arguments_1(???);
    });
 });


 */

impl PipelineParamInterface {

    // Which type do I use here? What's the name of the method?
    fn set_arguments_0(&self, args: ???) {

    }

}

impl<'a> RenderPass<'a> {
    pub fn bind_pipeline<G: GraphicsPipeline>(
        &mut self,
        pipeline: &G,
        pipeline_ctx: FnOnce(&mut G::PipelineInterface),
    ) {
        pipeline.create_interface();
    }
}
