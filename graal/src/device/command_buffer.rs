use crate::{
    aspects_for_format,
    device::{Barrier, DependencyState, ImageOrBuffer, PipelineBarrierBuilder, ResourceHandle, UsageScope},
    is_write_access, vk, ArgumentKind, Arguments, Attachments, Buffer, BufferId, BufferRangeAny, ClearColorValue,
    ColorBlendEquation, ConservativeRasterizationMode, Device, Format, GraphicsPipeline, GroupId, Image,
    ImageCopyBuffer, ImageCopyView, ImageDataLayout, ImageId, ImageSubresourceLayers, ImageView, IndexType,
    PipelineBindPoint, PrimitiveTopology, Queue, Rect2D, Rect3D, RefCounted, ResourceId, ResourceStateOld, ResourceUse,
    SemaphoreSignal, SemaphoreWait, VertexBufferDescriptor, VertexInput,
};
use ash::prelude::VkResult;
use fxhash::FxHashMap;
use slotmap::SecondaryMap;
use std::{
    ffi::c_void,
    mem,
    mem::MaybeUninit,
    ops::{Deref, DerefMut, Range},
    ptr,
};

////////////////////////////////////////////////////////////////////////////////////////////////////

fn map_buffer_use_to_barrier(usage: ResourceUse) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if usage.contains(ResourceUse::MAP_READ) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_READ;
    }
    if usage.contains(ResourceUse::MAP_WRITE) {
        stages |= vk::PipelineStageFlags2::HOST;
        access |= vk::AccessFlags2::HOST_WRITE;
    }
    if usage.contains(ResourceUse::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if usage.contains(ResourceUse::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if usage.contains(ResourceUse::UNIFORM) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::UNIFORM_READ;
    }
    if usage.intersects(ResourceUse::STORAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if usage.intersects(ResourceUse::STORAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }
    if usage.contains(ResourceUse::INDEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::INDEX_READ;
    }
    if usage.contains(ResourceUse::VERTEX) {
        stages |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }
    if usage.contains(ResourceUse::INDIRECT) {
        stages |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        access |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    (stages, access)
}

fn map_texture_usage_to_barrier(usage: ResourceUse) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stages = vk::PipelineStageFlags2::empty();
    let mut access = vk::AccessFlags2::empty();
    let shader_stages = vk::PipelineStageFlags2::VERTEX_SHADER
        | vk::PipelineStageFlags2::FRAGMENT_SHADER
        | vk::PipelineStageFlags2::COMPUTE_SHADER;

    if usage.contains(ResourceUse::COPY_SRC) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_READ;
    }
    if usage.contains(ResourceUse::COPY_DST) {
        stages |= vk::PipelineStageFlags2::TRANSFER;
        access |= vk::AccessFlags2::TRANSFER_WRITE;
    }
    if usage.contains(ResourceUse::SAMPLED_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if usage.contains(ResourceUse::COLOR_TARGET) {
        stages |= vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT;
        access |= vk::AccessFlags2::COLOR_ATTACHMENT_READ | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE;
    }
    if usage.intersects(ResourceUse::DEPTH_STENCIL_READ) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ;
    }
    if usage.intersects(ResourceUse::DEPTH_STENCIL_WRITE) {
        stages |= vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS;
        access |= vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE;
    }
    if usage.contains(ResourceUse::STORAGE_READ) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ;
    }
    if usage.contains(ResourceUse::STORAGE_READ_WRITE) {
        stages |= shader_stages;
        access |= vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE;
    }

    if usage == ResourceUse::UNINITIALIZED || usage == ResourceUse::PRESENT {
        (vk::PipelineStageFlags2::TOP_OF_PIPE, vk::AccessFlags2::empty())
    } else {
        (stages, access)
    }
}

fn map_texture_usage_to_layout(usage: ResourceUse, format: Format) -> vk::ImageLayout {
    // Note: depth textures are always sampled with RODS layout
    let is_color = aspects_for_format(format).contains(vk::ImageAspectFlags::COLOR);
    match usage {
        ResourceUse::UNINITIALIZED => vk::ImageLayout::UNDEFINED,
        ResourceUse::COPY_SRC => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        ResourceUse::COPY_DST => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        ResourceUse::SAMPLED_READ if is_color => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        ResourceUse::COLOR_TARGET => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
        ResourceUse::DEPTH_STENCIL_WRITE => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
        _ => {
            if usage == ResourceUse::PRESENT {
                vk::ImageLayout::PRESENT_SRC_KHR
            } else if is_color {
                vk::ImageLayout::GENERAL
            } else {
                vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////

pub(super) unsafe fn emit_barriers(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    barriers: &[Barrier],
    //current_scope: &mut UsageScope,
    //new_scope: &UsageScope,
) {
    //let barriers = current_scope.merge(new_scope);

    let mut buffer_barriers = Vec::new();
    let mut image_barriers = Vec::new();

    let mut src_stages = vk::PipelineStageFlags2::empty();
    let mut dst_stages = vk::PipelineStageFlags2::empty();

    for &Barrier { resource, src, dst } in barriers {
        match resource {
            ImageOrBuffer::Image { image, format } => {
                let (ssrc, src_access_mask) = map_texture_usage_to_barrier(src);
                src_stages |= ssrc;
                let (sdst, dst_access_mask) = map_texture_usage_to_barrier(dst);
                dst_stages |= sdst;
                image_barriers.push(vk::ImageMemoryBarrier2 {
                    src_access_mask,
                    dst_access_mask,
                    old_layout: map_texture_usage_to_layout(src, format),
                    new_layout: map_texture_usage_to_layout(dst, format),
                    src_queue_family_index: 0, // TODO?
                    dst_queue_family_index: 0,
                    image: Default::default(),
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: aspects_for_format(format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    },
                    ..Default::default()
                })
            }
            ImageOrBuffer::Buffer(handle) => {
                let (ssrc, src_access_mask) = map_buffer_use_to_barrier(src);
                src_stages |= ssrc;
                let (sdst, dst_access_mask) = map_buffer_use_to_barrier(dst);
                dst_stages |= sdst;
                buffer_barriers.push(vk::BufferMemoryBarrier2 {
                    src_access_mask,
                    dst_access_mask,
                    src_queue_family_index: 0, // TODO?
                    dst_queue_family_index: 0,
                    buffer: handle,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                })
            }
        }
    }

    device.cmd_pipeline_barrier2(
        command_buffer,
        &vk::DependencyInfo {
            dependency_flags: Default::default(),
            memory_barrier_count: 0,
            p_memory_barriers: ptr::null(),
            buffer_memory_barrier_count: buffer_barriers.len() as u32,
            p_buffer_memory_barriers: buffer_barriers.as_ptr(),
            image_memory_barrier_count: image_barriers.len() as u32,
            p_image_memory_barriers: image_barriers.as_ptr(),
            ..Default::default()
        },
    );
}

/// Command buffers
pub struct CommandBuffer {
    device: Device,
    pub(super) refs: Vec<RefCounted<ResourceId>>,
    pub(super) command_buffer: vk::CommandBuffer,
    pub(super) usage_scope: UsageScope,
}

impl CommandBuffer {
    pub(super) fn new(device: &Device, command_buffer: vk::CommandBuffer) -> CommandBuffer {
        Self {
            device: device.clone(),
            refs: vec![],
            command_buffer,
            usage_scope: UsageScope::new(),
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /*pub(super) fn flush_barriers(&mut self) {
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
     */

    // FIXME WTF does this mean
    fn merge_usage_scope(&mut self, usage_scope: &UsageScope) {
        let barriers = self.usage_scope.merge(usage_scope);
        if !barriers.is_empty() {
            unsafe {
                emit_barriers(&self.device, self.command_buffer, &barriers);
            }
        }
    }

    /// Encode a blit operation
    pub fn begin_blit(&mut self) -> BlitCommandEncoder {
        BlitCommandEncoder::new(self)
    }

    /// Start a rendering pass
    ///
    /// # Arguments
    ///
    /// * `attachments` - The attachments to use for the render pass
    /// * `render_area` - The area to render to. If `None`, the entire area of the attached images is rendered to.
    pub fn begin_rendering<A: Attachments>(&mut self, attachments: &A) -> RenderEncoder {
        // collect attachments
        let color_attachments: Vec<_> = attachments.color_attachments().collect();
        let depth_attachment = attachments.depth_attachment();
        let stencil_attachment = attachments.stencil_attachment();

        // determine render area
        let render_area = {
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

        // TODO: RenderEncoder::new
        let mut encoder = RenderEncoder {
            base: EncoderBase {
                command_buffer: self,
                push_constant_data: vec![],
                arguments: vec![],
                usage_scope: UsageScope::new(),
            },
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
            depth_stencil_target: depth_attachment.as_ref().map(|attachment| DepthStencilTarget {
                image_view: attachment.image_view.handle(),
                load_op: attachment.load_op,
                store_op: attachment.store_op,
                clear_value: attachment.clear_value,
            }),
            commands: vec![],
        };

        // register resource uses
        for color in color_attachments.iter() {
            encoder.base.use_image_view(color.image_view, ResourceUse::COLOR_TARGET);
        }
        if let Some(ref depth) = depth_attachment {
            // TODO we don't know whether the depth attachment will be written to
            encoder.base.use_image_view(
                depth.image_view,
                ResourceUse::DEPTH_STENCIL_READ | ResourceUse::DEPTH_STENCIL_WRITE,
            );
        }

        /*if let Some(ref stencil) = stencil_attachment {
            // Should be the same image as depth, but resource tracking should be OK with that
            encoder.base.use_resource(
                stencil.image_view.parent_image.value.into(),
                ResourceUse::DEPTH_STENCIL_READ | ResourceUse::DEPTH_STENCIL_WRITE,
            );
        }*/

        encoder
    }
}

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

/// Render pass commands.
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

enum BlitCommand {
    CopyBufferToBuffer {
        src: vk::Buffer,
        dst: vk::Buffer,
    },
    CopyBufferToTexture {
        src: vk::Buffer,
        src_data_layout: ImageDataLayout,
        dst: vk::Image,
        dst_mip_level: u32,
        dst_origin: vk::Offset3D,
        dst_aspect: vk::ImageAspectFlags,
        size: vk::Extent3D,
    },
    CopyTextureToBuffer {
        src: vk::Image,
        src_layout: vk::ImageLayout,
        dst: vk::Buffer,
        regions: Vec<vk::BufferImageCopy>,
    },
    CopyTextureToTexture {
        src: vk::Image,
        src_mip: u32,
        dst: vk::Image,
        dst_mip: u32,
        aspect: vk::ImageAspectFlags,
        src_offset: vk::Offset3D,
        dst_offset: vk::Offset3D,
        size: vk::Extent3D,
    },
    BlitImage {
        src: vk::Image,
        src_subresource: ImageSubresourceLayers,
        src_region: Rect3D,
        dst: vk::Image,
        dst_subresource: ImageSubresourceLayers,
        dst_region: Rect3D,
        filter: vk::Filter,
    },
    FillBuffer {
        dst: vk::Buffer,
        dst_offset: u64,
        size: u64,
        data: u32,
    },
    ClearImage {
        handle: vk::Image,
        color: ClearColorValue,
        depth: f32,
        stencil: u32,
    },
}

////////////////////////////////////////////////////////////////////////////////////////////////////

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

unsafe fn record_push_descriptors(
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

fn record_push_constants(
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
    arguments: Vec<DescriptorWrite>,
    usage_scope: UsageScope,
}

impl<'a> EncoderBase<'a> {
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
        let offset = self.arguments.len() as u32;
        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image { image_view, use_ } => {
                    self.use_image_view(image_view, use_);
                    self.arguments.push(DescriptorWrite::Image {
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
                    self.arguments.push(DescriptorWrite::Buffer {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        buffer: buffer.handle(),
                        use_,
                        offset,
                        size,
                    });
                }
                ArgumentKind::Sampler { sampler } => {
                    self.arguments.push(DescriptorWrite::Sampler {
                        binding: arg.binding,
                        descriptor_type: arg.descriptor_type,
                        sampler: sampler.handle(),
                    });
                }
            }
        }
        let count = self.arguments.len() as u32 - offset;
        (offset, count)
    }

    pub fn use_image(&mut self, image: &Image, use_: ResourceUse) {
        self.usage_scope
            .insert_or_merge(
                image.id.value.into(),
                ImageOrBuffer::Image {
                    image: image.handle,
                    format: image.format,
                },
                use_,
            )
            .expect("usage conflict");
    }

    pub fn use_image_view(&mut self, image_view: &ImageView, use_: ResourceUse) {
        self.usage_scope
            .insert_or_merge(
                image_view.parent_image.value.into(),
                ImageOrBuffer::Image {
                    image: image_view.image_handle,
                    format: image_view.format(),
                },
                use_,
            )
            .expect("usage conflict");
    }

    pub fn use_buffer(&mut self, buffer: &Buffer, use_: ResourceUse) {
        self.usage_scope
            .insert_or_merge(buffer.id().value.into(), ImageOrBuffer::Buffer(buffer.handle()), use_)
            .expect("usage conflict");
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct ColorTarget {
    image_view: vk::ImageView,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
    clear_value: Option<vk::ClearValue>,
}

struct DepthStencilTarget {
    image_view: vk::ImageView,
    load_op: vk::AttachmentLoadOp,
    store_op: vk::AttachmentStoreOp,
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

    // SAFETY: TBD
    pub fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        self.commands.push(RenderCommand::BindPipeline {
            pipeline: pipeline.pipeline,
            pipeline_layout: pipeline.pipeline_layout,
        })
    }

    /// Binds a vertex buffer.
    pub fn bind_vertex_buffer(&mut self, vertex_buffer: &VertexBufferDescriptor) {
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
        self.base.command_buffer.merge_usage_scope(&self.base.usage_scope);

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
        let depth_attachment_info = if let Some(ref depth) = self.depth_stencil_target {
            Some(vk::RenderingAttachmentInfo {
                image_view: depth.image_view,
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
        let stencil_attachment_info = if let Some(ref stencil) = self.depth_stencil_target {
            Some(vk::RenderingAttachmentInfo {
                image_view: stencil.image_view,
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
            render_area: self.render_area,
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
                    record_push_descriptors(
                        device,
                        command_buffer,
                        vk::PipelineBindPoint::GRAPHICS,
                        current_pipeline_layout,
                        set,
                        &self.base.arguments[(offset as usize)..(offset as usize + count as usize)],
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
                    record_push_constants(
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

////////////////////////////////////////////////////////////////////////////////////////////////////

pub struct BlitCommandEncoder<'a> {
    base: EncoderBase<'a>,
    commands: Vec<BlitCommand>,
}

impl<'a> BlitCommandEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer) -> Self {
        Self {
            base: EncoderBase {
                command_buffer,
                push_constant_data: vec![],
                arguments: vec![],
                usage_scope: UsageScope::new(),
            },
            commands: vec![],
        }
    }

    pub fn fill_buffer(&mut self, buffer: &BufferRangeAny, data: u32) {
        self.base.use_buffer(buffer.buffer, ResourceUse::COPY_DST);
        self.commands.push(BlitCommand::FillBuffer {
            dst: buffer.buffer.handle(),
            dst_offset: buffer.offset,
            size: buffer.size,
            data,
        });
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        self.base.use_image(image, ResourceUse::COPY_DST);
        self.commands.push(BlitCommand::ClearImage {
            handle: image.handle,
            color: clear_color_value,
            depth: 0.0,
            stencil: 0,
        });
    }

    pub unsafe fn copy_image_to_image(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.base.use_image(source.image, ResourceUse::COPY_SRC);
        self.base.use_image(destination.image, ResourceUse::COPY_DST);
        self.commands.push(BlitCommand::CopyTextureToTexture {
            src: source.image.handle,
            src_mip: source.mip_level,
            dst: destination.image.handle,
            dst_mip: destination.mip_level,
            aspect: source.aspect,
            src_offset: source.origin,
            dst_offset: destination.origin,
            size: copy_size,
        });
    }

    pub unsafe fn copy_buffer_to_image(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.base.use_buffer(source.buffer, ResourceUse::COPY_SRC);
        self.base.use_image(destination.image, ResourceUse::COPY_DST);
        self.commands.push(BlitCommand::CopyBufferToTexture {
            src: source.buffer.handle(),
            src_data_layout: source.layout,
            dst: destination.image.handle,
            dst_mip_level: destination.mip_level,
            dst_origin: destination.origin,
            dst_aspect: destination.aspect,
            size: copy_size,
        });
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
        self.base.use_image(src, ResourceUse::COPY_SRC);
        self.base.use_image(dst, ResourceUse::COPY_DST);
        self.commands.push(BlitCommand::BlitImage {
            src: src.handle,
            src_subresource,
            src_region,
            dst: dst.handle,
            dst_subresource,
            dst_region,
            filter,
        });
    }

    unsafe fn record_blit_commands(&mut self) {
        self.base.command_buffer.merge_usage_scope(&self.base.usage_scope);

        let device = &self.base.command_buffer.device;
        let command_buffer = self.base.command_buffer.command_buffer;

        for command in self.commands.iter() {
            match *command {
                BlitCommand::CopyBufferToBuffer { src, dst } => {
                    //device.cmd_copy_buffer(command_buffer, src, dst, regions);
                }
                BlitCommand::CopyBufferToTexture {
                    src,
                    src_data_layout,
                    dst,
                    dst_mip_level,
                    dst_origin,
                    dst_aspect,
                    size,
                } => {
                    let regions = [vk::BufferImageCopy {
                        buffer_offset: src_data_layout.offset,
                        buffer_row_length: src_data_layout.row_length.unwrap_or(0),
                        buffer_image_height: src_data_layout.image_height.unwrap_or(0),
                        image_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: dst_aspect,
                            mip_level: dst_mip_level,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        image_offset: dst_origin,
                        image_extent: size,
                    }];

                    unsafe {
                        device.cmd_copy_buffer_to_image(
                            command_buffer,
                            src,
                            dst,
                            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                            &regions,
                        );
                    }
                }
                BlitCommand::CopyTextureToBuffer {
                    src,
                    src_layout,
                    dst,
                    ref regions,
                } => {
                    device.cmd_copy_image_to_buffer(command_buffer, src, src_layout, dst, regions);
                }
                BlitCommand::CopyTextureToTexture {
                    src,
                    src_mip,
                    dst,
                    dst_mip,
                    aspect,
                    src_offset,
                    dst_offset,
                    size,
                } => {
                    let regions = [vk::ImageCopy {
                        src_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: src_mip,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        src_offset,
                        dst_subresource: vk::ImageSubresourceLayers {
                            aspect_mask: aspect,
                            mip_level: dst_mip,
                            base_array_layer: 0,
                            layer_count: 1,
                        },
                        dst_offset,
                        extent: size,
                    }];

                    device.cmd_copy_image(
                        command_buffer,
                        src,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        dst,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &regions,
                    );
                }
                BlitCommand::FillBuffer {
                    dst,
                    dst_offset,
                    size,
                    data,
                } => {
                    device.cmd_fill_buffer(command_buffer, dst, dst_offset, size, data);
                }
                BlitCommand::ClearImage {
                    handle,
                    color,
                    depth,
                    stencil,
                } => {
                    device.cmd_clear_color_image(
                        command_buffer,
                        handle,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &color.into(),
                        &[vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        }],
                    );
                }
                BlitCommand::BlitImage {
                    src,
                    src_subresource,
                    src_region,
                    dst,
                    dst_subresource,
                    dst_region,
                    filter,
                } => {
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

                    device.cmd_blit_image(
                        command_buffer,
                        src,
                        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                        dst,
                        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                        &blits,
                        filter,
                    );
                }
            }
        }
    }
}

impl Drop for BlitCommandEncoder<'_> {
    fn drop(&mut self) {
        unsafe {
            self.record_blit_commands();
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
