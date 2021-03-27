use crate::{
    context::{
        descriptor::DescriptorSet,
        format_aspect_mask, is_read_access, is_write_access,
        pass::{Pass, PassKind, ResourceAccess},
        resource::{
            BufferId, BufferInfo, ImageId, ResourceAccessDetails, ResourceId, ResourceKind,
            ResourceMemory, TypedBufferInfo,
        },
        FrameSerialNumber, CommandContext, FrameInFlight, SubmissionNumber, SwapchainImage,
    },
    descriptor::BufferDescriptor,
    device::QueuesInfo,
    vk, BufferData, BufferResourceCreateInfo, Context, DescriptorSetInterface, Device, ImageInfo,
    ImageResourceCreateInfo, ResourceMemoryInfo, MAX_QUEUES,
};
use ash::version::DeviceV1_0;
use bitflags::bitflags;
use slotmap::Key;
use std::{
    cell::{Ref, RefCell, RefMut},
    marker::PhantomData,
    mem,
    mem::MaybeUninit,
    ptr, slice,
};

type TemporarySet = std::collections::BTreeSet<ResourceId>;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AccessType {
    /// Vertex attribute read in the vertex input stage
    VertexAttributeRead,
    /// Index read in the vertex input stage
    IndexRead,

    /// Uniform buffer read in the vertex shader stage.
    VertexShaderReadUniformBuffer,
    /// Uniform buffer read in the fragment shader stage.
    FragmentShaderReadUniformBuffer,
    /// Uniform buffer read in the geometry shader stage.
    GeometryShaderReadUniformBuffer,
    /// Uniform buffer read in the tessellation control shader stage.
    TessControlShaderReadUniformBuffer,
    /// Uniform buffer read in the tessellation evaluation shader stage.
    TessEvalShaderReadUniformBuffer,
    /// Uniform buffer read in the compute shader stage.
    ComputeShaderReadUniformBuffer,
    /// Uniform buffer read in any shader stage.
    AnyShaderReadUniformBuffer,

    /// Sampled image read in the vertex shader stage.
    VertexShaderReadSampledImage,
    /// Sampled image read in the fragment shader stage.
    FragmentShaderReadSampledImage,
    /// Sampled image read in the geometry shader stage.
    GeometryShaderReadSampledImage,
    /// Sampled image read in the tessellation control shader stage.
    TessControlShaderReadSampledImage,
    /// Sampled image read in the tessellation evaluation shader stage.
    TessEvalShaderReadSampledImage,
    /// Sampled image read in the compute shader stage.
    ComputeShaderReadSampledImage,
    /// Sampled image read in any shader stage.
    AnyShaderReadSampledImage,

    /// Any read other than uniform buffers & sampled images in the vertex shader stage.
    VertexShaderReadOther,
    /// Any read other than uniform buffers & sampled images in the fragment shader stage.
    FragmentShaderReadOther,
    /// Any read other than uniform buffers & sampled images in the geometry shader stage.
    GeometryShaderReadOther,
    /// Any read other than uniform buffers & sampled images in the tessellation control shader stage.
    TessControlShaderReadOther,
    /// Any read other than uniform buffers & sampled images in the tessellation evaluation shader stage.
    TessEvalShaderReadOther,
    /// Any read other than uniform buffers & sampled images in the compute shader stage.
    ComputeShaderReadOther,
    /// Any read other than uniform buffers & sampled images in any shader stage.
    AnyShaderReadOther,

    /// Any write in the vertex shader stage.
    VertexShaderWrite,
    /// Any write in the fragment shader stage.
    FragmentShaderWrite,
    /// Any write in the geometry shader stage.
    GeometryShaderWrite,
    /// Any write in the tessellation control shader stage.
    TessControlShaderWrite,
    /// Any write in the tessellation evaluation shader stage.
    TessEvalShaderWrite,
    /// Any write in the compute shader stage.
    ComputeShaderWrite,
    /// Any write in any shader stage.
    AnyShaderWrite,

    /// Read transfer source
    TransferRead,
    /// Written transfer destination
    TransferWrite,

    /// Written color attachment
    ColorAttachmentWrite,
    /// Read and written color attachment.
    ColorAttachmentReadWrite,
    /// Read-only depth attachment
    DepthStencilAttachmentRead,
    /// Written depth attachment
    DepthStencilAttachmentWrite,
    /// Read/written depth attachment
    DepthStencilAttachmentReadWrite,
}

// FIXME it would be more precise to specify each shader stage instead of ALL_COMMANDS, but
// then we need to check that the device actually supports the stages in question.
// VK_KHR_synchronization2 has VK_PIPELINE_STAGE_2_PRE_RASTERIZATION_SHADERS_BIT_KHR that can be used to this effect.
const ANY_SHADER_STAGE: vk::PipelineStageFlags = vk::PipelineStageFlags::ALL_COMMANDS;

impl AccessType {
    pub fn to_access_info(&self) -> AccessTypeInfo {
        match *self {
            AccessType::VertexAttributeRead => AccessTypeInfo {
                access_mask: vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
                input_stage: vk::PipelineStageFlags::VERTEX_INPUT,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::IndexRead => AccessTypeInfo {
                access_mask: vk::AccessFlags::INDEX_READ,
                input_stage: vk::PipelineStageFlags::VERTEX_INPUT,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::VertexShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::VERTEX_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::FragmentShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::GeometryShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::GEOMETRY_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessControlShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::TessEvalShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::ComputeShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },
            AccessType::AnyShaderReadUniformBuffer => AccessTypeInfo {
                access_mask: vk::AccessFlags::UNIFORM_READ,
                input_stage: ANY_SHADER_STAGE,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::UNDEFINED,
            },

            AccessType::VertexShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::VERTEX_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::FragmentShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::GeometryShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::GEOMETRY_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::TessControlShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::TessEvalShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::ComputeShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },
            AccessType::AnyShaderReadSampledImage => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: ANY_SHADER_STAGE,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            },

            AccessType::VertexShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::VERTEX_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::FragmentShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::GEOMETRY_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessControlShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessEvalShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ComputeShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderReadOther => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_READ,
                input_stage: ANY_SHADER_STAGE,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::GENERAL,
            },

            AccessType::VertexShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::VERTEX_SHADER,
                output_stage: vk::PipelineStageFlags::VERTEX_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::FragmentShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                output_stage: vk::PipelineStageFlags::FRAGMENT_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::GeometryShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::GEOMETRY_SHADER,
                output_stage: vk::PipelineStageFlags::GEOMETRY_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessControlShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                output_stage: vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TessEvalShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                output_stage: vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::ComputeShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                output_stage: vk::PipelineStageFlags::COMPUTE_SHADER,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::AnyShaderWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::SHADER_WRITE,
                input_stage: ANY_SHADER_STAGE,
                output_stage: ANY_SHADER_STAGE,
                layout: vk::ImageLayout::GENERAL,
            },
            AccessType::TransferRead => AccessTypeInfo {
                access_mask: vk::AccessFlags::TRANSFER_READ,
                input_stage: vk::PipelineStageFlags::TRANSFER,
                output_stage: vk::PipelineStageFlags::empty(),
                layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            },
            AccessType::TransferWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::TRANSFER_WRITE,
                input_stage: vk::PipelineStageFlags::TRANSFER,
                output_stage: vk::PipelineStageFlags::TRANSFER,
                layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            },

            AccessType::ColorAttachmentWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                input_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                output_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },

            AccessType::ColorAttachmentReadWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::COLOR_ATTACHMENT_READ
                    | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                input_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                output_stage: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                input_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                output_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentReadWrite => AccessTypeInfo {
                access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                    | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                input_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                output_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            },
            AccessType::DepthStencilAttachmentRead => AccessTypeInfo {
                access_mask: vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
                input_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                output_stage: vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS
                    | vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                layout: vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            },
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct AccessTypeInfo {
    pub access_mask: vk::AccessFlags,
    pub input_stage: vk::PipelineStageFlags,
    pub output_stage: vk::PipelineStageFlags,
    pub layout: vk::ImageLayout,
}

/// Adds an execution dependency between a source and destination pass, identified by their submission numbers.
fn add_execution_dependency(
    src_snn: SubmissionNumber,
    src: Option<&mut Pass>,
    dst: &mut Pass,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    if let Some(src) = src {
        // --- Intra-batch synchronization
        if src_snn.queue() != dst.snn.queue() {
            // cross-queue dependency w/ timeline semaphore
            src.signal_after = true;
            let q = src_snn.queue();
            dst.wait_before = true;
            dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
            dst.wait_dst_stages[q as usize] |= dst_stage_mask;
        } else {
            // same-queue dependency, a pipeline barrier is sufficient
            dst.src_stage_mask |= src.output_stage_mask;
        }

        dst.preds.push(src.frame_index);
        src.succs.push(dst.frame_index);
    } else {
        // --- Inter-batch synchronization w/ timeline semaphore
        let q = src_snn.queue();
        dst.wait_before = true;
        dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
        dst.wait_dst_stages[q as usize] |= dst_stage_mask;
    }
}

/// Helper to find the pass given a submission number.
fn get_pass_mut<'a, 'b>(
    start_serial: u64,
    passes: &'a mut [Pass<'b>],
    snn: SubmissionNumber,
) -> Option<&'a mut Pass<'b>> {
    if snn.serial() <= start_serial {
        None
    } else {
        let pass_index = (snn.serial() - start_serial - 1) as usize;
        Some(&mut passes[pass_index])
    }
}

/// Builder object for passes.
pub struct PassBuilder<'a, 'frame> {
    batch: &'frame mut FrameInner<'a>,
    pass: Pass<'a>,
}

impl<'a, 'frame> PassBuilder<'a, 'frame> {
    pub fn register_image_access(&mut self, id: ImageId, access_type: AccessType) {
        let AccessTypeInfo {
            access_mask,
            input_stage,
            output_stage,
            layout,
        } = access_type.to_access_info();
        self.register_image_access_2(id, access_mask, input_stage, output_stage, layout);
    }

    /// Registers an image access made by this pass.
    pub fn register_image_access_2(
        &mut self,
        id: ImageId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
        layout: vk::ImageLayout,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                layout,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    pub fn register_buffer_access(&mut self, id: BufferId, access_type: AccessType) {
        let AccessTypeInfo {
            access_mask,
            input_stage,
            output_stage,
            ..
        } = access_type.to_access_info();
        self.register_buffer_access_2(id, access_mask, input_stage, output_stage);
    }

    pub fn register_buffer_access_2(
        &mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                layout: vk::ImageLayout::UNDEFINED,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    /// Sets the command handler for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_commands(
        &mut self,
        commands: impl FnOnce(&mut CommandContext, vk::CommandBuffer) + 'a,
    ) {
        self.pass.commands = Some(Box::new(commands));
    }
}

struct FrameInner<'a> {
    base_serial: u64,
    context: &'a mut Context,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the frame
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass<'a>>,
}

impl<'a> FrameInner<'a> {
    /// Registers an access to a resource within the specified pass and updates the dependency graph.
    ///
    /// This is the meat of the automatic synchronization system: given the known state of the resources,
    /// this function infers the necessary execution barriers, memory barriers, and layout transitions,
    /// and updates the state of the resources.
    fn add_resource_dependency(
        &mut self,
        pass: &mut Pass,
        id: ResourceId,
        access: &ResourceAccessDetails,
    ) {
        //------------------------
        // first, add the resource into the set of temporaries used within this frame
        let resource = self.context.resources.get_mut(id).unwrap();
        if self.temporary_set.insert(id) {
            // this is the first time the resource has been used in the frame
            match resource.memory {
                ResourceMemory::Aliasable(_) => {
                    if resource.tracking.has_writer() || resource.tracking.has_readers() {
                        panic!("transient resource was already used in a previous frame")
                    }
                }
                _ => {}
            }

            self.temporaries.push(id);
        }

        //------------------------
        let is_write = !access.output_stage.is_empty() || resource.tracking.layout != access.layout;

        // update input stage mask
        pass.input_stage_mask |= access.input_stage;

        // handle external semaphore dependency
        let semaphore = mem::take(&mut resource.tracking.wait_binary_semaphore);
        if semaphore != vk::Semaphore::null() {
            pass.wait_binary_semaphores.push(semaphore);
            pass.wait_before = true;
        }

        //------------------------
        // infer execution dependencies
        if is_write {
            if !resource.tracking.has_readers() && resource.tracking.has_writer() {
                // write-after-write
                add_execution_dependency(
                    resource.tracking.writer,
                    get_pass_mut(self.base_serial, &mut self.passes, resource.tracking.writer),
                    pass,
                    access.input_stage,
                );
            } else {
                // write-after-read
                for q in 0..MAX_QUEUES as u8 {
                    if resource.tracking.readers[q] != 0 {
                        let src_snn = SubmissionNumber::new(q, resource.tracking.readers[q]);
                        add_execution_dependency(
                            src_snn,
                            get_pass_mut(self.base_serial, &mut self.passes, src_snn),
                            pass,
                            access.input_stage,
                        );
                    }
                }
            }
            // update the resource writer
            pass.output_stage_mask = access.output_stage;
        } else {
            if resource.tracking.has_writer() {
                // read-after-write
                // NOTE a read without a write is probably an uninitialized access
                add_execution_dependency(
                    resource.tracking.writer,
                    get_pass_mut(self.base_serial, &mut self.passes, resource.tracking.writer),
                    pass,
                    access.input_stage,
                );
            }
        }

        //------------------------
        // infer memory barriers

        // Q: do we need a memory barrier?
        // A: we need a memory barrier if
        //      - if the operation needs to see all previous writes to the resource:
        //          - if the resource visibility mask doesn't contain the requested access type
        //      - if a layout transition is necessary
        //
        // Note: if the pass overwrites the resource entirely, then the operation technically doesn't need to
        // see the last version of the resource.

        // are all writes to the resource visible to the requested access type?
        // resource was last written in a previous frame, so all writes are made visible
        // by the semaphore wait inserted by the execution dependency (FIXME is that true? is it not "available" only?)
        // resource.tracking.writer.serial() <= self.base_serial ||
        // resource visible to all MEMORY_READ, or to the requested mask
        let writes_visible = resource
            .tracking
            .visibility_mask
            .contains(access.access_mask)
            || resource
                .tracking
                .visibility_mask
                .contains(vk::AccessFlags::MEMORY_READ | access.access_mask);

        // is the layout of the resource different? do we need a transition?
        let layout_transition = resource.tracking.layout != access.layout;
        // is there a possible write-after-write hazard, that requires a memory dependency?
        let write_after_write_hazard =
            is_write && is_write_access(resource.tracking.availability_mask);

        if !writes_visible || layout_transition || write_after_write_hazard {
            // if the last writer of the serial is in another frame, all writes are made available because of the semaphore
            // wait inserted by the execution dependency. Otherwise, we need to consider the available writes on the resource.
            let src_access_mask = if resource.tracking.writer.serial() <= self.base_serial {
                vk::AccessFlags::empty()
            } else {
                resource.tracking.availability_mask
            };
            // no need to make memory visible if we're only writing to the resource
            let dst_access_mask = if !is_read_access(access.access_mask) {
                vk::AccessFlags::empty()
            } else {
                access.access_mask
            };
            // the resource access needs a memory barrier
            match &resource.kind {
                ResourceKind::Image(img) => {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: format_aspect_mask(img.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    };

                    pass.image_memory_barriers.push(vk::ImageMemoryBarrier {
                        src_access_mask,
                        dst_access_mask,
                        old_layout: resource.tracking.layout,
                        new_layout: access.layout,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: img.handle,
                        subresource_range,
                        ..Default::default()
                    })
                }
                ResourceKind::Buffer(buf) => {
                    pass.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                        src_access_mask,
                        dst_access_mask,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: buf.handle,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    })
                }
            }
            // all previous writes to the resource have been made available by the barrier ...
            resource.tracking.availability_mask = vk::AccessFlags::empty();
            // ... but not *made visible* to all access types: update the access types that can now see the resource
            resource.tracking.visibility_mask |= access.access_mask;
            resource.tracking.layout = access.layout;
        }

        if is_write_access(access.access_mask) {
            // we're writing to the resource, so reset visibility...
            resource.tracking.visibility_mask = vk::AccessFlags::empty();
            // ... but signal that there is data to be made available for this resource.
            resource.tracking.availability_mask |= access.access_mask;
        }

        // update output stage
        // FIXME doubt
        if is_write {
            resource.tracking.stages = access.output_stage;
            resource.tracking.clear_readers();
            resource.tracking.writer = pass.snn;
        } else {
            // update the resource readers
            resource
                .tracking
                .readers
                .assign_max_serial(pass.snn.queue(), pass.snn.serial());
        }

        pass.accesses.push(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }
}

pub struct Frame<'a> {
    base_serial: u64,
    frame_serial: FrameSerialNumber,
    inner: RefCell<FrameInner<'a>>,
}

impl<'a> Frame<'a> {
    pub(crate) fn new(context: &'a mut Context) -> Frame<'a> {
        let base_serial = context.next_serial;
        Frame {
            base_serial,
            frame_serial: FrameSerialNumber(context.submitted_frame_count + 1),
            inner: RefCell::new(FrameInner {
                base_serial,
                context,
                temporaries: vec![],
                temporary_set: TemporarySet::new(),
                passes: vec![],
            }),
        }
    }

    /// Returns the context from which this frame was started
    pub fn context(&self) -> RefMut<Context> {
        RefMut::map(self.inner.borrow_mut(), |inner| inner.context)
    }

    /// Returns this frame's serial
    pub fn serial(&self) -> FrameSerialNumber {
        self.frame_serial
    }

    /// Creates a transient image.
    pub fn create_transient_image(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_create_info: &ImageResourceCreateInfo,
    ) -> ImageInfo {
        let mut inner = self.inner.borrow_mut();
        inner
            .context
            .create_image(name, memory_info, image_create_info, true)
    }

    /// Creates a transient buffer.
    pub fn create_transient_buffer(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {
        let mut inner = self.inner.borrow_mut();
        inner
            .context
            .create_buffer(name, memory_info, buffer_create_info, true)
    }

    /// Creates a transient mapped buffer. The caller must ensure that `memory_info` describes
    /// a mappable buffer (HOST_VISIBLE).
    pub fn create_mapped_buffer(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {
        assert!(memory_info
            .required_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE));
        let mut inner = self.inner.borrow_mut();
        let buffer_info = inner
            .context
            .create_buffer(name, memory_info, buffer_create_info, true);
        assert!(!buffer_info.mapped_ptr.is_null());
        buffer_info
    }

    pub fn add_graphics_pass(&self, name: &str, handler: impl FnOnce(&mut PassBuilder<'a, '_>)) {
        self.add_pass(name, PassKind::Render, false, handler)
    }

    /// Starts building a compute pass
    pub fn add_compute_pass(
        &self,
        name: &str,
        async_compute: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassKind::Compute, async_compute, handler)
    }

    /// Starts building a transfer pass
    pub fn add_transfer_pass(
        &self,
        name: &str,
        async_transfer: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassKind::Transfer, async_transfer, handler)
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&self, name: &str, image: &SwapchainImage) {
        self.add_pass(
            name,
            PassKind::Present {
                swapchain: image.swapchain_handle,
                image_index: image.image_index,
            },
            false,
            |builder| {
                builder.register_image_access_2(
                    image.image_info.id,
                    vk::AccessFlags::MEMORY_READ,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            },
        );
    }

    /// Common code for `build_xxx_pass`
    fn add_pass(
        &self,
        name: &str,
        kind: PassKind,
        async_pass: bool,
        handler: impl FnOnce(&mut PassBuilder<'a, '_>),
    ) {
        let mut inner = self.inner.borrow_mut();
        let serial = inner.context.get_next_serial();
        let batch_index = inner.passes.len();
        let queue_index = match kind {
            PassKind::Compute if async_pass => inner.context.device.queues_info.indices.compute,
            PassKind::Transfer if async_pass => inner.context.device.queues_info.indices.transfer,
            PassKind::Present { .. } => inner.context.device.queues_info.indices.present,
            _ => inner.context.device.queues_info.indices.graphics,
        };
        let snn = SubmissionNumber::new(queue_index, serial);

        let mut builder = PassBuilder {
            batch: &mut *inner,
            pass: Pass::new(name, batch_index, snn, kind),
        };

        handler(&mut builder);

        let pass = builder.pass;
        inner.passes.push(pass);
    }

    /// Uploads the given object to a transient buffer with the given usage flags and returns the
    /// created buffer resource.
    pub fn upload<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &T,
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of::<T>();
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);
        unsafe {
            ptr::write(mapped_ptr as *mut T, *data);
        }
        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Uploads the given slice to a transient buffer with the given usage flags and returns the created buffer resource.
    ///
    /// Example:
    /// ```
    /// let vertices : [[f32;2];3] = [[0.0,0.0],[0.0,1.0],[1.0,1.0]];
    /// batch.upload_slice(graal::vk::BufferUsageFlags::VERTEX_BUFFER, &vertices, "vertices");
    /// ```
    pub fn upload_slice<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of_val(data);
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut T, data.len());
        }

        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Allocates a buffer to hold a value of type `T` and returns a reference to the uninitialized
    /// value inside the buffer.
    ///
    /// Example:
    /// ```
    /// let (buf, val) = batch.alloc_upload::<[f32;4]>(graal::vk::BufferUsageFlags::UNIFORM_BUFFER, None);
    /// unsafe { val.as_mut_ptr().write(Default::default()); }
    /// ```
    pub fn alloc_upload<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of::<T>();
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);

        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Allocates a buffer to hold `size` values of type `T` and returns the slice of the uninitialized
    /// values inside the buffer.
    ///
    /// Example:
    /// ```
    /// let (buf, val) = batch.alloc_upload::<[f32;4]>(graal::vk::BufferUsageFlags::UNIFORM_BUFFER, None);
    /// unsafe { val.as_mut_ptr().write(Default::default()); }
    /// ```
    pub fn alloc_upload_slice<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        size: usize,
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of::<T>() * size;
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);
        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Creates a transient buffer mapped in host-coherent memory.
    fn create_upload_buffer(
        &self,
        usage: vk::BufferUsageFlags,
        byte_size: usize,
        name: Option<&str>,
    ) -> BufferInfo {
        self.inner.borrow_mut().context.create_buffer(
            name.unwrap_or("upload buffer"),
            &ResourceMemoryInfo::HOST_VISIBLE_COHERENT,
            &BufferResourceCreateInfo {
                usage,
                byte_size: byte_size as u64,
                map_on_create: true,
            },
            true,
        )
    }

    /// Finishes building the frame and submits all the passes to the command queues.
    pub fn finish(mut self) {
        //println!("====== Batch #{} ======", self.batch_serial.0);

        let mut inner = self.inner.into_inner();
        let context = inner.context;

        /*println!("Passes:");
        for p in inner.passes.iter() {
            println!("- `{}` ({:?})", p.name, p.snn);
            if p.wait_before {
                println!("    semaphore wait:");
                if p.wait_serials[0] != 0 {
                    println!("        0:{}|{:?}", p.wait_serials[0], p.wait_dst_stages[0]);
                }
                if p.wait_serials[1] != 0 {
                    println!("        1:{}|{:?}", p.wait_serials[1], p.wait_dst_stages[1]);
                }
                if p.wait_serials[2] != 0 {
                    println!("        2:{}|{:?}", p.wait_serials[2], p.wait_dst_stages[2]);
                }
                if p.wait_serials[3] != 0 {
                    println!("        3:{}|{:?}", p.wait_serials[3], p.wait_dst_stages[3]);
                }
            }
            println!(
                "    input execution barrier: {:?}->{:?}",
                p.src_stage_mask, p.input_stage_mask
            );
            println!("    input memory barriers:");
            for imb in p.image_memory_barriers.iter() {
                let id = context.image_resource_by_handle(imb.image);
                print!("        image handle={:?} ", imb.image);
                if !id.is_null() {
                    print!(
                        "(id={:?}, name={})",
                        id,
                        context.resources.get(id).unwrap().name
                    );
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?} layout:{:?}->{:?}",
                    imb.src_access_mask, imb.dst_access_mask, imb.old_layout, imb.new_layout
                );
            }
            for bmb in p.buffer_memory_barriers.iter() {
                let id = context.buffer_resource_by_handle(bmb.buffer);
                print!("        buffer handle={:?} ", bmb.buffer);
                if !id.is_null() {
                    print!(
                        "(id={:?}, name={})",
                        id,
                        context.resources.get(id).unwrap().name
                    );
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?}",
                    bmb.src_access_mask, bmb.dst_access_mask
                );
            }

            println!("    output stage: {:?}", p.output_stage_mask);
            if p.signal_after {
                println!("    semaphore signal: {:?}", p.snn);
            }
        }

        println!("Final resource states: ");
        for &id in inner.temporaries.iter() {
            let resource = context.resources.get(id).unwrap();
            println!("`{}`", resource.name);
            println!("    stages={:?}", resource.tracking.stages);
            println!("    avail={:?}", resource.tracking.availability_mask);
            println!("    vis={:?}", resource.tracking.visibility_mask);
            println!("    layout={:?}", resource.tracking.layout);

            if resource.tracking.has_readers() {
                println!("    readers: ");
                if resource.tracking.readers[0] != 0 {
                    println!("        0:{}", resource.tracking.readers[0]);
                }
                if resource.tracking.readers[1] != 0 {
                    println!("        1:{}", resource.tracking.readers[1]);
                }
                if resource.tracking.readers[2] != 0 {
                    println!("        2:{}", resource.tracking.readers[2]);
                }
                if resource.tracking.readers[3] != 0 {
                    println!("        3:{}", resource.tracking.readers[3]);
                }
            }
            if resource.tracking.has_writer() {
                println!("    writer: {:?}", resource.tracking.writer);
            }
        }*/

        // First, wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the frames that are not in use anymore.
        context.wait_for_frames_in_flight();
        // Allocate and assign memory for all transient resources of this frame.
        let transient_allocations = context.allocate_memory_for_transients(
            self.base_serial,
            &inner.temporaries,
            &inner.passes,
        );
        // All resources now have a block of device memory assigned. We're ready to
        // build the command buffers and submit them to the device queues.
        let submission_result = context.submit_frame(&mut inner.passes);
        // Add this frame to the list of "frames in flight": frames that might be executing on the device.
        // When this frame is completed, all resources of the frame will be automatically recycled.
        // This includes:
        // - device memory blocks for transient allocations
        // - command buffers (in command pools)
        // - image views
        // - framebuffers
        // - descriptor sets
        context.in_flight.push_back(FrameInFlight {
            signalled_serials: submission_result.signalled_serials,
            transient_allocations,
            command_pools: submission_result.command_pools,
            image_views: submission_result.image_views,
            framebuffers: submission_result.framebuffers,
            descriptor_sets: submission_result.descriptor_sets,
            semaphores: submission_result.semaphores,
        });

        context.submitted_frame_count += 1;
        context.dump_state();
    }
}
