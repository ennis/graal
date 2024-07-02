use std::{mem, mem::MaybeUninit, slice};

use ash::vk;

use crate::{
    command::{do_cmd_push_constants, do_cmd_push_descriptor_sets, DescriptorWrite},
    ArgumentKind, Arguments, CommandStream, ComputePipeline, Device,
};

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct ComputeEncoder<'a> {
    stream: &'a mut CommandStream,
    command_buffer: vk::CommandBuffer,
    pipeline_layout: vk::PipelineLayout,
}

impl<'a> ComputeEncoder<'a> {
    pub fn device(&self) -> &Device {
        self.stream.device()
    }

    pub fn bind_arguments<A: Arguments>(&mut self, set: u32, arguments: &A) {
        let mut descriptor_writes = vec![];
        for arg in arguments.arguments() {
            match arg.kind {
                ArgumentKind::Image { image_view, access } => {
                    self.stream.use_image_view(image_view, access);
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
                    self.stream.use_buffer(&buffer, access);
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
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                set,
                descriptor_writes.as_slice(),
            );
        }
    }

    // SAFETY: TBD
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let device = self.stream.device();
        unsafe {
            device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
        }
        self.pipeline_layout = pipeline.pipeline_layout;
        // TODO: we need to hold a reference to the pipeline until the command buffers are submitted
    }

    /// Binds push constants.
    pub fn bind_push_constants<P>(&mut self, data: &P)
    where
        P: Copy + ?Sized,
    {
        unsafe {
            do_cmd_push_constants(
                &self.stream.device,
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                slice::from_raw_parts(data as *const P as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.stream.flush_barriers();
        unsafe {
            self.stream
                .device
                .cmd_dispatch(self.command_buffer, group_count_x, group_count_y, group_count_z);
        }
    }

    pub fn finish(self) {
        // Nothing to do. Provided for consistency with other encoders.
    }
}

impl CommandStream {
    /// Start a compute pass
    pub fn begin_compute(&mut self) -> ComputeEncoder {
        let command_buffer = self.get_or_create_command_buffer();
        ComputeEncoder {
            stream: self,
            command_buffer,
            pipeline_layout: Default::default(),
        }
    }
}
