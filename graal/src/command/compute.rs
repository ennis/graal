use std::{mem, mem::MaybeUninit, slice};

use ash::vk;

use crate::{Barrier, CommandStream, ComputePipeline, Descriptor, Device, GpuResource};

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

    pub fn reference_resource<R: GpuResource>(&mut self, resource: &R) {
        self.stream.reference_resource(resource);
    }

    pub unsafe fn bind_descriptor_set(&mut self, index: u32, set: vk::DescriptorSet) {
        self.stream.device.cmd_bind_descriptor_sets(
            self.command_buffer,
            vk::PipelineBindPoint::COMPUTE,
            self.pipeline_layout,
            index,
            &[set],
            &[],
        )
    }

    pub fn push_descriptors(&mut self, set: u32, bindings: &[(u32, Descriptor)]) {
        assert!(
            self.pipeline_layout != vk::PipelineLayout::null(),
            "encoder must have a pipeline bound before binding arguments"
        );

        unsafe {
            self.stream.do_cmd_push_descriptor_set(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                set,
                bindings,
            );
        }
    }

    // SAFETY: TBD
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        let device = self.stream.device();
        unsafe {
            device.cmd_bind_pipeline(self.command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if pipeline.bindless {
                self.stream.bind_bindless_descriptor_sets(
                    self.command_buffer,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.pipeline_layout,
                );
            }
        }
        self.pipeline_layout = pipeline.pipeline_layout;

        // TODO: we need to hold a reference to the pipeline until the command buffers are submitted
    }

    pub fn barrier(&mut self, barrier: Barrier) {
        self.stream.barrier(barrier);
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
                vk::PipelineBindPoint::COMPUTE,
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
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                slice::from_raw_parts(data.as_ptr() as *const MaybeUninit<u8>, mem::size_of_val(data)),
            );
        }
    }

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
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
