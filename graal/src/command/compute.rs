use std::{ffi::c_void, mem};

use ash::vk;

use crate::{
    command::{do_cmd_push_constants, do_cmd_push_descriptor_sets, EncoderBase},
    Arguments, CommandBuffer, ComputePipeline, Device,
};

enum ComputeCommand {
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
    BindPushConstants {
        /// Offset in encoder buffer (in bytes)
        offset: u32,
        /// Size in bytes
        size: u32,
    },
    Dispatch {
        group_count_x: u32,
        group_count_y: u32,
        group_count_z: u32,
    },
}

/// A context object to submit commands to a command buffer after a pipeline has been bound to it.
///
/// This is used in `RenderPass::bind_pipeline`.
pub struct ComputeEncoder<'a> {
    base: EncoderBase<'a>,
    commands: Vec<ComputeCommand>,
}

impl<'a> ComputeEncoder<'a> {
    pub fn device(&self) -> &Device {
        self.base.command_buffer.device()
    }

    pub fn bind_arguments<A: Arguments>(&mut self, set: u32, arguments: &A) {
        unsafe {
            let (offset, count) = self.base.record_arguments(arguments);
            self.commands.push(ComputeCommand::BindArguments { set, offset, count })
        }
    }

    // SAFETY: TBD
    pub fn bind_compute_pipeline(&mut self, pipeline: &ComputePipeline) {
        self.commands.push(ComputeCommand::BindPipeline {
            pipeline: pipeline.pipeline,
            pipeline_layout: pipeline.pipeline_layout,
        })
    }

    /// Binds push constants.
    pub fn bind_push_constants<P>(&mut self, data: &P)
    where
        P: Copy + ?Sized,
    {
        let size = mem::size_of_val(data);
        let offset = unsafe { self.base.push_constants_raw(size, data as *const _ as *const c_void) };
        self.commands.push(ComputeCommand::BindPushConstants { offset, size: size as u32 });
    }

    pub fn dispatch(&mut self, group_count_x: u32, group_count_y: u32, group_count_z: u32) {
        self.commands.push(ComputeCommand::Dispatch {
            group_count_x,
            group_count_y,
            group_count_z,
        })
    }

    unsafe fn record_compute_commands(&mut self) {
        self.base.flush_pipeline_barriers();

        let device = &self.base.command_buffer.device;
        let command_buffer = self.base.command_buffer.command_buffer;
        let mut current_pipeline_layout = vk::PipelineLayout::null();

        for command in self.commands.iter() {
            match *command {
                ComputeCommand::BindPipeline { pipeline, pipeline_layout } => {
                    device.cmd_bind_pipeline(command_buffer, vk::PipelineBindPoint::COMPUTE, pipeline);
                    current_pipeline_layout = pipeline_layout;
                }
                ComputeCommand::BindArguments { set, offset, count } => {
                    do_cmd_push_descriptor_sets(
                        device,
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        current_pipeline_layout,
                        set,
                        &self.base.descriptor_writes[(offset as usize)..(offset as usize + count as usize)],
                    );
                }
                ComputeCommand::BindPushConstants { offset, size } => {
                    do_cmd_push_constants(
                        device,
                        command_buffer,
                        vk::PipelineBindPoint::COMPUTE,
                        current_pipeline_layout,
                        &self.base.push_constant_data[(offset as usize)..(offset as usize + size as usize)],
                    );
                }
                ComputeCommand::Dispatch {
                    group_count_x,
                    group_count_y,
                    group_count_z,
                } => {
                    device.cmd_dispatch(command_buffer, group_count_x, group_count_y, group_count_z);
                }
            }
        }
    }
}

impl<'a> Drop for ComputeEncoder<'a> {
    fn drop(&mut self) {
        unsafe {
            self.record_compute_commands();
        }
    }
}

impl CommandBuffer {
    /// Start a compute pass
    pub fn begin_compute(&mut self) -> ComputeEncoder {
        ComputeEncoder {
            base: EncoderBase::new(self),
            commands: vec![],
        }
    }
}
