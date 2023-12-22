//! Blit command encoders
use ash::vk;
use crate::{BufferRangeAny, ClearColorValue, Device, Image, ImageCopyBuffer, ImageCopyView, ImageDataLayout, ImageSubresourceLayers, Rect3D, ResourceUse};
use crate::tracker::Tracker;
use super::{CommandBuffer, EncoderBase};

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

pub struct BlitCommandEncoder<'a> {
    base: EncoderBase<'a>,
    commands: Vec<BlitCommand>,
}

impl<'a> BlitCommandEncoder<'a> {
    pub(super) fn new(command_buffer: &'a mut CommandBuffer) -> Self {
        Self {
            base: EncoderBase::new(command_buffer),
            commands: vec![],
        }
    }

    pub fn device(&self) -> &Device {
        self.base.command_buffer.device()
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
            handle: image.handle(),
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
            src: source.image.handle(),
            src_mip: source.mip_level,
            dst: destination.image.handle(),
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
            dst: destination.image.handle(),
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
        self.base.flush_pipeline_barriers();

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

impl CommandBuffer {
    /// Encode a blit operation
    pub fn begin_blit(&mut self) -> BlitCommandEncoder {
        BlitCommandEncoder::new(self)
    }
}