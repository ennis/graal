//! Blit command encoders
use ash::vk;

use crate::{
    BufferAccess, BufferRangeUntyped, ClearColorValue, CommandStream, Image, ImageAccess, ImageCopyBuffer,
    ImageCopyView, ImageSubresourceLayers, Rect3D,
};

impl CommandStream {
    pub fn fill_buffer(&mut self, buffer: &BufferRangeUntyped, data: u32) {
        self.use_buffer(&buffer.buffer, BufferAccess::COPY_DST);
        self.flush_barriers();

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device
                .cmd_fill_buffer(cb, buffer.buffer.handle, buffer.offset, buffer.size, data);
        }
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        self.use_image(image, ImageAccess::COPY_DST);
        self.flush_barriers();

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_clear_color_image(
                cb,
                image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color_value.into(),
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: vk::REMAINING_MIP_LEVELS,
                    base_array_layer: 0,
                    layer_count: vk::REMAINING_ARRAY_LAYERS,
                }],
            );
        }
    }

    pub fn copy_image_to_image(
        &mut self,
        source: ImageCopyView<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.use_image(&source.image, ImageAccess::COPY_SRC);
        self.use_image(&destination.image, ImageAccess::COPY_DST);

        // TODO: this is not required for multi-planar formats
        assert_eq!(source.aspect, destination.aspect);

        let regions = [vk::ImageCopy {
            src_subresource: vk::ImageSubresourceLayers {
                aspect_mask: source.aspect,
                mip_level: source.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            src_offset: source.origin,
            dst_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect,
                mip_level: destination.mip_level, // FIXME: should be the same
                base_array_layer: 0,
                layer_count: 1,
            },
            dst_offset: destination.origin,
            extent: copy_size,
        }];

        self.flush_barriers();

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_copy_image(
                cb,
                source.image.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                destination.image.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
    }

    pub fn copy_buffer_to_image(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.use_buffer(&source.buffer, BufferAccess::COPY_SRC);
        self.use_image(&destination.image, ImageAccess::COPY_DST);

        let regions = [vk::BufferImageCopy {
            buffer_offset: source.layout.offset,
            buffer_row_length: source.layout.row_length.unwrap_or(0),
            buffer_image_height: source.layout.image_height.unwrap_or(0),
            image_subresource: vk::ImageSubresourceLayers {
                aspect_mask: destination.aspect,
                mip_level: destination.mip_level,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: destination.origin,
            image_extent: copy_size,
        }];

        self.flush_barriers();

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_copy_buffer_to_image(
                cb,
                source.buffer.handle(),
                destination.image.handle(),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &regions,
            );
        }
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
        self.use_image(src, ImageAccess::COPY_SRC);
        self.use_image(dst, ImageAccess::COPY_DST);
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

        self.flush_barriers();

        // SAFETY: command buffer is OK, params OK
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_blit_image(
                cb,
                src.handle,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                dst.handle,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &blits,
                filter,
            );
        }
    }
}
