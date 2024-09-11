//! Blit command encoders
use ash::vk;

use crate::{
    Barrier, BufferRangeUntyped, BufferUntyped, ClearColorValue, CommandStream, Image, ImageCopyBuffer, ImageCopyView,
    ImageSubresourceLayers, Point3D, Rect3D,
};

impl CommandStream {
    pub fn fill_buffer(&mut self, buffer: &BufferRangeUntyped, data: u32) {
        self.reference_resource(&buffer.buffer);
        self.barrier(Barrier::new().transfer_write());

        let cb = self.get_or_create_command_buffer();
        unsafe {
            // SAFETY: FFI call and parameters are valid
            self.device
                .cmd_fill_buffer(cb, buffer.buffer.handle, buffer.offset, buffer.size, data);
        }
    }

    // TODO specify subresources
    pub fn clear_image(&mut self, image: &Image, clear_color_value: ClearColorValue) {
        self.reference_resource(image);
        self.barrier(Barrier::new().transfer_write_image(image));

        let cb = self.get_or_create_command_buffer();
        unsafe {
            // SAFETY: FFI call and parameters are valid
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
        // TODO: this is not required for multi-planar formats
        assert_eq!(source.aspect, destination.aspect);

        // TODO barriers should automatically reference the resources in the command stream
        self.reference_resource(source.image);
        self.reference_resource(destination.image);

        self.barrier(
            Barrier::new()
                .transfer_read_image(source.image)
                .transfer_write_image(destination.image),
        );

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

    /// Copies data from one buffer to another.
    pub fn copy_buffer(
        &mut self,
        source: &BufferUntyped,
        src_offset: u64,
        destination: &BufferUntyped,
        dst_offset: u64,
        size: u64,
    ) {
        assert!(src_offset + size <= source.size);
        assert!(dst_offset + size <= destination.size);

        self.reference_resource(source);
        self.reference_resource(destination);
        self.barrier(Barrier::new().transfer_read().transfer_write());

        // SAFETY: FFI call and parameters are valid
        let cb = self.get_or_create_command_buffer();
        unsafe {
            self.device.cmd_copy_buffer(
                cb,
                source.handle(),
                destination.handle(),
                &[vk::BufferCopy {
                    src_offset,
                    dst_offset,
                    size,
                }],
            );
        }
    }

    /// Copies data from a buffer to an image.
    pub fn copy_buffer_to_image(
        &mut self,
        source: ImageCopyBuffer<'_>,
        destination: ImageCopyView<'_>,
        copy_size: vk::Extent3D,
    ) {
        self.reference_resource(source.buffer);
        self.reference_resource(destination.image);

        self.barrier(Barrier::new().transfer_read().transfer_write_image(destination.image));

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
        self.reference_resource(src);
        self.reference_resource(dst);

        self.barrier(Barrier::new().transfer_read_image(src).transfer_write_image(dst));

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
