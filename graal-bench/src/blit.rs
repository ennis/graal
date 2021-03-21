use graal::{ash::version::DeviceV1_0, vk, ImageInfo};

fn blit_images(
    batch: &graal::Batch,
    src_image: ImageInfo,
    dst_image: ImageInfo,
    size: (u32, u32),
    aspect_mask: vk::ImageAspectFlags,
) {
    batch.add_graphics_pass("blit_images", |pass| {
        pass.add_image_usage(
            src_image.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::empty(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        );
        pass.add_image_usage(
            dst_image.id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        );

        pass.set_commands(|context, command_buffer| {
            let regions = &[vk::ImageBlit {
                src_subresource: vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: size.0 as i32,
                        y: size.1 as i32,
                        z: 1,
                    },
                ],
                dst_subresource: vk::ImageSubresourceLayers {
                    aspect_mask,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                dst_offsets: [
                    vk::Offset3D { x: 0, y: 0, z: 0 },
                    vk::Offset3D {
                        x: size.0 as i32,
                        y: size.1 as i32,
                        z: 1,
                    },
                ],
            }];

            unsafe {
                context.device().cmd_blit_image(
                    command_buffer,
                    src_image.handle,
                    vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    dst_image.handle,
                    vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    regions,
                    vk::Filter::NEAREST,
                );
            }
        });
    });
}
