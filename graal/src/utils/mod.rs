use crate::{vk, AccessType, Frame, ImageInfo};
use ash::version::DeviceV1_0;

pub fn blit_images(
    frame: &Frame,
    src_image: ImageInfo,
    dst_image: ImageInfo,
    size: (u32, u32),
    aspect_mask: vk::ImageAspectFlags,
) {
    frame.add_graphics_pass("blit_images", |pass| {
        pass.register_image_access(src_image.id, AccessType::TransferRead);
        pass.register_image_access(dst_image.id, AccessType::TransferWrite);

        pass.set_commands(move |context, command_buffer| {
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
                context.vulkan_device().cmd_blit_image(
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
