use crate::{context::Frame, vk, ImageInfo, PassBuilder};

pub fn blit_images<UserContext>(
    frame: &mut Frame<UserContext>,
    src_image: ImageInfo,
    dst_image: ImageInfo,
    size: (u32, u32),
    aspect_mask: vk::ImageAspectFlags,
) {
    let blit_pass = PassBuilder::new()
        .name("blit_images")
        .image_dependency(
            src_image.id,
            vk::AccessFlags::TRANSFER_READ,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        )
        .image_dependency(
            dst_image.id,
            vk::AccessFlags::TRANSFER_WRITE,
            vk::PipelineStageFlags::TRANSFER,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        )
        .record_callback(move |context, _, command_buffer| {
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
    frame.add_pass(blit_pass);
}
