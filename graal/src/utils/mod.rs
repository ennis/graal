use crate::{
    device::ImageInfo,
    queue::{Queue, ResourceState},
    vk,
};

pub unsafe fn blit_images(
    queue: &mut Queue,
    src_image: ImageInfo,
    dst_image: ImageInfo,
    width: u32,
    height: u32,
    aspect_mask: vk::ImageAspectFlags,
) {
    let cb = queue.allocate_command_buffer();
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
                x: width as i32,
                y: height as i32,
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
                x: width as i32,
                y: height as i32,
                z: 1,
            },
        ],
    }];

    let device = queue.device();
    device
        .begin_command_buffer(
            cb,
            &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            },
        )
        .unwrap();
    device.cmd_blit_image(
        cb,
        src_image.handle,
        vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
        dst_image.handle,
        vk::ImageLayout::TRANSFER_DST_OPTIMAL,
        regions,
        vk::Filter::NEAREST,
    );
    device.end_command_buffer(cb).unwrap();

    let mut blit = queue.build_submission();
    blit.set_name("blit_images");
    blit.use_image(src_image.id, ResourceState::TRANSFER_SRC);
    blit.use_image(dst_image.id, ResourceState::TRANSFER_DST);
    blit.push_command_buffer(cb);
    blit.submit().expect("blit_images failed");
}
