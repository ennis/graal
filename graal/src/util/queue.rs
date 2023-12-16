use crate::{
    aspects_for_format, util::DeviceExt, vk, BufferUsage, Image, ImageCopyBuffer, ImageCopyView, ImageCreateInfo,
    ImageDataLayout, Queue, Size3D,
};

pub trait QueueExt {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]);
    fn create_image_with_data(&mut self, name: &str, create_info: &ImageCreateInfo, data: &[u8]) -> Image;
}

impl QueueExt for Queue {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]) {
        let staging_buffer =
            self.device()
                .upload_array_buffer("upload_image_data staging buffer", BufferUsage::TRANSFER_SRC, data);

        let width = image.image.size().width;
        let height = image.image.size().height;

        let mut cmd_buf = self.create_command_buffer();
        let mut encoder = cmd_buf.begin_blit();
        unsafe {
            encoder.copy_buffer_to_image(
                ImageCopyBuffer {
                    buffer: &staging_buffer.buffer,
                    layout: ImageDataLayout {
                        offset: 0,
                        row_length: Some(width),
                        image_height: Some(height),
                    },
                },
                image,
                vk::Extent3D {
                    width: size.width,
                    height: size.height,
                    depth: size.depth,
                },
            );
        }
        encoder.finish();

        self.submit([cmd_buf]).unwrap();
    }

    fn create_image_with_data(&mut self, name: &str, create_info: &ImageCreateInfo, data: &[u8]) -> Image {
        let image = self.device().create_image(name, create_info);
        self.upload_image_data(
            ImageCopyView {
                image: &image,
                mip_level: 0,
                origin: vk::Offset3D { x: 0, y: 0, z: 0 },
                aspect: aspects_for_format(create_info.format), // FIXME
            },
            Size3D {
                width: create_info.width,
                height: create_info.height,
                depth: create_info.depth,
            },
            data,
        );
        image
    }
}
