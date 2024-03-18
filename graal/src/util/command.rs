use crate::{
    aspects_for_format, util::DeviceExt, vk, BufferUsage, CommandStream, Image, ImageCopyBuffer, ImageCopyView,
    ImageCreateInfo, ImageDataLayout, Size3D,
};

pub trait CommandStreamExt {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]);
    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, data: &[u8]) -> Image;
}

impl CommandStreamExt for CommandStream {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]) {
        let staging_buffer = self.device().upload_array_buffer(BufferUsage::TRANSFER_SRC, data);

        let width = image.image.size().width;
        let height = image.image.size().height;

        let mut encoder = self.begin_blit();
        encoder.copy_buffer_to_image(
            ImageCopyBuffer {
                buffer: &staging_buffer.untyped,
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
        encoder.finish();
    }

    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, data: &[u8]) -> Image {
        let image = self.device().create_image(create_info);
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
