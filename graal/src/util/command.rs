use crate::{
    aspects_for_format, util::DeviceExt, vk, BufferUsage, CommandStream, Image, ImageCopyBuffer, ImageCopyView,
    ImageCreateInfo, ImageDataLayout, ImageSubresourceLayers, ImageUsage, Point3D, Rect3D, Size3D,
};

pub trait CommandStreamExt {
    /// Copies the data to a region of an image.
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]);

    /// Creates an image and copies the data to it.
    ///
    /// # Notes
    /// The image will be created with `ImageUsage::TRANSFER_DST` if it is not already specified in
    /// `create_info.usage`.
    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, data: &[u8]) -> Image;

    /// Shorthand to `blit_image` for blitting the top-level mip level of an image.
    fn blit_full_image_top_mip_level(&mut self, src: &Image, dst: &Image);
}

impl CommandStreamExt for CommandStream {
    fn upload_image_data(&mut self, image: ImageCopyView, size: Size3D, data: &[u8]) {
        let staging_buffer = self.device().upload_array_buffer(BufferUsage::TRANSFER_SRC, data);

        let width = image.image.size().width;
        let height = image.image.size().height;

        self.copy_buffer_to_image(
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
    }

    fn create_image_with_data(&mut self, create_info: &ImageCreateInfo, data: &[u8]) -> Image {
        let mut create_info_with_transfer_dst = create_info.clone();
        create_info_with_transfer_dst.usage |= ImageUsage::TRANSFER_DST;
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

    fn blit_full_image_top_mip_level(&mut self, src: &Image, dst: &Image) {
        let width = src.width() as i32;
        let height = src.height() as i32;
        self.blit_image(
            &src,
            ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            Rect3D {
                min: Point3D { x: 0, y: 0, z: 0 },
                max: Point3D {
                    x: width,
                    y: height,
                    z: 1,
                },
            },
            &dst,
            ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            Rect3D {
                min: Point3D { x: 0, y: 0, z: 0 },
                max: Point3D {
                    x: width,
                    y: height,
                    z: 1,
                },
            },
            vk::Filter::NEAREST,
        );
    }
}
