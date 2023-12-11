use crate::{
    aspects_for_format,
    attachments::Attachment,
    device::{Device, RefCounted},
    vk, ImageId,
};

/// Wrapper around a Vulkan image.
#[derive(Debug)]
pub struct ImageAny {
    device: Device,
    id: RefCounted<ImageId>,
    handle: vk::Image,
    usage: vk::ImageUsageFlags,
    type_: vk::ImageType,
    format: vk::Format,
    extent: vk::Extent3D,
}

/// Describe a subresource range of an image.
///
/// Same as VkImageSubresourceRange, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceRange {
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceLayers {
    pub aspect_mask: vk::ImageAspectFlags,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

/// The parameters of an image view.
///
/// Same as VkImageViewCreateInfo, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub subresource_range: ImageSubresourceRange,
    pub component_mapping: [vk::ComponentSwizzle; 4],
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl ImageAny {
    pub(super) fn new(
        device: Device,
        id: RefCounted<ImageId>,
        handle: vk::Image,
        usage: vk::ImageUsageFlags,
        type_: vk::ImageType,
        format: vk::Format,
        extent: vk::Extent3D,
    ) -> Self {
        ImageAny {
            device,
            id,
            handle,
            usage,
            type_,
            format,
            extent,
        }
    }

    /// Returns the `vk::ImageType` of the image.
    pub fn image_type(&self) -> vk::ImageType {
        self.type_
    }

    /// Returns the `vk::Format` of the image.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    /// Returns the `vk::Extent3D` of the image.
    pub fn size(&self) -> vk::Extent3D {
        self.extent
    }

    /// Returns the usage flags of the image.
    pub fn usage(&self) -> vk::ImageUsageFlags {
        self.usage
    }

    pub fn id(&self) -> RefCounted<ImageId> {
        self.id.clone()
    }

    /// Returns the image handle.
    pub fn handle(&self) -> vk::Image {
        self.handle
    }

    /*/// Returns an attachment descriptor for the base mip level of this image.
    ///
    /// The default load and store operations are used: `load_op == LOAD` and `store_op == STORE`.
    ///
    /// # Panics
    ///
    /// * If the image is not a 2D image (`vk::ImageViewType::TYPE_2D`).
    pub fn attachment(&self) -> Attachment {
        Attachment {
            image: self,
            view_info: ImageViewInfo {
                view_type: match self.image_type() {
                    vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                    _ => panic!("unsupported image type for attachment"),
                },
                format: Default::default(),
                subresource_range: ImageSubresourceRange {
                    aspect_mask: aspects_for_format(self.format()),
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                component_mapping: [
                    vk::ComponentSwizzle::IDENTITY,
                    vk::ComponentSwizzle::IDENTITY,
                    vk::ComponentSwizzle::IDENTITY,
                    vk::ComponentSwizzle::IDENTITY,
                ],
            },
            load_op: Default::default(),
            store_op: Default::default(),
            clear_value: None,
        }
    }*/

    /// Creates an image view for the base mip level of this image,
    /// suitable for use as a rendering attachment.
    pub fn create_top_level_view(&self) -> ImageView {
        self.create_view(&ImageViewInfo {
            view_type: match self.image_type() {
                vk::ImageType::TYPE_2D => vk::ImageViewType::TYPE_2D,
                _ => panic!("unsupported image type for attachment"),
            },
            format: self.format(),
            subresource_range: ImageSubresourceRange {
                aspect_mask: aspects_for_format(self.format()),
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            component_mapping: [
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
                vk::ComponentSwizzle::IDENTITY,
            ],
        })
    }

    /// Creates an `ImageView` object.
    pub(crate) fn create_view(&self, info: &ImageViewInfo) -> ImageView {
        // TODO: check that format is compatible

        // FIXME: support non-zero base mip level
        if info.subresource_range.base_mip_level != 0 {
            unimplemented!("non-zero base mip level");
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: self.handle,
            view_type: info.view_type,
            format: info.format,
            components: vk::ComponentMapping {
                r: info.component_mapping[0],
                g: info.component_mapping[1],
                b: info.component_mapping[2],
                a: info.component_mapping[3],
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: info.subresource_range.aspect_mask,
                base_mip_level: info.subresource_range.base_mip_level,
                level_count: info.subresource_range.level_count,
                base_array_layer: info.subresource_range.base_array_layer,
                layer_count: info.subresource_range.layer_count,
            },
            ..Default::default()
        };

        // SAFETY: the device is valid, the create info is valid
        let handle = unsafe {
            self.device
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        };

        ImageView {
            device: self.device.clone(),
            parent_image: self.id.clone(),
            handle,
            image_handle: self.handle,
            format: info.format,
            original_format: self.format,
            // TODO: size of mip level
            size: self.extent,
        }
    }
}

#[derive(Debug)]
pub struct ImageView {
    device: Device,
    parent_image: RefCounted<ImageId>,
    image_handle: vk::Image,
    handle: vk::ImageView,
    format: vk::Format,
    original_format: vk::Format,
    size: vk::Extent3D,
}

impl ImageView {
    /// Returns the format of the image view.
    pub fn format(&self) -> vk::Format {
        self.format
    }

    pub fn size(&self) -> vk::Extent3D {
        self.size
    }

    pub fn handle(&self) -> vk::ImageView {
        self.handle
    }

    pub(super) fn image_handle(&self) -> vk::Image {
        self.image_handle
    }

    pub(super) fn original_format(&self) -> vk::Format {
        self.original_format
    }

    pub(super) fn parent_id(&self) -> RefCounted<ImageId> {
        self.parent_image.clone()
    }
}

impl Drop for ImageView {
    fn drop(&mut self) {
        unsafe {
            self.device.delete_later(self.handle);
        }
    }
}
