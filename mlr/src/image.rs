use crate::vk;
use graal::{aspects_for_format, device::Device};
use std::rc::Rc;

use crate::attachments::Attachment;
pub use graal::device::ImageHandle;

/// Wrapper around a Vulkan image.
#[derive(Clone, Debug)]
pub struct ImageAny {
    device: Device,
    handle: ImageHandle,
    usage: vk::ImageUsageFlags,
    type_: vk::ImageType,
    format: vk::Format,
    extent: vk::Extent3D,
    /// Cached ImageViews.
    views: Vec<(ImageViewInfo, vk::ImageView)>,
}

/// A view over a subresource range of an image.
#[derive(Copy, Clone)]
pub struct ImageView<'a> {
    pub image: &'a ImageAny,
    pub view_info: ImageViewInfo,
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

#[derive(Debug)]
struct ImageInner {}

impl ImageAny {
    /// Returns the `vk::ImageType` of the image.
    pub fn image_type(&self) -> vk::ImageType {
        self.0.type_
    }

    /// Returns the `vk::Format` of the image.
    pub fn format(&self) -> vk::Format {
        self.0.format
    }

    /// Returns the `vk::Extent3D` of the image.
    pub fn extent(&self) -> vk::Extent3D {
        self.0.extent
    }

    /// Returns the usage flags of the image.
    pub fn usage(&self) -> vk::ImageUsageFlags {
        self.0.usage
    }

    /// Returns an image view of the specified mip level.
    pub fn view(&self, view_info: &ImageViewInfo) -> ImageView {
        ImageView {
            image: self,
            view_info: view_info.clone(),
        }
    }

    /// Returns the image handle.
    pub fn handle(&self) -> ImageHandle {
        self.0.handle
    }

    /// Returns an attachment descriptor for the base mip level of this image.
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
    }

    /// Creates a `VkImageView` object.
    pub(crate) fn create_view(&self, info: &ImageViewInfo) -> vk::ImageView {
        if let Some((_, view)) = self.views.iter().find(|(i, _)| i == info) {
            return *view;
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: self.handle.vk,
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
        unsafe {
            self.device
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        }
    }
}
