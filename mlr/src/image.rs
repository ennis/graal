use crate::vk;
use graal::device::Device;
use std::rc::Rc;

pub use graal::device::ImageHandle;

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceRange {
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub subresource_range: vk::ImageSubresourceRange,
    pub component_mapping: [vk::ComponentSwizzle; 4],
}

#[derive(Debug)]
struct ImageInner {
    device: Rc<Device>,
    handle: ImageHandle,
    usage: vk::ImageUsageFlags,
    type_: vk::ImageType,
    format: vk::Format,
    extent: vk::Extent3D,

    /// Cached ImageViews.
    views: Vec<(ImageViewInfo, vk::ImageView)>,
}

/// Wrapper around a Vulkan image.
#[derive(Clone, Debug)]
pub struct ImageAny(Rc<ImageInner>);

impl ImageAny {
    fn create_view(&self, info: &ImageViewInfo) -> vk::ImageView {
        if let Some((_, view)) = self.0.views.iter().find(|(i, _)| i == info) {
            return *view;
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: self.0.handle.vk,
            view_type: info.view_type,
            format: info.format,
            components: vk::ComponentMapping {
                r: info.component_mapping[0],
                g: info.component_mapping[1],
                b: info.component_mapping[2],
                a: info.component_mapping[3],
            },
            subresource_range: info.subresource_range,
            ..Default::default()
        };

        // SAFETY: the device is valid, the create info is valid
        unsafe {
            self.0
                .device
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        }
    }
}
