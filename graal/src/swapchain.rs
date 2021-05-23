use crate::{Device, ImageInfo, Frame, Context, ImageId};
use ash::vk;
use std::ptr;
use crate::context::pass::{SemaphoreWait, SemaphoreWaitKind};
use crate::context::{ImageRegistrationInfo, ResourceRegistrationInfo, ResourceOwnership};

/// Chooses a swapchain surface format among a list of supported formats.
fn get_preferred_swapchain_surface_format(
    surface_formats: &[vk::SurfaceFormatKHR],
) -> vk::SurfaceFormatKHR {
    surface_formats
        .iter()
        .find_map(|&fmt| {
            if fmt.format == vk::Format::B8G8R8A8_SRGB
                && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR
            {
                Some(fmt)
            } else {
                None
            }
        })
        .expect("no suitable surface format available")
}

/// Chooses a present mode among a list of supported modes.
fn get_preferred_present_mode(
    available_present_modes: &[vk::PresentModeKHR],
) -> vk::PresentModeKHR {
    if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// Computes the preferred swap extent.
fn get_preferred_swap_extent(
    framebuffer_size: (u32, u32),
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: framebuffer_size.0.clamp(
                capabilities.min_image_extent.width,
                capabilities.max_image_extent.width,
            ),
            height: framebuffer_size.1.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

#[derive(Debug)]
pub struct Swapchain {
    pub handle: vk::SwapchainKHR,
    pub surface: vk::SurfaceKHR,
    pub images: Vec<vk::Image>,
    pub format: vk::Format,
}

/// Contains information about an image in a swapchain.
#[derive(Copy, Clone, Debug)]
pub struct SwapchainImage {
    /// Handle of the swapchain that owns this image.
    pub swapchain_handle: vk::SwapchainKHR,
    /// Index of the image in the swap chain.
    pub image_index: u32,
    pub image_info: ImageInfo,
}

impl Swapchain {
    /// Creates a swapchain object.
    pub unsafe fn new(
        context: &Context,
        surface: vk::SurfaceKHR,
        size: (u32, u32),
    ) -> Swapchain {
        let mut swapchain = Swapchain {
            handle: Default::default(),
            surface,
            images: vec![],
            format: Default::default(),
        };
        swapchain.resize(context, size);
        swapchain
    }

    /// Resizes a swapchain.
    pub unsafe fn resize(&mut self, context: &Context, size: (u32, u32)) {
        let phy = context.device.physical_device;
        let capabilities = context.device
            .vk_khr_surface
            .get_physical_device_surface_capabilities(phy, self.surface)
            .unwrap();
        let formats = context.device
            .vk_khr_surface
            .get_physical_device_surface_formats(phy, self.surface)
            .unwrap();
        let present_modes = context.device
            .vk_khr_surface
            .get_physical_device_surface_present_modes(phy, self.surface)
            .unwrap();

        let image_format = get_preferred_swapchain_surface_format(&formats);
        let present_mode = get_preferred_present_mode(&present_modes);
        let image_extent = get_preferred_swap_extent(size, &capabilities);
        let image_count = if capabilities.max_image_count > 0
            && capabilities.min_image_count + 1 > capabilities.max_image_count
        {
            capabilities.max_image_count
        } else {
            capabilities.min_image_count + 1
        };

        let create_info = vk::SwapchainCreateInfoKHR {
            flags: Default::default(),
            surface: self.surface,
            min_image_count: image_count,
            image_format: image_format.format,
            image_color_space: image_format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: self.handle,
            ..Default::default()
        };

        let new_handle = context.device
            .vk_khr_swapchain
            .create_swapchain(&create_info, None)
            .expect("failed to create swapchain");
        if self.handle != vk::SwapchainKHR::null() {
            // FIXME what if the images are in use?
            context.device.vk_khr_swapchain.destroy_swapchain(self.handle, None);
        }


        self.handle = new_handle;
        self.images = context.device
            .vk_khr_swapchain
            .get_swapchain_images(self.handle)
            .unwrap();
        self.format = image_format.format;
    }

    /// Acquires the next image in the swapchain.
    /// See `vkAcquireNextImageKHR`.
    pub unsafe fn acquire_next_image(&mut self, context: &mut Context) -> SwapchainImage
    {
        let image_available = context.create_semaphore();

        let (image_index, _suboptimal) = context.device
            .vk_khr_swapchain
            .acquire_next_image(
                self.handle,
                1_000_000_000,
                image_available,
                vk::Fence::null(),
            )
            .expect("AcquireNextImage failed");

        let handle = self.images[image_index as usize];
        let name = format!("swapchain {:?} image #{}", handle, image_index);
        let id = context.register_image_resource(
            ImageRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name: &name,
                    initial_wait: Some(SemaphoreWait {
                        semaphore: image_available,
                        owned: true,
                        dst_stage: Default::default(),
                        wait_kind: SemaphoreWaitKind::Binary
                    }),
                    ownership: ResourceOwnership::Referenced
                },
                handle,
                format: self.format
            }
        );

        SwapchainImage {
            swapchain_handle: self.handle,
            image_info: ImageInfo {
                id,
                handle,
            },
            image_index,
        }
    }
}
