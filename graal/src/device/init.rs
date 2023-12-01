//! Device initialization helpers
use super::{Device, DeviceCreateError, DeviceInner, Feature, QueueData, QueueFamilyConfig};
use crate::{
    instance::{get_vulkan_entry, get_vulkan_instance, vk_khr_surface},
    platform_impl,
    queue::Queue,
    vk,
};
use std::{
    cell::{Cell, RefCell},
    ffi::{c_void, CStr, CString},
    ptr,
    rc::Rc,
};
use tracing::debug;

struct PhysicalDeviceAndProperties {
    physical_device: vk::PhysicalDevice,
    properties: vk::PhysicalDeviceProperties,
    //features: vk::PhysicalDeviceFeatures,
}

/// Chooses a present mode among a list of supported modes.
pub(super) fn get_preferred_present_mode(available_present_modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    if available_present_modes.contains(&vk::PresentModeKHR::MAILBOX) {
        vk::PresentModeKHR::MAILBOX
    } else if available_present_modes.contains(&vk::PresentModeKHR::IMMEDIATE) {
        vk::PresentModeKHR::IMMEDIATE
    } else {
        vk::PresentModeKHR::FIFO
    }
}

/// Computes the preferred swap extent.
pub(super) fn get_preferred_swap_extent(
    framebuffer_size: (u32, u32),
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    if capabilities.current_extent.width != u32::MAX {
        capabilities.current_extent
    } else {
        vk::Extent2D {
            width: framebuffer_size
                .0
                .clamp(capabilities.min_image_extent.width, capabilities.max_image_extent.width),
            height: framebuffer_size.1.clamp(
                capabilities.min_image_extent.height,
                capabilities.max_image_extent.height,
            ),
        }
    }
}

unsafe fn select_physical_device(instance: &ash::Instance) -> PhysicalDeviceAndProperties {
    let physical_devices = instance
        .enumerate_physical_devices()
        .expect("failed to enumerate physical devices");
    if physical_devices.is_empty() {
        panic!("no device with vulkan support");
    }

    let mut selected_phy = None;
    let mut selected_phy_properties = Default::default();
    //let mut selected_phy_features = Default::default();
    for phy in physical_devices {
        let props = instance.get_physical_device_properties(phy);
        let _features = instance.get_physical_device_features(phy);
        if props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
            selected_phy = Some(phy);
            selected_phy_properties = props;
            //selected_phy_features = features;
        }
    }
    // TODO fallbacks

    PhysicalDeviceAndProperties {
        physical_device: selected_phy.expect("no suitable physical device"),
        properties: selected_phy_properties,
        //features: selected_phy_features,
    }
}

unsafe fn find_queue_family(
    phy: vk::PhysicalDevice,
    vk_khr_surface: &ash::extensions::khr::Surface,
    queue_families: &[vk::QueueFamilyProperties],
    flags: vk::QueueFlags,
    present_surface: Option<vk::SurfaceKHR>,
) -> u32 {
    let mut best_queue_family: Option<u32> = None;
    let mut best_flags = 0u32;
    let mut index = 0u32;
    for queue_family in queue_families {
        if queue_family.queue_flags.contains(flags) {
            // matches the intended usage
            // if present_surface != nullptr, check that it also supports presentation
            // to the given surface
            if let Some(surface) = present_surface {
                if !vk_khr_surface
                    .get_physical_device_surface_support(phy, index, surface)
                    .unwrap()
                {
                    // does not support presentation, skip it
                    continue;
                }
            }

            if let Some(ref mut i) = best_queue_family {
                // there was already a queue for the specified usage,
                // change it only if it is more specialized.
                // to determine if it is more specialized, count number of bits (XXX sketchy?)
                if queue_family.queue_flags.as_raw().count_ones() < best_flags.count_ones() {
                    *i = index;
                    best_flags = queue_family.queue_flags.as_raw();
                }
            } else {
                best_queue_family = Some(index);
                best_flags = queue_family.queue_flags.as_raw();
            }
        }
        index += 1;
    }

    best_queue_family.expect("could not find a compatible queue")
}

const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

impl Device {
    fn find_compatible_memory_type_internal(
        &self,
        memory_type_bits: u32,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.inner.physical_device_memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0
                && self.inner.physical_device_memory_properties.memory_types[i as usize]
                    .property_flags
                    .contains(memory_properties)
            {
                return Some(i);
            }
        }
        None
    }

    /// Returns the index of the first memory type compatible with the specified memory type bitmask and additional memory property flags.
    pub(crate) fn find_compatible_memory_type(
        &self,
        memory_type_bits: u32,
        required_memory_properties: vk::MemoryPropertyFlags,
        preferred_memory_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        // first, try required+preferred, otherwise fallback on just required
        self.find_compatible_memory_type_internal(
            memory_type_bits,
            required_memory_properties | preferred_memory_properties,
        )
        .or_else(|| self.find_compatible_memory_type_internal(memory_type_bits, required_memory_properties))
    }

    /*/// Returns whether this device is compatible for presentation on the specified surface.
    ///
    /// More precisely, it checks that the graphics queue created for this device can present to the given surface.
    pub unsafe fn is_compatible_for_presentation(&self, surface: vk::SurfaceKHR) -> bool {
        vk_khr_surface()
            .get_physical_device_surface_support(self.inner.physical_device, self.graphics_queue().1, surface)
            .unwrap()
    }*/

    // TODO: enabled features?
    pub unsafe fn from_existing(
        physical_device: vk::PhysicalDevice,
        device: vk::Device,
        queue_config: &[QueueFamilyConfig],
    ) -> Result<Device, DeviceCreateError> {
        let entry = get_vulkan_entry();
        let instance = get_vulkan_instance();
        let device = ash::Device::load(instance.fp_v1_0(), device);

        let mut queues = vec![];

        // fetch queues and create the timeline semaphore for all of them
        for cfg in queue_config.iter() {
            for i in 0..cfg.count {
                let queue = device.get_device_queue(cfg.family_index, i);

                // create timeline semaphore
                let mut timeline_create_info = vk::SemaphoreTypeCreateInfo {
                    semaphore_type: vk::SemaphoreType::TIMELINE,
                    initial_value: 0,
                    ..Default::default()
                };
                let semaphore_create_info = vk::SemaphoreCreateInfo {
                    p_next: &mut timeline_create_info as *mut _ as *mut c_void,
                    ..Default::default()
                };
                let timeline = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("failed to queue timeline semaphore");

                queues.push(QueueData {
                    family_index: cfg.family_index,
                    index: i,
                    queue,
                    timeline,
                    next_submission_timestamp: Cell::new(1),
                });
            }
        }

        // Create the GPU memory allocator
        let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            debug_settings: Default::default(),
            device: device.clone(),     // not cheap!
            instance: instance.clone(), // not cheap!
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        };

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator");

        // More stuff
        let vk_khr_swapchain = ash::extensions::khr::Swapchain::new(instance, &device);
        let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let platform_extensions = platform_impl::PlatformExtensions::load(entry, instance, &device);
        let physical_device_properties = instance.get_physical_device_properties(physical_device);

        Ok(Device {
            inner: Rc::new(DeviceInner {
                device,
                platform_extensions,
                physical_device,
                physical_device_properties,
                physical_device_memory_properties,
                queues,
                allocator: RefCell::new(allocator),
                vk_khr_swapchain,
                resources: RefCell::new(Default::default()),
                resource_groups: RefCell::new(Default::default()),
                deletion_lists: RefCell::new(vec![]),
            }),
        })
    }

    /// Creates a new `Device` that can render to the specified `present_surface` if one is specified.
    ///
    /// Also creates queues as requested.
    pub unsafe fn new(present_surface: Option<vk::SurfaceKHR>) -> Result<Device, DeviceCreateError> {
        let instance = get_vulkan_instance();
        let vk_khr_surface = vk_khr_surface();

        let phy = select_physical_device(instance);
        let queue_family_properties = instance.get_physical_device_queue_family_properties(phy.physical_device);
        let graphics_queue_family = find_queue_family(
            phy.physical_device,
            &vk_khr_surface,
            &queue_family_properties,
            vk::QueueFlags::GRAPHICS,
            present_surface,
        );

        debug!(
            "Selected physical device: {:?}",
            CStr::from_ptr(phy.properties.device_name.as_ptr())
        );

        // ------ Setup device create info ------
        let queue_priorities = [1.0f32];
        let mut device_queue_create_infos = &[vk::DeviceQueueCreateInfo {
            flags: Default::default(),
            queue_family_index: graphics_queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let mut timeline_features = vk::PhysicalDeviceTimelineSemaphoreFeatures {
            timeline_semaphore: vk::TRUE,
            ..Default::default()
        };

        let mut vk13_features = vk::PhysicalDeviceVulkan13Features {
            p_next: &mut timeline_features as *mut _ as *mut c_void,
            synchronization2: vk::TRUE,
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut vk13_features as *mut _ as *mut c_void,
            features: vk::PhysicalDeviceFeatures {
                tessellation_shader: vk::TRUE,
                fill_mode_non_solid: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                shader_storage_image_extended_formats: vk::TRUE,
                ..Default::default()
            },
            ..Default::default()
        };

        // Convert extension strings into C-strings
        let c_device_extensions: Vec<_> = DEVICE_EXTENSIONS
            .iter()
            .chain(platform_impl::PlatformExtensions::names().iter())
            .map(|&s| CString::new(s).unwrap())
            .collect();

        let device_extensions: Vec<_> = c_device_extensions.iter().map(|s| s.as_ptr()).collect();

        let device_create_info = vk::DeviceCreateInfo {
            p_next: &mut features2 as *mut _ as *mut c_void,
            flags: Default::default(),
            queue_create_info_count: device_queue_create_infos.len() as u32,
            p_queue_create_infos: device_queue_create_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            p_enabled_features: ptr::null(),
            ..Default::default()
        };

        // ------ Create device ------
        let device: ash::Device = instance
            .create_device(phy.physical_device, &device_create_info, None)
            .expect("could not create vulkan device");

        let queue_config = [QueueFamilyConfig {
            family_index: graphics_queue_family,
            count: 1,
        }];

        Self::from_existing(phy.physical_device, device.handle(), &queue_config)
    }

    /// Returns the physical device that this device was created on.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.inner.physical_device
    }

    /// Returns the physical device properties.
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.inner.physical_device_properties
    }

    ///
    pub fn get_queue(&self, family_index: u32, index: u32) -> Queue {
        let global_index = self
            .inner
            .queues
            .iter()
            .position(|q| q.family_index == family_index && q.index == index)
            .unwrap();
        self.get_queue_by_global_index(global_index)
    }

    pub fn get_queue_by_global_index(&self, global_index: usize) -> Queue {
        unsafe {
            Queue::new(
                self.clone(),
                global_index,
                self.inner.queues[global_index].queue,
                self.inner.queues[global_index].timeline,
                self.inner.queues[global_index].family_index,
                self.inner.queues[global_index].next_submission_timestamp.get(),
            )
        }
    }
}
