//! Device initialization helpers
use super::{Device, QueueIndices, QueuesInfo, MAX_QUEUES};
use crate::{
    instance::{VULKAN_ENTRY, VULKAN_INSTANCE},
    platform_impl, vk,
};
use std::{
    cell::RefCell,
    ffi::{c_void, CStr, CString},
    ptr,
};
use tracing::debug;

struct PhysicalDeviceAndProperties {
    phy: vk::PhysicalDevice,
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
        phy: selected_phy.expect("no suitable physical device"),
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

// Vulkan message callback
unsafe extern "system" fn debug_utils_message_callback(
    message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _message_types: vk::DebugUtilsMessageTypeFlagsEXT,
    p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT,
    _p_user_data: *mut c_void,
) -> vk::Bool32 {
    let message = CStr::from_ptr((*p_callback_data).p_message).to_str().unwrap();

    /*let message_id_name = CStr::from_ptr((*p_callback_data).p_message_id_name)
        .to_str()
        .unwrap();
    let objects = slice::from_raw_parts(
        (*p_callback_data).p_objects,
        (*p_callback_data).object_count as usize,
    );
    let queue_labels = slice::from_raw_parts(
        (*p_callback_data).p_queue_labels,
        (*p_callback_data).queue_label_count as usize,
    );*/

    /*// convert objects into a dumpable form
    #[derive(Debug)]
    struct Object<'a> {
        object_name: &'a str,
        object_type: vk::ObjectType,
        object_handle: u64,
    }
    let objects: Vec<_> = objects
        .iter()
        .map(|obj| Object {
            object_name: if obj.p_object_name.is_null() {
                "<unnamed>"
            } else {
                CStr::from_ptr(obj.p_object_name).to_str().unwrap()
            },
            object_type: obj.object_type,
            object_handle: obj.object_handle,
        })
        .collect();*/

    // translate message severity into tracing's log level
    match message_severity {
        vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => {
            tracing::event!(tracing::Level::TRACE, "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::INFO => {
            tracing::event!(tracing::Level::INFO, "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => {
            tracing::event!(tracing::Level::WARN, "{}", message);
        }
        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => {
            tracing::event!(tracing::Level::ERROR, "{}", message);
        }
        _ => {
            panic!("unexpected message severity flags")
        }
    };

    vk::FALSE
}

const DEVICE_EXTENSIONS: &[&str] = &["VK_KHR_swapchain"];

impl Device {
    fn find_compatible_memory_type_internal(
        &self,
        memory_type_bits: u32,
        memory_properties: vk::MemoryPropertyFlags,
    ) -> Option<u32> {
        for i in 0..self.physical_device_memory_properties.memory_type_count {
            if memory_type_bits & (1 << i) != 0
                && self.physical_device_memory_properties.memory_types[i as usize]
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

    /// Returns whether this device is compatible for presentation on the specified surface.
    ///
    /// More precisely, it checks that the graphics queue created for this device can present to the given surface.
    pub unsafe fn is_compatible_for_presentation(&self, surface: vk::SurfaceKHR) -> bool {
        self.vk_khr_surface
            .get_physical_device_surface_support(self.physical_device, self.graphics_queue().1, surface)
            .unwrap()
    }

    /// Creates a new `Device` that can render to the specified `present_surface` if one is specified.
    pub unsafe fn new(present_surface: Option<vk::SurfaceKHR>) -> Device {
        let instance: &ash::Instance = &*VULKAN_INSTANCE;
        let vk_khr_surface = ash::extensions::khr::Surface::new(&*VULKAN_ENTRY, instance);

        let phy = select_physical_device(instance);
        let queue_family_properties = instance.get_physical_device_queue_family_properties(phy.phy);

        let graphics_queue_family = find_queue_family(
            phy.phy,
            &vk_khr_surface,
            &queue_family_properties,
            vk::QueueFlags::GRAPHICS,
            present_surface,
        );
        let compute_queue_family = find_queue_family(
            phy.phy,
            &vk_khr_surface,
            &queue_family_properties,
            vk::QueueFlags::COMPUTE,
            None,
        );
        let transfer_queue_family = find_queue_family(
            phy.phy,
            &vk_khr_surface,
            &queue_family_properties,
            vk::QueueFlags::TRANSFER,
            None,
        );

        debug!(
            "Selected physical device: {:?}",
            CStr::from_ptr(phy.properties.device_name.as_ptr())
        );

        debug!(
            "Graphics queue family: {} ({:?})",
            graphics_queue_family, queue_family_properties[graphics_queue_family as usize].queue_flags
        );
        debug!(
            "Compute queue family: {} ({:?})",
            compute_queue_family, queue_family_properties[compute_queue_family as usize].queue_flags
        );
        debug!(
            "Transfer queue family: {} ({:?})",
            transfer_queue_family, queue_family_properties[transfer_queue_family as usize].queue_flags
        );

        let mut device_queue_create_infos = Vec::<vk::DeviceQueueCreateInfo>::new();
        let queue_priorities = [1.0f32];
        for &f in &[graphics_queue_family, compute_queue_family, transfer_queue_family] {
            let already_created = device_queue_create_infos.iter().any(|ci| ci.queue_family_index == f);
            if already_created {
                continue;
            }

            device_queue_create_infos.push(vk::DeviceQueueCreateInfo {
                flags: Default::default(),
                queue_family_index: f,
                queue_count: 1,
                p_queue_priorities: queue_priorities.as_ptr(),
                ..Default::default()
            });
        }

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

        let device: ash::Device = instance
            .create_device(phy.phy, &device_create_info, None)
            .expect("could not create vulkan device");
        let graphics_queue = device.get_device_queue(graphics_queue_family, 0);
        let compute_queue = device.get_device_queue(compute_queue_family, 0);
        let transfer_queue = device.get_device_queue(transfer_queue_family, 0);

        // queues are accessed by index. there are three different indices
        // - graphics
        // - compute
        // - transfer
        // (present is always == graphics)
        // Some of those indices may be equal. E.g. the graphics and compute queues might be the
        // same, and graphics == compute.
        let graphics_queue_index: u8 = 0u8;
        let compute_queue_index: u8 = if compute_queue == graphics_queue { 0 } else { 1 };
        let transfer_queue_index: u8 = if transfer_queue == graphics_queue {
            0
        } else if transfer_queue == compute_queue {
            1
        } else {
            2
        };

        //let mut queues_info = QueuesInfo::default();
        let mut queues = [vk::Queue::null(); MAX_QUEUES];
        let mut queue_families = [0u32; MAX_QUEUES];

        queues[graphics_queue_index as usize] = graphics_queue;
        queues[compute_queue_index as usize] = compute_queue;
        queues[transfer_queue_index as usize] = transfer_queue;

        queue_families[graphics_queue_index as usize] = graphics_queue_family;
        queue_families[compute_queue_index as usize] = compute_queue_family;
        queue_families[transfer_queue_index as usize] = transfer_queue_family;

        let queue_count = *[graphics_queue_index, compute_queue_index, transfer_queue_index]
            .iter()
            .max()
            .unwrap() as usize
            + 1;

        // create a timeline semaphore for each queue, for use by `Context`
        let mut queue_timelines = [vk::Semaphore::null(); MAX_QUEUES];
        {
            let mut timeline_create_info = vk::SemaphoreTypeCreateInfo {
                semaphore_type: vk::SemaphoreType::TIMELINE,
                initial_value: 0,
                ..Default::default()
            };
            let semaphore_create_info = vk::SemaphoreCreateInfo {
                p_next: &mut timeline_create_info as *mut _ as *mut c_void,
                ..Default::default()
            };

            for i in 0..queue_count {
                queue_timelines[i] = device
                    .create_semaphore(&semaphore_create_info, None)
                    .expect("failed to create semaphore");
            }
        }

        let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device: phy.phy,
            debug_settings: Default::default(),
            device: device.clone(),     // not cheap!
            instance: instance.clone(), // not cheap!
            buffer_device_address: false,
            /*flags: Default::default(),
            preferred_large_heap_block_size: 0, // default
            frame_in_use_count: 2,
            heap_size_limits: None,*/
            allocation_sizes: Default::default(),
        };

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator");

        let vk_khr_swapchain = ash::extensions::khr::Swapchain::new(&*VULKAN_INSTANCE, &device);

        // FIXME this should be created after the instance.
        let vk_ext_debug_utils = ash::extensions::ext::DebugUtils::new(&*VULKAN_ENTRY, &*VULKAN_INSTANCE);

        let debug_utils_messenger_create_info = vk::DebugUtilsMessengerCreateInfoEXT {
            flags: vk::DebugUtilsMessengerCreateFlagsEXT::empty(),
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE
                | vk::DebugUtilsMessageSeverityFlagsEXT::INFO
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                | vk::DebugUtilsMessageSeverityFlagsEXT::ERROR,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION,
            pfn_user_callback: Some(debug_utils_message_callback),
            p_user_data: ptr::null_mut(),
            ..Default::default()
        };

        let debug_messenger = vk_ext_debug_utils
            .create_debug_utils_messenger(&debug_utils_messenger_create_info, None)
            .unwrap();

        let physical_device_memory_properties = VULKAN_INSTANCE.get_physical_device_memory_properties(phy.phy);

        let platform_extensions = platform_impl::PlatformExtensions::load(&*VULKAN_ENTRY, &*VULKAN_INSTANCE, &device);

        Device {
            device,
            platform_extensions,
            physical_device: phy.phy,
            physical_device_properties: phy.properties,
            //physical_device_features: phy.features,
            physical_device_memory_properties,
            queues_info: QueuesInfo {
                queue_count,
                indices: QueueIndices {
                    graphics: graphics_queue_index,
                    compute: compute_queue_index,
                    present: graphics_queue_index,
                    transfer: transfer_queue_index,
                },
                families: queue_families,
                queues,
                timelines: queue_timelines,
            },
            allocator: RefCell::new(allocator),
            vk_khr_swapchain,
            vk_khr_surface,
            vk_ext_debug_utils,
            debug_messenger,
            resources: RefCell::new(Default::default()),
            resource_groups: RefCell::new(Default::default()),
        }
    }

    /// Returns the physical device that this device was created on.
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    /// Returns the physical device properties.
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.physical_device_properties
    }

    /// Returns the graphics queue handle and family index.
    pub fn graphics_queue(&self) -> (vk::Queue, u32) {
        let q = self.queues_info.indices.graphics as usize;
        (self.queues_info.queues[q], self.queues_info.families[q])
    }
}