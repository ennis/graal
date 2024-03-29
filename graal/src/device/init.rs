//! Device initialization helpers
use std::{
    cell::Cell,
    ffi::{c_void, CStr, CString},
    ptr,
    rc::Rc,
    sync::{Arc, Mutex},
};

use tracing::debug;

use crate::{
    instance::{get_vulkan_entry, get_vulkan_instance, vk_khr_surface},
    platform_impl, vk, ResourceMaps,
};

use super::{CommandStream, Device, DeviceCreateError, DeviceInner, DeviceTracker, QueueFamilyConfig, QueueShared};

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

const DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_swapchain",
    //"VK_KHR_dynamic_rendering",   // promoted to core in 1.3
    "VK_KHR_push_descriptor",
    "VK_EXT_extended_dynamic_state3",
    "VK_EXT_line_rasterization",
    "VK_EXT_mesh_shader",
    "VK_EXT_conservative_rasterization",
    "VK_EXT_fragment_shader_interlock",
    //"VK_EXT_descriptor_buffer",
];

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

                let global_index = queues.len() as u32;
                queues.push(Arc::new(QueueShared {
                    family: cfg.family_index,
                    index_in_family: i,
                    index: global_index,
                    queue,
                    timeline,
                    next_submission_timestamp: Cell::new(1),
                    free_command_pools: Mutex::new(vec![]),
                }));
            }
        }

        // Create the GPU memory allocator
        let allocator_create_desc = gpu_allocator::vulkan::AllocatorCreateDesc {
            physical_device,
            debug_settings: Default::default(),
            device: device.clone(),     // not cheap!
            instance: instance.clone(), // not cheap!
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        };

        let allocator =
            gpu_allocator::vulkan::Allocator::new(&allocator_create_desc).expect("failed to create GPU allocator");

        // Extensions
        let vk_khr_swapchain = ash::extensions::khr::Swapchain::new(instance, &device);
        let vk_ext_shader_object = ash::extensions::ext::ShaderObject::new(instance, &device);
        let vk_khr_push_descriptor = ash::extensions::khr::PushDescriptor::new(instance, &device);
        let vk_ext_extended_dynamic_state3 = ash::extensions::ext::ExtendedDynamicState3::new(instance, &device);
        let vk_ext_mesh_shader = ash::extensions::ext::MeshShader::new(instance, &device);
        let vk_ext_descriptor_buffer = ash::extensions::ext::DescriptorBuffer::new(instance, &device);
        let physical_device_memory_properties = instance.get_physical_device_memory_properties(physical_device);
        let platform_extensions = platform_impl::PlatformExtensions::load(entry, instance, &device);

        let mut physical_device_descriptor_buffer_properties =
            vk::PhysicalDeviceDescriptorBufferPropertiesEXT::default();
        let mut physical_device_properties = vk::PhysicalDeviceProperties2 {
            p_next: &mut physical_device_descriptor_buffer_properties as *mut _ as *mut c_void,
            ..Default::default()
        };

        instance.get_physical_device_properties2(physical_device, &mut physical_device_properties);

        // Create the shader compiler instance
        let compiler = shaderc::Compiler::new().expect("failed to create the shader compiler");

        Ok(Device {
            inner: Rc::new(DeviceInner {
                device,
                platform_extensions,
                physical_device,
                physical_device_properties,
                physical_device_descriptor_buffer_properties,
                physical_device_memory_properties,
                queues,
                allocator: Mutex::new(allocator),
                vk_khr_swapchain,
                vk_ext_shader_object,
                vk_khr_push_descriptor,
                vk_ext_mesh_shader,
                vk_ext_extended_dynamic_state3,
                vk_ext_descriptor_buffer,
                image_ids: Mutex::new(Default::default()),
                buffer_ids: Mutex::new(Default::default()),
                tracker: Mutex::new(DeviceTracker::new()),
                deletion_lists: Mutex::new(vec![]),
                sampler_cache: Mutex::new(Default::default()),
                compiler,
                groups: Mutex::new(Default::default()),
                free_command_pools: Mutex::new(Default::default()),
                image_view_ids: Mutex::new(Default::default()),
                dropped_resources: Mutex::new(ResourceMaps::new()),
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
        let device_queue_create_infos = &[vk::DeviceQueueCreateInfo {
            flags: Default::default(),
            queue_family_index: graphics_queue_family,
            queue_count: 1,
            p_queue_priorities: queue_priorities.as_ptr(),
            ..Default::default()
        }];

        let mut fragment_shader_interlock_features = vk::PhysicalDeviceFragmentShaderInterlockFeaturesEXT {
            //p_next: &mut timeline_features as *mut _ as *mut c_void,
            fragment_shader_pixel_interlock: vk::TRUE,
            ..Default::default()
        };

        let mut descriptor_buffer_features = vk::PhysicalDeviceDescriptorBufferFeaturesEXT {
            p_next: &mut fragment_shader_interlock_features as *mut _ as *mut c_void,
            descriptor_buffer: vk::TRUE,
            ..Default::default()
        };

        let mut mesh_shader_features = vk::PhysicalDeviceMeshShaderFeaturesEXT {
            p_next: &mut descriptor_buffer_features as *mut _ as *mut c_void,
            task_shader: vk::TRUE,
            mesh_shader: vk::TRUE,
            ..Default::default()
        };

        let mut ext_dynamic_state = vk::PhysicalDeviceExtendedDynamicState3FeaturesEXT {
            p_next: &mut mesh_shader_features as *mut _ as *mut c_void,
            extended_dynamic_state3_tessellation_domain_origin: vk::TRUE,
            extended_dynamic_state3_depth_clamp_enable: vk::TRUE,
            extended_dynamic_state3_polygon_mode: vk::TRUE,
            extended_dynamic_state3_rasterization_samples: vk::TRUE,
            extended_dynamic_state3_sample_mask: vk::TRUE,
            extended_dynamic_state3_alpha_to_coverage_enable: vk::TRUE,
            extended_dynamic_state3_alpha_to_one_enable: vk::TRUE,
            extended_dynamic_state3_logic_op_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_equation: vk::TRUE,
            extended_dynamic_state3_color_write_mask: vk::TRUE,
            extended_dynamic_state3_rasterization_stream: vk::TRUE,
            extended_dynamic_state3_conservative_rasterization_mode: vk::TRUE,
            extended_dynamic_state3_extra_primitive_overestimation_size: vk::TRUE,
            extended_dynamic_state3_depth_clip_enable: vk::TRUE,
            extended_dynamic_state3_sample_locations_enable: vk::TRUE,
            extended_dynamic_state3_color_blend_advanced: vk::TRUE,
            extended_dynamic_state3_provoking_vertex_mode: vk::TRUE,
            extended_dynamic_state3_line_rasterization_mode: vk::TRUE,
            extended_dynamic_state3_line_stipple_enable: vk::TRUE,
            extended_dynamic_state3_depth_clip_negative_one_to_one: vk::TRUE,
            extended_dynamic_state3_viewport_w_scaling_enable: vk::TRUE,
            extended_dynamic_state3_viewport_swizzle: vk::TRUE,
            extended_dynamic_state3_coverage_to_color_enable: vk::TRUE,
            extended_dynamic_state3_coverage_to_color_location: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_mode: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_table_enable: vk::TRUE,
            extended_dynamic_state3_coverage_modulation_table: vk::TRUE,
            extended_dynamic_state3_coverage_reduction_mode: vk::TRUE,
            extended_dynamic_state3_representative_fragment_test_enable: vk::TRUE,
            extended_dynamic_state3_shading_rate_image_enable: vk::TRUE,
            ..Default::default()
        };

        let mut line_rasterization_features = vk::PhysicalDeviceLineRasterizationFeaturesEXT {
            p_next: &mut ext_dynamic_state as *mut _ as *mut c_void,
            bresenham_lines: vk::TRUE,
            smooth_lines: vk::TRUE,
            rectangular_lines: vk::TRUE,
            ..Default::default()
        };

        let mut vk13_features = vk::PhysicalDeviceVulkan13Features {
            p_next: &mut line_rasterization_features as *mut _ as *mut c_void,
            synchronization2: vk::TRUE,
            dynamic_rendering: vk::TRUE,
            ..Default::default()
        };

        let mut vk12_features = vk::PhysicalDeviceVulkan12Features {
            p_next: &mut vk13_features as *mut _ as *mut c_void,
            descriptor_indexing: vk::TRUE,
            shader_uniform_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_storage_buffer_array_non_uniform_indexing: vk::TRUE,
            shader_sampled_image_array_non_uniform_indexing: vk::TRUE,
            shader_storage_image_array_non_uniform_indexing: vk::TRUE,
            runtime_descriptor_array: vk::TRUE,
            buffer_device_address: vk::TRUE,
            timeline_semaphore: vk::TRUE,
            storage_buffer8_bit_access: vk::TRUE,
            shader_int8: vk::TRUE,
            scalar_block_layout: vk::TRUE,
            ..Default::default()
        };

        let mut features2 = vk::PhysicalDeviceFeatures2 {
            p_next: &mut vk12_features as *mut _ as *mut c_void,
            features: vk::PhysicalDeviceFeatures {
                tessellation_shader: vk::TRUE,
                fill_mode_non_solid: vk::TRUE,
                sampler_anisotropy: vk::TRUE,
                shader_storage_image_extended_formats: vk::TRUE,
                fragment_stores_and_atomics: vk::TRUE,
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
    pub fn physical_device_properties(&self) -> &vk::PhysicalDeviceProperties2 {
        &self.inner.physical_device_properties
    }

    pub fn create_command_stream(&self, queue_index: usize) -> CommandStream {
        let command_pool = self.get_or_create_command_pool(self.inner.queues[queue_index].family);
        CommandStream::new(self.clone(), command_pool, self.inner.queues[queue_index].clone())
    }
}
