use ash::version::EntryV1_0;
use ash::vk;
use core::ptr;
use lazy_static::lazy_static;
use std::ffi::CStr;
use std::os::raw::c_char;

/// List of validation layers to enable
const VALIDATION_LAYERS: [*const c_char; 1] =
    [b"VK_LAYER_KHRONOS_validation\0".as_ptr() as *const c_char];

lazy_static! {
    pub(crate) static ref VULKAN_ENTRY: ash::Entry = initialize_vulkan_entry();
    pub(crate) static ref VULKAN_INSTANCE: ash::Instance = create_vulkan_instance();
}

fn initialize_vulkan_entry() -> ash::Entry {
    ash::Entry::new().expect("failed to initialize vulkan entry points")
}

/// Checks if all validation layers are supported
unsafe fn check_validation_layer_support() -> bool {
    let available_layers = VULKAN_ENTRY
        .enumerate_instance_layer_properties()
        .expect("failed to enumerate instance layers");
    VALIDATION_LAYERS.iter().all(|&required_layer| {
        available_layers.iter().any(|&layer| {
            CStr::from_ptr(layer.layer_name.as_ptr()) == CStr::from_ptr(required_layer)
        })
    })
}

fn create_vulkan_instance() -> ash::Instance {
    unsafe {
        let validation_available = check_validation_layer_support();
        if !validation_available {
            eprintln!("validation layer not available");
        }

        let mut instance_extensions = Vec::new();

        instance_extensions.push(b"VK_KHR_get_surface_capabilities2\0".as_ptr() as *const c_char);
        instance_extensions.push(b"VK_EXT_debug_utils\0".as_ptr() as *const c_char);

        #[cfg(windows)]
        {
            instance_extensions.push(b"VK_KHR_surface\0".as_ptr() as *const c_char);
            instance_extensions.push(b"VK_KHR_win32_surface\0".as_ptr() as *const c_char);
        }
        // TODO the rest

        let application_info = vk::ApplicationInfo {
            p_application_name: b"GRAAL\0".as_ptr() as *const c_char,
            application_version: 0,
            p_engine_name: b"GRAAL\0".as_ptr() as *const c_char,
            engine_version: 0,
            api_version: vk::make_version(1, 2, 0),
            ..Default::default()
        };

        let mut instance_create_info = vk::InstanceCreateInfo {
            flags: Default::default(),
            p_application_info: &application_info,
            enabled_layer_count: 0,
            pp_enabled_layer_names: ptr::null(),
            enabled_extension_count: instance_extensions.len() as u32,
            pp_enabled_extension_names: instance_extensions.as_ptr(),
            ..Default::default()
        };

        if validation_available {
            instance_create_info.enabled_layer_count = 1;
            instance_create_info.pp_enabled_layer_names = VALIDATION_LAYERS.as_ptr();
        }

        VULKAN_ENTRY
            .create_instance(&instance_create_info, None)
            .expect("failed to create vulkan instance")
    }
}
