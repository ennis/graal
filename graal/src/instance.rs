use ash::vk;
use core::ptr;
use lazy_static::lazy_static;
use std::{
    ffi::{CStr, CString},
    os::raw::c_char,
};

/// List of validation layers to enable
const VALIDATION_LAYERS: &[&str] = &[/*"VK_LAYER_KHRONOS_validation"*/];

lazy_static! {
    pub(crate) static ref VULKAN_ENTRY: ash::Entry = initialize_vulkan_entry();
    pub(crate) static ref VULKAN_INSTANCE: ash::Instance = create_vulkan_instance();
}

fn initialize_vulkan_entry() -> ash::Entry {
    unsafe { ash::Entry::new().expect("failed to initialize vulkan entry points") }
}

/// Checks if all validation layers are supported
unsafe fn check_validation_layer_support() -> bool {
    let available_layers = VULKAN_ENTRY
        .enumerate_instance_layer_properties()
        .expect("failed to enumerate instance layers");

    VALIDATION_LAYERS.iter().all(|&required_layer| {
        let c_required_layer = CString::new(required_layer).unwrap();
        available_layers
            .iter()
            .any(|&layer| CStr::from_ptr(layer.layer_name.as_ptr()) == c_required_layer.as_c_str())
    })
}

#[cfg(windows)]
const INSTANCE_EXTENSIONS: &[&str] = &[
    "VK_KHR_get_surface_capabilities2",
    "VK_EXT_debug_utils",
    "VK_KHR_surface",
    "VK_KHR_win32_surface",
];
// TODO other platforms

fn create_vulkan_instance() -> ash::Instance {
    unsafe {
        let validation_available = check_validation_layer_support();
        if !validation_available {
            eprintln!("validation layer not available");
        }

        // Convert instance extension strings into C-strings
        let c_instance_extensions: Vec<_> = INSTANCE_EXTENSIONS
            .iter()
            .map(|&s| CString::new(s).unwrap())
            .collect();

        let instance_extensions: Vec<_> =
            c_instance_extensions.iter().map(|s| s.as_ptr()).collect();

        // Convert validation layer names into C-strings
        let c_validation_layers: Vec<_> = VALIDATION_LAYERS
            .iter()
            .map(|&s| CString::new(s).unwrap())
            .collect();

        let validation_layers: Vec<_> = c_validation_layers.iter().map(|s| s.as_ptr()).collect();

        let application_info = vk::ApplicationInfo {
            // TODO let the user provide their own name here
            p_application_name: b"GRAAL\0".as_ptr() as *const c_char,
            application_version: 0,
            p_engine_name: b"GRAAL\0".as_ptr() as *const c_char,
            engine_version: 0,
            // require vulkan 1.2
            api_version: vk::make_api_version(0, 1, 2, 0),
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
            instance_create_info.enabled_layer_count = validation_layers.len() as u32;
            instance_create_info.pp_enabled_layer_names = validation_layers.as_ptr();
        }

        VULKAN_ENTRY
            .create_instance(&instance_create_info, None)
            .expect("failed to create vulkan instance")
    }
}

/// Returns the global `ash::Entry` object.
pub fn get_vulkan_entry() -> &'static ash::Entry {
    &VULKAN_ENTRY
}

/// Returns the global vulkan instance object.
pub fn get_vulkan_instance() -> &'static ash::Instance {
    &VULKAN_INSTANCE
}

/// Returns the list of instance extensions that the instance was created with.
pub fn get_instance_extensions() -> &'static [&'static str] {
    INSTANCE_EXTENSIONS
}
