use ash::{
    version::InstanceV1_0,
    vk::{KhrExternalMemoryWin32Fn, KhrExternalSemaphoreWin32Fn},
};
use std::mem;

const PLATFORM_DEVICE_EXTENSIONS: &[&str] = &[
    "VK_KHR_external_memory_win32",
    "VK_KHR_external_semaphore_win32",
];

/// Windows-specific vulkan extensions
pub struct PlatformExtensions {
    pub khr_external_memory_win32: KhrExternalMemoryWin32Fn,
    pub khr_external_semaphore_win32: KhrExternalSemaphoreWin32Fn,
}

impl PlatformExtensions {
    pub(crate) fn names() -> &'static [&'static str] {
        PLATFORM_DEVICE_EXTENSIONS
    }

    pub(crate) fn load(
        _entry: &ash::Entry,
        instance: &ash::Instance,
        device: &ash::Device,
    ) -> PlatformExtensions {
        unsafe {
            let khr_external_memory_win32 = KhrExternalMemoryWin32Fn::load(|name| {
                mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
            });
            let khr_external_semaphore_win32 = KhrExternalSemaphoreWin32Fn::load(|name| {
                mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
            });

            PlatformExtensions {
                khr_external_memory_win32,
                khr_external_semaphore_win32,
            }
        }
    }
}
