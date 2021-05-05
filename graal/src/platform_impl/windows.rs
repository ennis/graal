use crate::{vk, Device, ExternalMemoryHandle};
use ash::{version::DeviceV1_0, vk::KhrExternalMemoryWin32Fn};
use std::{ffi::OsStr, mem};
use crate::ash::version::InstanceV1_0;
use std::ffi::c_void;

/// Windows-specific vulkan extensions
pub struct PlatformExtensions {
    pub khr_external_memory_win32: KhrExternalMemoryWin32Fn,
}

impl PlatformExtensions {
    pub(crate) fn load(_entry: &ash::Entry, instance: &ash::Instance, device: &ash::Device) -> PlatformExtensions {
        unsafe {
            let khr_external_memory_win32 = KhrExternalMemoryWin32Fn::load(|name| {
                mem::transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr()))
            });

            PlatformExtensions {
                khr_external_memory_win32,
            }
        }
    }
}

pub(crate) unsafe fn allocate_external_memory(
    device: &Device,
    memory_requirements: &vk::MemoryRequirements,
    required_flags: vk::MemoryPropertyFlags,
    preferred_flags: vk::MemoryPropertyFlags,
    external_memory_handle: &ExternalMemoryHandle,
) -> vk::DeviceMemory {

    use std::os::windows::ffi::OsStrExt;

    let vk_device = &device.device;

    let (win32_handle, handle_type) = match external_memory_handle {
        ExternalMemoryHandle::D3D11Texture(h) => {
            (*h, vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE)
        }
        ExternalMemoryHandle::D3D11TextureKMT(h) => {
            (*h, vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT)
        }
    };

    let mut win32_handle_properties = vk::MemoryWin32HandlePropertiesKHR::default();
    let result = device
        .platform_extensions
        .khr_external_memory_win32
        .get_memory_win32_handle_properties_khr(
            vk_device.handle(),
            handle_type,
            win32_handle.handle,
            &mut win32_handle_properties,
        );
    assert_eq!(result, vk::Result::SUCCESS);

    // find a memory type that both matches the resource requirement and the external handle requirements for importing
    let memory_type_bits =
        memory_requirements.memory_type_bits & win32_handle_properties.memory_type_bits;
    let memory_type_index = device
        .find_compatible_memory_type(
            memory_type_bits,
            required_flags,
            preferred_flags,
        )
        .expect("could not find a compatible memory type for importing external memory");

    // import memory
    let name_wstr: Vec<u16> = OsStr::new(win32_handle.name).encode_wide().collect();

    let import_memory_win32_handle_info = vk::ImportMemoryWin32HandleInfoKHR {
        handle_type,
        handle: win32_handle.handle,
        name: name_wstr.as_ptr(),
        ..Default::default()
    };

    let memory_allocate_info = vk::MemoryAllocateInfo {
        p_next: &import_memory_win32_handle_info as *const _ as *const c_void,
        allocation_size: memory_requirements.size,
        memory_type_index,
        ..Default::default()
    };

    let device_memory = vk_device
        .allocate_memory(&memory_allocate_info, None)
        .unwrap();

    device_memory
}
