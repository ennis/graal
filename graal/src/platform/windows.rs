use crate::{
    context::{get_vk_sample_count},
    vk,
    Context, Device, ImageInfo, ImageResourceCreateInfo, ResourceMemoryInfo,
};
use ash::{
    prelude::VkResult,
    version::DeviceV1_0,
    vk::{HANDLE, SECURITY_ATTRIBUTES},
};
use std::{
    ffi::{c_void, OsStr},
    ptr,
};

/*/// External memory handle types.
#[derive(Clone, Debug)]
pub enum Win32ExternalMemoryHandle<'a> {
    D3D11Texture(crate::platform::windows::Win32Handle<'a>),
    D3D11TextureKMT(crate::platform::windows::Win32Handle<'a>),
}

impl Win32ExternalMemoryHandle {
    pub(crate) fn handle_type(&self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            Win32ExternalMemoryHandle::D3D11Texture(_) => {
                vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE
            }
            Win32ExternalMemoryHandle::D3D11TextureKMT(_) => {
                vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT
            }
            _ => unreachable!(), // or is it?
        }
    }
}*/

/*#[derive(Copy, Clone, Debug)]
pub struct Win32Handle<'a> {
    pub handle: HANDLE,
    pub name: &'a str,
}*/

fn handle_name_to_wstr(name: Option<&str>) -> (Vec<u16>, *const u16) {
    use std::os::windows::ffi::OsStrExt;

    if let Some(name) = name {
        let mut w_name: Vec<u16> = OsStr::new(name).encode_wide().collect();
        w_name.push(0);
        let ptr = w_name.as_ptr();
        (w_name, ptr)
    } else {
        (Vec::new(), ptr::null())
    }
}

pub(crate) unsafe fn import_external_memory(
    device: &Device,
    memory_requirements: &vk::MemoryRequirements,
    required_flags: vk::MemoryPropertyFlags,
    preferred_flags: vk::MemoryPropertyFlags,
    handle_type: vk::ExternalMemoryHandleTypeFlags,
    handle: HANDLE,
    handle_name: Option<&str>,
) -> vk::DeviceMemory {
    let vk_device = &device.device;

    let mut win32_handle_properties = vk::MemoryWin32HandlePropertiesKHR::default();

    VkResult::from(
        device
            .platform_extensions
            .khr_external_memory_win32
            .get_memory_win32_handle_properties_khr(
                vk_device.handle(),
                handle_type,
                handle,
                &mut win32_handle_properties,
            ),
    )
    .unwrap();

    // find a memory type that both matches the resource requirement and the external handle requirements for importing
    let memory_type_bits =
        memory_requirements.memory_type_bits & win32_handle_properties.memory_type_bits;
    let memory_type_index = device
        .find_compatible_memory_type(memory_type_bits, required_flags, preferred_flags)
        .expect("could not find a compatible memory type for importing external memory");

    // import memory
    let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

    let import_memory_win32_handle_info = vk::ImportMemoryWin32HandleInfoKHR {
        handle_type,
        handle,
        name: handle_name_wstr,
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

pub trait ContextExtWindows {
    unsafe fn create_imported_image_win32(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> ImageInfo;

    unsafe fn create_exported_image_win32(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (ImageInfo, HANDLE);

    /// Creates a semaphore and exports it to a Windows handle of the specified type.
    /// The returned semaphore should be deleted with `vkDestroySemaphore`.
    unsafe fn create_exported_semaphore_win32(
        &mut self,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (vk::Semaphore, HANDLE);

    unsafe fn create_imported_semaphore_win32(
        &mut self,
        import_flags: vk::SemaphoreImportFlags,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> vk::Semaphore;
}

impl ContextExtWindows for Context {
    unsafe fn create_imported_image_win32(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        win32_handle_type: vk::ExternalMemoryHandleTypeFlags,
        win32_handle: HANDLE,
        win32_handle_name: Option<&str>,
    ) -> ImageInfo {
        let vk_device = self.vulkan_device();
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.image_type,
            format: image_info.format,
            extent: image_info.extent,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: image_info.tiling,
            usage: image_info.usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle =
            vk_device
                .create_image(&create_info, None)
                .expect("failed to create image");
        let mem_req = vk_device.get_image_memory_requirements(handle);
        let device_memory = import_external_memory(
            &self.device,
            &mem_req,
            memory_info.required_flags,
            memory_info.preferred_flags,
            win32_handle_type,
            win32_handle,
            win32_handle_name,
        );
        let memory = ResourceMemory::External { device_memory };
        let id = self.register_image_resource(name, handle, image_info.format, memory, false);

        ImageInfo { id, handle }
    }

    unsafe fn create_exported_image_win32(
        &mut self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_info: &ImageResourceCreateInfo,
        handle_type: vk::ExternalMemoryHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (ImageInfo, HANDLE) {
        let vk_device = self.vulkan_device();
        let external_memory_image_create_info = vk::ExternalMemoryImageCreateInfo {
            handle_types: handle_type,
            ..Default::default()
        };
        let create_info = vk::ImageCreateInfo {
            p_next: &external_memory_image_create_info as *const _ as *const c_void,
            image_type: image_info.image_type,
            format: image_info.format,
            extent: image_info.extent,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: image_info.tiling,
            usage: image_info.usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: self.device.queues_info.queue_count as u32,
            p_queue_family_indices: self.device.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle =
            vk_device
                .create_image(&create_info, None)
                .expect("failed to create image");
        let mem_req = vk_device.get_image_memory_requirements(handle);

        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        let memory_type_index = self
            .device
            .find_compatible_memory_type(
                mem_req.memory_type_bits,
                memory_info.required_flags,
                memory_info.preferred_flags,
            )
            .expect("could not find a compatible memory type for exporting memory");

        let win32_handle_info = vk::ExportMemoryWin32HandleInfoKHR {
            p_attributes: security_attributes,
            dw_access: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_memory_allocate_info = vk::ExportMemoryAllocateInfo {
            p_next: &win32_handle_info as *const _ as *const c_void,
            handle_types: handle_type,
            ..Default::default()
        };
        let memory_allocate_info = vk::MemoryAllocateInfo {
            p_next: &export_memory_allocate_info as *const _ as *const c_void,
            allocation_size: mem_req.size,
            memory_type_index,
            ..Default::default()
        };

        let device_memory = vk_device
            .allocate_memory(&memory_allocate_info, None)
            .expect("failed to allocate exported memory");

        // retrieve the win32 handle
        let get_win32_handle_info = vk::MemoryGetWin32HandleInfoKHR {
            memory: device_memory,
            handle_type,
            ..Default::default()
        };

        let mut win32_handle = ptr::null_mut();

        VkResult::from(
            self.device
                .platform_extensions
                .khr_external_memory_win32
                .get_memory_win32_handle_khr(
                    vk_device.handle(),
                    &get_win32_handle_info,
                    &mut win32_handle,
                ),
        )
        .unwrap();

        // bind memory
        self.device
            .device
            .bind_image_memory(handle, device_memory, 0)
            .unwrap();

        let id = self.register_image_resource(
            name,
            handle,
            image_info.format,
            ResourceMemory::External { device_memory },
            false,
        );

        let image_info = ImageInfo { id, handle };

        (image_info, win32_handle)
    }

    unsafe fn create_exported_semaphore_win32(
        &mut self,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        security_attributes: *const SECURITY_ATTRIBUTES,
        access_flags: u32,
        handle_name: Option<&str>,
    ) -> (vk::Semaphore, HANDLE) {
        let vk_device = self.vulkan_device();
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        let export_semaphore_win32_handle_info = vk::ExportSemaphoreWin32HandleInfoKHR {
            p_attributes: security_attributes,
            dw_access: access_flags,
            name: handle_name_wstr,
            ..Default::default()
        };
        let export_semaphore_create_info = vk::ExportSemaphoreCreateInfo {
            p_next: &export_semaphore_win32_handle_info as *const _ as *const c_void,
            handle_types: handle_type,
            ..Default::default()
        };
        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: &export_semaphore_create_info as *const _ as *const c_void,
            ..Default::default()
        };

        let semaphore = vk_device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();

        let mut handle = ptr::null_mut();
        let get_win32_handle_info = vk::SemaphoreGetWin32HandleInfoKHR {
            semaphore,
            handle_type,
            ..Default::default()
        };
        VkResult::from(
            self.device
                .platform_extensions
                .khr_external_semaphore_win32
                .get_semaphore_win32_handle_khr(
                    vk_device.handle(),
                    &get_win32_handle_info,
                    &mut handle,
                ),
        )
        .unwrap();

        (semaphore, handle)
    }

    unsafe fn create_imported_semaphore_win32(
        &mut self,
        import_flags: vk::SemaphoreImportFlags,
        handle_type: vk::ExternalSemaphoreHandleTypeFlags,
        handle: HANDLE,
        handle_name: Option<&str>,
    ) -> vk::Semaphore {
        let vk_device = self.vulkan_device();
        let (_, handle_name_wstr) = handle_name_to_wstr(handle_name);

        // create the semaphore
        let is_timeline = match handle_type {
            vk::ExternalSemaphoreHandleTypeFlags::D3D12_FENCE => true,
            _ => panic!("unsupported external semaphore type"),
        };

        let timeline_create_info = vk::SemaphoreTypeCreateInfo {
            semaphore_type: vk::SemaphoreType::TIMELINE,
            initial_value: 0,
            ..Default::default()
        };

        let semaphore_create_info = vk::SemaphoreCreateInfo {
            p_next: if is_timeline {
                &timeline_create_info as *const _ as *const c_void
            } else {
                ptr::null()
            },
            ..Default::default()
        };

        let semaphore = vk_device
            .create_semaphore(&semaphore_create_info, None)
            .unwrap();

        let import_semaphore_win32_handle_info = vk::ImportSemaphoreWin32HandleInfoKHR {
            semaphore,
            flags: import_flags, // ?????
            handle_type,
            handle,
            name: handle_name_wstr,
            ..Default::default()
        };

        VkResult::from(
            self.device
                .platform_extensions
                .khr_external_semaphore_win32
                .import_semaphore_win32_handle_khr(
                    vk_device.handle(),
                    &import_semaphore_win32_handle_info,
                ),
        )
        .unwrap();
        semaphore
    }
}
