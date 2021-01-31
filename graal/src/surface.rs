#[cfg(windows)]
mod platform {
    use crate::VULKAN_ENTRY;
    use crate::VULKAN_INSTANCE;
    use ash::extensions::khr::Win32Surface;
    use ash::vk;
    use lazy_static::lazy_static;
    use raw_window_handle::RawWindowHandle;
    use std::os::raw::c_void;

    lazy_static! {
        pub(crate) static ref VULKAN_SURFACE_WIN32_KHR: Win32Surface =
            Win32Surface::new(&*VULKAN_ENTRY, &*VULKAN_INSTANCE);
    }

    pub fn get_vulkan_surface(handle: RawWindowHandle) -> vk::SurfaceKHR {
        let win32_handle = match handle {
            RawWindowHandle::Windows(h) => h,
            _ => panic!("incompatible window handle"),
        };

        let create_info = vk::Win32SurfaceCreateInfoKHR {
            flags: Default::default(),
            hinstance: win32_handle.hinstance as *const c_void,
            hwnd: win32_handle.hwnd as *const c_void,
            ..Default::default()
        };
        unsafe {
            VULKAN_SURFACE_WIN32_KHR
                .create_win32_surface(&create_info, None)
                .expect("failed to create win32 surface")
        }
    }
}

pub use self::platform::get_vulkan_surface;
