#[cfg(windows)]
mod platform {
    use crate::instance::{get_vulkan_entry, get_vulkan_instance};
    use ash::{extensions::khr::Win32Surface, vk};
    use once_cell::sync::Lazy;
    use raw_window_handle::RawWindowHandle;
    use std::os::raw::c_void;

    static VK_KHR_SURFACE_WIN32: Lazy<Win32Surface> = Lazy::new(create_vk_khr_surface);

    fn create_vk_khr_surface() -> Win32Surface {
        ash::extensions::khr::Win32Surface::new(get_vulkan_entry(), get_vulkan_instance())
    }

    pub fn get_vulkan_surface(handle: RawWindowHandle) -> vk::SurfaceKHR {
        let win32_handle = match handle {
            RawWindowHandle::Win32(h) => h,
            _ => panic!("incompatible window handle"),
        };

        let create_info = vk::Win32SurfaceCreateInfoKHR {
            flags: Default::default(),
            hinstance: win32_handle.hinstance.unwrap().get() as *const c_void,
            hwnd: win32_handle.hwnd.get() as *const c_void,
            ..Default::default()
        };
        unsafe {
            VK_KHR_SURFACE_WIN32
                .create_win32_surface(&create_info, None)
                .expect("failed to create win32 surface")
        }
    }
}

pub use self::platform::get_vulkan_surface;
