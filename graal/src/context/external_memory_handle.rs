use ash::vk;

/// External memory handle types.
#[derive(Copy, Clone, Debug)]
pub enum ExternalMemoryHandle<'a> {
    #[cfg(windows)]
    D3D11Texture(crate::platform::windows::Win32Handle<'a>),
    #[cfg(windows)]
    D3D11TextureKMT(crate::platform::windows::Win32Handle<'a>),
}

impl<'a> ExternalMemoryHandle<'a> {
    pub(crate) fn handle_type(&self) -> vk::ExternalMemoryHandleTypeFlags {
        match self {
            #[cfg(windows)] ExternalMemoryHandle::D3D11Texture(_) => vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE,
            #[cfg(windows)] ExternalMemoryHandle::D3D11TextureKMT(_) => vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE_KMT,
            _ => unreachable!() // or is it?
        }
    }
}

