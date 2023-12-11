use crate::{device::DeviceCreateError, vk};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("failed to create device")]
    DeviceCreationFailed(#[from] DeviceCreateError),
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("compilation error")]
    Shaderc(#[from] shaderc::Error),
    #[error("Vulkan error")]
    Vulkan(#[from] vk::Result),
}
