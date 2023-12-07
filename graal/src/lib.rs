//#![feature(maybe_uninit_slice)]

pub use ash::{self, vk};
pub use gpu_allocator::MemoryLocation;

pub mod device;
pub mod instance;
pub mod platform;
pub mod queue;
pub mod surface;
pub mod utils;

mod command_allocator;
mod platform_impl;

////////////////////////////////////////////////////////////////////////////////////////////////////

fn is_write_access(mask: vk::AccessFlags2) -> bool {
    // TODO: this is not exhaustive
    mask.intersects(
        vk::AccessFlags2::SHADER_WRITE
            | vk::AccessFlags2::COLOR_ATTACHMENT_WRITE
            | vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE
            | vk::AccessFlags2::TRANSFER_WRITE
            | vk::AccessFlags2::HOST_WRITE
            | vk::AccessFlags2::MEMORY_WRITE
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_WRITE_EXT
            | vk::AccessFlags2::TRANSFORM_FEEDBACK_COUNTER_WRITE_EXT
            | vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR
            | vk::AccessFlags2::COMMAND_PREPROCESS_WRITE_NV,
    )
}

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Computes the number of mip levels for a 2D image of the given size.
///
/// # Examples
///
/// ```
/// use graal::mip_level_count;
/// assert_eq!(mip_level_count(512, 512), 9);
/// assert_eq!(mip_level_count(512, 256), 9);
/// assert_eq!(mip_level_count(511, 256), 8);
/// ```
pub fn mip_level_count(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2().floor() as u32
}

pub fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

pub fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT
    )
}

pub fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, vk::Format::S8_UINT)
}

pub fn aspects_for_format(fmt: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_only_format(fmt) {
        vk::ImageAspectFlags::DEPTH
    } else if is_stencil_only_format(fmt) {
        vk::ImageAspectFlags::STENCIL
    } else if is_depth_and_stencil_format(fmt) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
}

pub use device::create_device_and_queue;
