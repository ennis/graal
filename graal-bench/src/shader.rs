//! Shader utilities
use graal::{ash::version::DeviceV1_0, vk};

pub fn create_shader_module(context: &mut graal::Context, code: &[u32]) -> vk::ShaderModule {
    unsafe {
        context
            .vulkan_device()
            .create_shader_module(
                &vk::ShaderModuleCreateInfo {
                    flags: Default::default(),
                    code_size: code.len() * 4,
                    p_code: code.as_ptr(),
                    ..Default::default()
                },
                None,
            )
            .expect("failed to create shader module")
    }
}
