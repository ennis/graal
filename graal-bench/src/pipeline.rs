use graal::vk;
use std::ptr;

static INPUT_ASSEMBLY_STATE_TRIANGLE_LIST: vk::PipelineInputAssemblyStateCreateInfo =
    vk::PipelineInputAssemblyStateCreateInfo {
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
        primitive_restart_enable: 0,
        ..Default::default()
    };

static VIEWPORT_STATE_SINGLE_VIEWPORT_SCISSOR: vk::PipelineViewportStateCreateInfo =
    vk::PipelineViewportStateCreateInfo {
        flags: Default::default(),
        viewport_count: 1,
        p_viewports: ptr::null(),
        scissor_count: 1,
        p_scissors: ptr::null(),
        ..Default::default()
    };

