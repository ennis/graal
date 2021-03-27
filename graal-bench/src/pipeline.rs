use graal::vk;

fn input_assembly_state_triangle_list() -> vk::PipelineInputAssemblyStateCreateInfo {
    vk::PipelineInputAssemblyStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
        p_next: ::std::ptr::null(),
        flags: vk::PipelineInputAssemblyStateCreateFlags::empty(),
        primitive_restart_enable: vk::FALSE,
        topology: vk::PrimitiveTopology::TRIANGLE_LIST,
    }
}

fn viewport_state_single_viewport_scissor() -> vk::PipelineViewportStateCreateInfo {
    vk::PipelineViewportStateCreateInfo {
        s_type: vk::StructureType::PIPELINE_VIEWPORT_STATE_CREATE_INFO,
        p_next: ::std::ptr::null(),
        flags: vk::PipelineViewportStateCreateFlags::default(),
        viewport_count: 1,
        p_viewports: ::std::ptr::null(),
        scissor_count: 1,
        p_scissors: ::std::ptr::null(),
    }
}
