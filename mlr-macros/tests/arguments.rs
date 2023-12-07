use mlr::{
    argument::{Arguments, PushConstants, SampledImage, StaticArguments, StaticPushConstants, UniformBuffer},
    vk,
};
use std::mem;

#[repr(C)]
struct InlineData {
    u_color: [f32; 4],
    u_matrix: [[f32; 4]; 4],
}

#[derive(Arguments)]
struct ArgumentsInlineOnly {
    u_color: [f32; 4],
    u_matrix: [[f32; 4]; 4],
}

#[derive(Arguments)]
struct ArgumentsNoInline {
    #[argument(binding = 0)]
    u0: UniformBuffer<u64>,
    #[argument(binding = 1)]
    t1: SampledImage,
}

#[derive(Arguments)]
struct ArgumentsNoInlineBindGaps {
    #[argument(binding = 1)]
    u0: UniformBuffer<u64>,
    #[argument(binding = 5)]
    t1: SampledImage,
}

#[derive(Arguments)]
struct ArgumentsMixed {
    u_color: [f32; 4],
    u_matrix: [[f32; 4]; 4],
    #[argument(binding = 1)]
    u0: UniformBuffer<u64>,
    #[argument(binding = 2)]
    t1: SampledImage,
}

// Should be invalid:

/*
#[derive(Arguments)]
struct ArgumentsMixedInvalid {
    u_color: [f32; 4],
    u_matrix: [[f32; 4]; 4],
    #[argument(binding = 0)] // already occupied by inline data
    u0: UniformBuffer<u64>,
    #[argument(binding = 1)]
    t1: SampledImage,
}*/

#[derive(Arguments)]
struct ArgumentsTupleInlineOnly([f32; 4], [[f32; 4]; 4]);

#[derive(Arguments)]
struct ArgumentsTupleNoInline(
    #[argument(binding = 1)] UniformBuffer<u64>,
    #[argument(binding = 2)] SampledImage,
);

#[derive(Arguments)]
struct ArgumentsTupleMixed(
    [f32; 4],
    [[f32; 4]; 4],
    #[argument(binding = 1)] UniformBuffer<u64>,
    #[argument(binding = 2)] SampledImage,
);

#[derive(PushConstants)]
#[repr(C)]
struct PushCst {
    #[stages(vertex)]
    u_matrix: [[f32; 4]; 4],
    #[stages(fragment)]
    u_color: [f32; 4],
    // all stages
    u_time: f32,
}

#[test]
#[rustfmt::skip]
fn test_push_constants() {
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[0].stage_flags, vk::ShaderStageFlags::VERTEX);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[0].offset, 0);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[0].size, 64);

    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[1].stage_flags, vk::ShaderStageFlags::FRAGMENT);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[1].offset, 64);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[1].size, mem::size_of::<[f32; 4]>() as u32);

    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[2].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[2].offset, mem::size_of::<[[f32; 4]; 4]>() as u32 + mem::size_of::<[f32; 4]>() as u32);
    assert_eq!(<PushCst as StaticPushConstants>::PUSH_CONSTANT_RANGES[2].size, mem::size_of::<f32>() as u32);
}

#[test]
#[rustfmt::skip]
fn test_arguments() {
    
    assert_eq!(<ArgumentsInlineOnly as StaticArguments>::LAYOUT.bindings[0].binding, 0);
    assert_eq!(<ArgumentsInlineOnly as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsInlineOnly as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::INLINE_UNIFORM_BLOCK);
    assert_eq!(<ArgumentsInlineOnly as StaticArguments>::LAYOUT.bindings[0].descriptor_count, mem::size_of::<InlineData>() as u32);
    assert_eq!(<ArgumentsInlineOnly as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());
    
    //----------------------------------------
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[0].binding, 0);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[0].descriptor_count, 1);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());
    
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[1].binding, 1);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[1].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[1].descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[1].descriptor_count, 1);
    assert_eq!(<ArgumentsNoInline as StaticArguments>::LAYOUT.bindings[1].p_immutable_samplers, std::ptr::null());
    
    //----------------------------------------
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[0].binding, 1);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[0].descriptor_count, 1);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());

    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[1].binding, 5);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[1].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[1].descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[1].descriptor_count, 1);
    assert_eq!(<ArgumentsNoInlineBindGaps as StaticArguments>::LAYOUT.bindings[1].p_immutable_samplers, std::ptr::null());

    //----------------------------------------
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[0].binding, 0);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::INLINE_UNIFORM_BLOCK);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[0].descriptor_count, mem::size_of::<InlineData>() as u32);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());
    
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[1].binding, 1);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[1].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[1].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[1].descriptor_count, 1);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[1].p_immutable_samplers, std::ptr::null());

    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[2].binding, 2);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[2].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[2].descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[2].descriptor_count, 1);
    assert_eq!(<ArgumentsMixed as StaticArguments>::LAYOUT.bindings[2].p_immutable_samplers, std::ptr::null());

    //----------------------------------------
    assert_eq!(<ArgumentsTupleInlineOnly as StaticArguments>::LAYOUT.bindings[0].binding, 0);
    assert_eq!(<ArgumentsTupleInlineOnly as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsTupleInlineOnly as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::INLINE_UNIFORM_BLOCK);
    assert_eq!(<ArgumentsTupleInlineOnly as StaticArguments>::LAYOUT.bindings[0].descriptor_count, mem::size_of::<InlineData>() as u32);
    assert_eq!(<ArgumentsTupleInlineOnly as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());

    //----------------------------------------
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[0].binding, 0);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[0].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[0].descriptor_type, vk::DescriptorType::INLINE_UNIFORM_BLOCK);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[0].descriptor_count, mem::size_of::<InlineData>() as u32);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[0].p_immutable_samplers, std::ptr::null());
    
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[1].binding, 1);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[1].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[1].descriptor_type, vk::DescriptorType::UNIFORM_BUFFER);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[1].descriptor_count, 1);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[1].p_immutable_samplers, std::ptr::null());

    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[2].binding, 2);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[2].stage_flags, vk::ShaderStageFlags::ALL);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[2].descriptor_type, vk::DescriptorType::SAMPLED_IMAGE);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[2].descriptor_count, 1);
    assert_eq!(<ArgumentsTupleMixed as StaticArguments>::LAYOUT.bindings[2].p_immutable_samplers, std::ptr::null());
}

/*
#[pipeline_layout]
#[push_constants(PushCst)]
#[arguments(0, ArgumentsInlineOnly)]
struct DefaultPipeline;
*/
