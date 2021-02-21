use graal::vk;
use graal::DescriptorSetInterface;
use graal_macros::DescriptorSetInterface;

#[derive(Copy,Clone,Debug)]
#[repr(C)]
struct PerObjectData {
    resolution: [f32;2],
    scroll_offset: [f32;2],
    zoom: f32,
}

#[derive(DescriptorSetInterface)]
struct GlobalResources<'a> {
    #[layout(binding=0, sampled_image, array, unbounded, max_count=1024)]
    textures: &'a [vk::DescriptorImageInfo],
}

#[derive(DescriptorSetInterface)]
struct PerObjectResources {
    #[layout(binding=0, uniform_buffer, stages(all_graphics))]
    uniforms: vk::DescriptorBufferInfo,
    #[layout(binding=1, uniform_buffer)]
    buffer: vk::DescriptorBufferInfo
}

#[test]
fn test_descriptor()
{
    eprintln!("GlobalResources: {:#?}", GlobalResources::LAYOUT);
    eprintln!("PerObjectResources: {:#?}", PerObjectResources::LAYOUT);
}