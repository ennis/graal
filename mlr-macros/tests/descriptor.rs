use std::marker::PhantomData;
use std::mem;
use mlr::Arguments;

/*#[derive(Copy, Clone, Debug)]
#[derive(mlr::ShaderArguments)]
#[repr(C)]
struct PerObjectData {
    resolution: [f32; 2],
    scroll_offset: [f32; 2],
    zoom: f32,
}*/

#[derive(mlr::Arguments)]
#[repr(C)]
#[derive(mlr::StructLayout)]
struct MaterialArguments<'a> {
    u_color: [f32;4],
    #[argument(binding=1)] t_color: mlr::SampledImage2D<'a>,
}

#[repr(C)]
struct PerObjectData {
    resolution: [f32; 2],
    scroll_offset: [f32; 2],
    zoom: f32,
}

/*impl mlr::ShaderArguments for PerObjectData {
    fn unique_type_id(&self) -> Option<::std::any::TypeId> {
        Some(::std::any::TypeId::of::<Self>())
    }
    fn get_descriptor_set_layout_bindings(&self) -> &[mlr::vk::DescriptorSetLayoutBinding] {
        const BINDINGS: &'static [mlr::vk::DescriptorSetLayoutBinding] = &[
            mlr::vk::DescriptorSetLayoutBinding {
                binding: 0u32,
                stage_flags: <[f32; 2] as mlr::DescriptorBinding>::STAGE_FLAGS,
                descriptor_type: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                descriptor_count: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                p_immutable_samplers: ::std::ptr::null(),
            },
            mlr::vk::DescriptorSetLayoutBinding {
                binding: 0u32,
                stage_flags: <[f32; 2] as mlr::DescriptorBinding>::STAGE_FLAGS,
                descriptor_type: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                descriptor_count: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                p_immutable_samplers: ::std::ptr::null(),
            },
            mlr::vk::DescriptorSetLayoutBinding {
                binding: 0u32,
                stage_flags: <f32 as mlr::DescriptorBinding>::STAGE_FLAGS,
                descriptor_type: <f32 as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                descriptor_count: <f32 as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                p_immutable_samplers: ::std::ptr::null(),
            },
        ];
        BINDINGS
    }
    fn get_descriptor_set_update_template_entries(
        &self,
    ) -> Option<&[mlr::vk::DescriptorUpdateTemplateEntry]> {
        const UPDATE_TEMPLATE_ENTRIES: &'static [mlr::vk::DescriptorUpdateTemplateEntry] = &[
            mlr::vk::DescriptorUpdateTemplateEntry {
                dst_binding: 0u32,
                dst_array_element: 0,
                descriptor_count: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                descriptor_type: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                offset: <[f32; 2] as mlr::DescriptorBinding>::UPDATE_OFFSET,
                stride: <[f32; 2] as mlr::DescriptorBinding>::UPDATE_STRIDE,
            },
            mlr::vk::DescriptorUpdateTemplateEntry {
                dst_binding: 0u32,
                dst_array_element: 0,
                descriptor_count: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                descriptor_type: <[f32; 2] as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                offset: <[f32; 2] as mlr::DescriptorBinding>::UPDATE_OFFSET,
                stride: <[f32; 2] as mlr::DescriptorBinding>::UPDATE_STRIDE,
            },
            mlr::vk::DescriptorUpdateTemplateEntry {
                dst_binding: 0u32,
                dst_array_element: 0,
                descriptor_count: <f32 as mlr::DescriptorBinding>::DESCRIPTOR_COUNT,
                descriptor_type: <f32 as mlr::DescriptorBinding>::DESCRIPTOR_TYPE,
                offset: <f32 as mlr::DescriptorBinding>::UPDATE_OFFSET,
                stride: <f32 as mlr::DescriptorBinding>::UPDATE_STRIDE,
            },
        ];
        Some(UPDATE_TEMPLATE_ENTRIES)
    }
    unsafe fn update_descriptor_set(
        &mut self,
        ctx: &mut mlr::RecordingContext,
        set: mlr::vk::DescriptorSet,
        update_template: Option<mlr::vk::DescriptorUpdateTemplate>,
    ) {
        ()
    }
}*/

#[test]
fn test_descriptor() {
    //eprintln!("GlobalResources: {:#?}", GlobalResources::DESC);
    //eprintln!("PerObjectResources: {:#?}", PerObjectResources::DESC);
}
