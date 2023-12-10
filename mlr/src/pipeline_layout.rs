//! Strongly-typed pipeline layouts.
// I'd have preferred to not expose them to the user, but it was just too complicated to avoid it.

use crate::{argument::ArgumentsLayout, device::Device, vk};
use std::borrow::Cow;

pub trait StaticPipelineInterface {
    fn push_constants(&self) -> Cow<[vk::PushConstantRange]>;
    fn arguments(&self) -> Cow<[ArgumentsLayout]>;
}

pub struct PipelineLayout {
    device: graal::device::Device,
    pipeline_layout: vk::PipelineLayout,
    set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Drop for PipelineLayout {
    fn drop(&mut self) {
        unsafe {
            self.device.delete_later(self.pipeline_layout);
            for set_layout in &self.set_layouts {
                self.device.delete_later(*set_layout);
            }
        }
    }
}

impl PipelineLayout {
    pub fn new(device: &Device) -> PipelineLayout {}

    pub fn pipeline_layout(&self) -> vk::PipelineLayout {
        self.pipeline_layout
    }
}

/*
macro_rules! pipeline_layouts {
    (@push_constants ) => { &[] }
    (@push_constants $push_cst_ty:ty) => { <$push_cst_ty as $crate::argument::StaticPushConstants>::PUSH_CONSTANT_RANGES };
    ($($v:vis struct $name:ident (arguments = ($($arg_ty:ty),*) $(, push_constants = $push_cst_ty:ty)?);)*) => {
        $(
            #[derive(Debug)]
            $v struct $name($crate::pipeline_layout::PipelineLayoutInner);

            impl $name {
                pub fn new(device: &$crate::device::Device) -> $name {
                    $name(
                        $crate::pipeline_layout::PipelineLayoutInner::new(
                            device.device(),
                            &[
                                $(&<$arg_ty as $crate::argument::StaticArguments>::LAYOUT,)*
                            ],
                            pipeline_layouts!(@push_constants $($push_cst_ty)?),
                        )
                    )
                }
            }
        )*
    };
}*/

/*
pipeline_layouts! {
    struct EmptyPipelineLayout(arguments = (Arg1, Arg2), push_constants = PushCst);
}

struct Shaders {
    background_layout: EmptyPipelineLayout,
}
*/
