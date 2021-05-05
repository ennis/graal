use crate::context::PipelineLayoutId;
use ash::vk;
use std::marker::PhantomData;

pub trait PipelineInterface {
    const VERTEX_INPUT_BINDINGS: &'static [vk::VertexInputBindingDescription];
    const VERTEX_INPUT_ATTRIBUTES: &'static [vk::VertexInputAttributeDescription];
    fn get_or_init_pipeline_layout(init: impl FnOnce() -> PipelineLayoutId) -> PipelineLayoutId;
}

pub struct Pipeline {
    pub handle: vk::Pipeline,
}

pub struct TypedPipeline<T: PipelineInterface> {
    pub handle: vk::Pipeline,
    _phantom: PhantomData<*const T>,
}
