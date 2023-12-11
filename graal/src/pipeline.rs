use std::{borrow::Cow, path::Path};

use crate::{
    argument::ArgumentsLayout,
    device::Device,
    vertex::{VertexBufferLayoutDescription, VertexInputAttributeDescription},
    vk,
};

pub trait StaticPipelineInterface {
    fn push_constants(&self) -> Cow<[vk::PushConstantRange]>;
    fn arguments(&self) -> Cow<[ArgumentsLayout]>;
}

/// Specifies the type of a graphics pipeline with the given vertex input, arguments and push constants.
#[macro_export]
macro_rules! graphics_pipeline_interface {
    [
        $(#[$meta:meta])*
        $v:vis struct $name:ident {
            arguments {
                $(
                    $(#[$arg_meta:meta])*
                    $arg_binding:literal => $arg_method:ident($arg_ty:ty)
                ),*
            }

            $(push_constants {
                $(#[$push_cst_meta:meta])*
                $push_constants_method:ident($push_constants_ty:ty)
            })?

            $(output_attachments($attachments_ty:ty))?
        }
    ] => {
        $(#[$meta])*
        $v struct $name<'a> {
            p: crate::encoder::RenderEncoder<'a>,
        }

        impl<'a> $name<'a> {
            $(
                $(#[$arg_meta])*
                pub fn $arg_method(&mut self, arg: &$arg_ty) {
                    unsafe {
                        self.p.bind_arguments($arg_binding, &arg)
                    }
                }
            )*

            $(
                $(#[$push_cst_meta])*
                pub fn $push_constants_method(&mut self, push_constants: &$push_constants_ty) {
                    unsafe {
                        self.p.bind_push_constants(push_constants)
                    }
                }
            )?
        }

        impl<'a> crate::device::StaticPipelineInterface for $name<'a> {
            const ARGUMENTS: &'static [&'static crate::argument::ArgumentsLayout<'static>] = &[
                $(
                    &$arg_ty::LAYOUT
                ),*
            ];

            const PUSH_CONSTANTS: &'static [crate::vk::PushConstantRange] = graphics_pipeline_interface!(@push_constants $($push_constants_ty)?);

            type Attachments = graphics_pipeline_interface!(@attachments $($attachments_ty)?);
        }
    };

    //------------------------------------------------

    // No push constants -> empty constant ranges
    (@push_constants ) => { &[] };
    (@push_constants $t:ty) => { <$t as $crate::argument::StaticPushConstants>::PUSH_CONSTANT_RANGES };

    // Default attachment type if left unspecified
    (@attachments ) => { () };
    (@attachments $t:ty) => { $t };
}

/// Describes the interface of a pipeline.
#[derive(Clone, Debug)]
pub struct PipelineInterfaceDescriptor<'a> {
    pub arguments: Cow<'a, [ArgumentsLayout<'a>]>,
    pub push_constants: Cow<'a, [vk::PushConstantRange]>,
    pub vertex_buffers: Cow<'a, [VertexBufferLayoutDescription]>,
    pub vertex_attributes: Cow<'a, [VertexInputAttributeDescription]>,
    pub color_attachment_formats: Cow<'a, [vk::Format]>,
    pub depth_attachment_format: Option<vk::Format>,
    pub stencil_attachment_format: Option<vk::Format>,
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    Content(&'a str),
    File(&'a Path),
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    TessControl,
    TessEvaluation,
    Mesh,
    Task,
}

impl ShaderKind {
    fn to_vk_shader_stage(&self) -> vk::ShaderStageFlags {
        match self {
            ShaderKind::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderKind::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderKind::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderKind::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderKind::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderKind::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ShaderKind::Mesh => vk::ShaderStageFlags::MESH_NV,
            ShaderKind::Task => vk::ShaderStageFlags::TASK_NV,
        }
    }
}

pub struct ShaderCreateInfo<'a> {
    pub kind: ShaderKind,
    pub source: ShaderSource<'a>,
    pub entry_point: &'a str,
    pub layouts: &'a [&'a [vk::DescriptorSetLayoutBinding]],
}

/// Shader code + entry point
#[derive(Debug, Clone, Copy)]
pub struct ShaderEntryPoint<'a> {
    pub code: ShaderCode<'a>,
    pub entry_point: &'a str,
}

/// Specifies the code of a shader.
#[derive(Debug, Clone, Copy)]
pub enum ShaderCode<'a> {
    /// Compile the shader from the specified source.
    Source(ShaderSource<'a>),
    /// Create the shader from the specified SPIR-V binary.
    Spirv(&'a [u32]),
}

/// Specifies the shaders of a graphics pipeline.
pub enum GraphicsShaders {
    /// Shaders of the primitive shading pipeline (the classic vertex, tessellation, geometry and fragment shaders).
    PrimitiveShading {
        vertex: ShaderEntryPoint<'static>,
        tess_control: Option<ShaderEntryPoint<'static>>,
        tess_evaluation: Option<ShaderEntryPoint<'static>>,
        geometry: Option<ShaderEntryPoint<'static>>,
        fragment: ShaderEntryPoint<'static>,
    },
    /// Shaders of the mesh shading pipeline (the new mesh and task shaders).
    MeshShading {
        mesh: ShaderEntryPoint<'static>,
        task: ShaderEntryPoint<'static>,
        fragment: ShaderEntryPoint<'static>,
    },
}

pub struct GraphicsPipelineCreateInfo {
    pub shaders: GraphicsShaders,
    pub tessellation: Option<vk::PipelineTessellationStateCreateInfo>,
    pub rasterization: vk::PipelineRasterizationStateCreateInfo,
    pub multisample: vk::PipelineMultisampleStateCreateInfo,
    pub depth_stencil: Option<vk::PipelineDepthStencilStateCreateInfo>,
    pub color_blend: vk::PipelineColorBlendStateCreateInfo,
}

pub struct GraphicsPipeline {
    device: Device,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub(super) fn new(device: Device, pipeline: vk::Pipeline, pipeline_layout: vk::PipelineLayout) -> Self {
        Self {
            device,
            pipeline,
            pipeline_layout,
        }
    }

    pub fn pipeline(&self) -> vk::Pipeline {
        self.pipeline
    }
}
