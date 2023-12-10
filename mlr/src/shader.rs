use crate::{
    argument::{Arguments, PushConstants, StaticArguments, StaticPushConstants},
    attachments::{Attachments, StaticAttachments},
    device::{Device, PipelineInterface, StaticPipelineInterface},
    encoder::RenderEncoder,
    pipeline_layout::PipelineLayout,
    vertex::StaticVertexInput,
    vk,
};
use once_cell::sync::Lazy;
use std::{
    borrow::Cow,
    ffi::{c_char, c_void, CString},
    path::{Path, PathBuf},
    ptr,
};

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

/*
/// Represents a shader.
pub struct Shader {
    device: graal::device::Device,
    shader: vk::ShaderEXT,
    descriptor_set_layouts: Vec<vk::DescriptorSetLayout>,
}

impl Drop for Shader {
    fn drop(&mut self) {
        self.device.delete_later(self.shader);
        unsafe {
            for &layout in self.descriptor_set_layouts.iter() {
                self.device.delete_later(layout);
            }
        }
    }
}*/
/*
fn create_shaders(device: &Device, create_infos: &[ShaderCreateInfo]) -> Result<Vec<Shader>, ShaderCreationError> {
    let mut shader_create_infos: Vec<vk::ShaderCreateInfoEXT> = Vec::with_capacity(create_infos.len());

    for (i, info) in create_infos.iter().enumerate() {
        let code = compile_shader(info)?;
        let entry_point = CString::new(info.entry_point).expect("entry point name contained null bytes");

        /* shader_create_infos.push(vk::ShaderCreateInfoEXT {
            flags: Default::default(),
            stage: info.kind.to_vk_shader_stage(),
            next_stage: vk::ShaderStageFlags::empty(),
            code_type: vk::ShaderCodeTypeEXT::SPIRV,
            code_size: 0,
            p_code: code.as_ptr() as *const c_void,
            p_name: entry_point.as_ptr(),
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_specialization_info: ptr::null(),
            ..Default::default()
        });*/

        /*shader_create_info[i] = vk::ShaderCreateInfoEXT {
            flags: Default::default(),
            stage: info.kind.to_vk_shader_stage(),
            next_stage: vk::ShaderStageFlags::empty(),
            code_type: vk::ShaderCodeTypeEXT::SPIRV,
            code_size: 0,
            p_code: code.as_ptr() as *const c_void,
            p_name: entry_point.as_ptr(),
            set_layout_count: info.layouts[i].len() as u32,
            p_set_layouts: info.layouts[i].as_ptr(),
            push_constant_range_count: 0,
            p_push_constant_ranges: ptr::null(),
            p_specialization_info: ptr::null(),
            ..Default::default()
        };*/
        // shader_create_info[i].
    }

    todo!()
    //    Ok(())
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Shader code + entry point
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
    device: graal::device::Device,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl GraphicsPipeline {
    pub(super) fn new(
        device: graal::device::Device,
        pipeline: vk::Pipeline,
        pipeline_layout: vk::PipelineLayout,
    ) -> Self {
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
