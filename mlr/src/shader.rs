use crate::{
    argument::{Arguments, PushConstants, StaticArguments, StaticPushConstants},
    attachments::{Attachments, StaticAttachments},
    device::Device,
    pipeline_layout::PipelineLayout,
    vk,
};
use graal::ash;
use once_cell::sync::Lazy;
use std::{
    ffi::{c_void, CString},
    path::{Path, PathBuf},
    ptr,
};

pub trait HasArguments<A: Arguments> {}

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    Content(&'a str),
    File(&'a Path),
}

pub enum ShaderKind {
    Vertex,
    Fragment,
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

const COMPILER: Lazy<shaderc::Compiler> =
    Lazy::new(|| shaderc::Compiler::new().expect("failed to create shader compiler"));

#[derive(Debug, thiserror::Error)]
pub enum ShaderCreationError {
    #[error("I/O error")]
    Io(#[from] std::io::Error),
    #[error("compilation error")]
    Shaderc(#[from] shaderc::Error),
}

/// Compiles a shader to SPIR-V
fn compile_shader(create_info: &ShaderCreateInfo) -> Result<Vec<u32>, ShaderCreationError> {
    let source_from_path;
    let source_content = match create_info.source {
        ShaderSource::Content(str) => str,
        ShaderSource::File(path) => {
            source_from_path = std::fs::read_to_string(path)?;
            source_from_path.as_str()
        }
    };
    let kind = match create_info.kind {
        ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
        ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
        ShaderKind::Compute => shaderc::ShaderKind::Compute,
        ShaderKind::TessControl => shaderc::ShaderKind::TessControl,
        ShaderKind::TessEvaluation => shaderc::ShaderKind::TessEvaluation,
        ShaderKind::Mesh => shaderc::ShaderKind::Mesh,
        ShaderKind::Task => shaderc::ShaderKind::Task,
    };

    let mut compile_options = shaderc::CompileOptions::new().unwrap();
    let mut base_include_path = std::env::current_dir().expect("failed to get current directory");

    match create_info.source {
        ShaderSource::File(path) => {
            if let Some(parent) = path.parent() {
                base_include_path = parent.to_path_buf();
            }
        }
        _ => {}
    }

    compile_options.set_include_callback(move |requested_source, type_, requesting_source, include_depth| {
        let mut path = base_include_path.clone();
        path.push(requested_source);
        let content = match std::fs::read_to_string(&path) {
            Ok(content) => content,
            Err(e) => return Err(e.to_string()),
        };
        Ok(shaderc::ResolvedInclude {
            resolved_name: path.display().to_string(),
            content,
        })
    });

    let compilation_artifact = COMPILER.compile_into_spirv(
        source_content,
        kind,
        "main",
        create_info.entry_point,
        Some(&compile_options),
    )?;

    Ok(compilation_artifact.as_binary().into())
}

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
}

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
}

////////////////////////////////////////////////////////////////////////////////////////////////////

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
        vertex: ShaderCode<'static>,
        tess_control: Option<ShaderCode<'static>>,
        tess_evaluation: Option<ShaderCode<'static>>,
        geometry: Option<ShaderCode<'static>>,
        fragment: ShaderCode<'static>,
    },
    /// Shaders of the mesh shading pipeline (the new mesh and task shaders).
    MeshShading {
        mesh: ShaderCode<'static>,
        task: ShaderCode<'static>,
        fragment: ShaderCode<'static>,
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

pub struct GraphicsPipeline<Args, PushCst, Output> {
    pipeline: vk::Pipeline,
    pipeline_layout: PipelineLayout,
    _phantom: std::marker::PhantomData<fn() -> (Args, PushCst, Output)>,
}

impl<Args: StaticArguments, PushCst: StaticPushConstants, Output: StaticAttachments>
    GraphicsPipeline<Args, PushCst, Output>
{
    /// Creates a new graphics pipeline with the given configuration.
    pub fn new(device: &Device, create_info: &GraphicsPipelineCreateInfo) -> Self {
        // create pipeline layout
        let pipeline_layout = PipelineLayout::new(
            device,
            &[&<Args as StaticArguments>::LAYOUT],
            PushCst::PUSH_CONSTANT_RANGES,
        );



        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            s_type: Default::default(),
            p_next: (),
            flags: Default::default(),
            stage_count: 0,
            p_stages: (),
            // Ignored, set dynamically
            p_vertex_input_state: vk::PipelineVertexInputStateCreateInfo::default(),
            p_input_assembly_state: (),
            p_tessellation_state: (),
            p_viewport_state: (),
            p_rasterization_state: (),
            p_multisample_state: (),
            p_depth_stencil_state: (),
            p_color_blend_state: (),
            p_dynamic_state: (),
            layout: Default::default(),
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default(),
        }
    }
}

/// Specifies the type of a graphics pipeline with the given vertex input, arguments and push constants.
#[macro_export]
macro_rules! GraphicsPipeline {
    [arguments = ($($args:ty),*) $(, push_constants=$push_cst_ty:ty)?, output_attachments = $attachments:ty] => {

    };
}
