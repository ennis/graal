//! Shader compilation utilities.
use crate::{Error, ShaderSource, ShaderStage};
use once_cell::sync::Lazy;
use shaderc::{EnvVersion, OptimizationLevel, SpirvVersion, TargetEnv};
use std::borrow::Cow;
use tracing::error;

static SHADER_COMPILER: Lazy<shaderc::Compiler> = Lazy::new(|| shaderc::Compiler::new().unwrap());

/// Returns the global instance of the shader compiler (glslang via shaderc).
pub fn get_shader_compiler() -> &'static shaderc::Compiler {
    &SHADER_COMPILER
}

/// Compiles a shader with the specified options.
///
/// # Arguments
///
/// * `shader_kind` - the type of shader to compile.
/// * `source` - specifies the source code of the shader. This can be either a string of shader code or a path to a file containing the shader code.
/// * `entry_point` - the shader entry point.
/// * `source_prefix` - an optional prefix source code to prepend to the shader source.
/// * `compile_options` - specifies compilation options. See [`shaderc::CompileOptions`] for more information.
///
/// # Defines
/// The following defines are added automatically depending on the type of shader:
/// * Vertex => `#define __VERTEX__`
/// * Fragment => `#define __FRAGMENT__`
/// * Geometry => `#define __GEOMETRY__`
/// * Compute => `#define __COMPUTE__`
/// * TessControl => `#define __TESS_CONTROL__`
/// * TessEvaluation => `#define __TESS_EVAL__`
/// * Mesh => `#define __MESH__`
/// * Task => `#define __TASK__`
///
/// # Returns
/// A `Vec<u32>` containing the compiled SPIR-V, or an [`Error`] otherwise, which could be due to a compilation error or an IO error.
pub fn compile_shader(
    stage: ShaderStage,
    source: ShaderSource,
    entry_point: &str,
    source_prefix: &str,
    mut compile_options: shaderc::CompileOptions,
) -> Result<Vec<u32>, Error> {
    let input_file_name = match source {
        ShaderSource::Content(_) => Cow::Borrowed("<embedded shader>"),
        ShaderSource::File(path) => path
            .file_name()
            .ok_or(Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "invalid path supplied to compile_shader",
            )))?
            .to_string_lossy(),
    };

    let mut source_content = match source {
        ShaderSource::Content(str) => str.to_string(),
        ShaderSource::File(path) => std::fs::read_to_string(path)?,
    };

    if !source_prefix.is_empty() {
        let input_file_path = match source {
            ShaderSource::Content(_) => "0".to_string(),
            ShaderSource::File(path) => path.to_string_lossy().to_string(),
        };

        let header = format!("{source_prefix}\n#line 0 \"{input_file_path}\"\n");
        source_content.insert_str(0, &header);
    }

    let kind = match stage {
        ShaderStage::Vertex => shaderc::ShaderKind::Vertex,
        ShaderStage::Fragment => shaderc::ShaderKind::Fragment,
        ShaderStage::Geometry => shaderc::ShaderKind::Geometry,
        ShaderStage::Compute => shaderc::ShaderKind::Compute,
        ShaderStage::TessControl => shaderc::ShaderKind::TessControl,
        ShaderStage::TessEvaluation => shaderc::ShaderKind::TessEvaluation,
        ShaderStage::Mesh => shaderc::ShaderKind::Mesh,
        ShaderStage::Task => shaderc::ShaderKind::Task,
    };

    let mut base_include_path = std::env::current_dir().expect("failed to get current directory");

    match source {
        ShaderSource::File(path) => {
            if let Some(parent) = path.parent() {
                base_include_path = parent.to_path_buf();
            }
        }
        _ => {}
    }

    compile_options.set_include_callback(move |requested_source, _type, _requesting_source, _include_depth| {
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
    compile_options.set_target_env(TargetEnv::Vulkan, EnvVersion::Vulkan1_3 as u32);
    compile_options.set_target_spirv(SpirvVersion::V1_5);
    compile_options.set_generate_debug_info();
    compile_options.set_optimization_level(OptimizationLevel::Zero);
    match kind {
        shaderc::ShaderKind::Vertex => {
            compile_options.add_macro_definition("__VERTEX__", None);
        }
        shaderc::ShaderKind::Fragment => {
            compile_options.add_macro_definition("__FRAGMENT__", None);
        }
        shaderc::ShaderKind::Geometry => {
            compile_options.add_macro_definition("__GEOMETRY__", None);
        }
        shaderc::ShaderKind::Compute => {
            compile_options.add_macro_definition("__COMPUTE__", None);
        }
        shaderc::ShaderKind::TessControl => {
            compile_options.add_macro_definition("__TESS_CONTROL__", None);
        }
        shaderc::ShaderKind::TessEvaluation => {
            compile_options.add_macro_definition("__TESS_EVAL__", None);
        }
        shaderc::ShaderKind::Mesh => {
            compile_options.add_macro_definition("__MESH__", None);
        }
        shaderc::ShaderKind::Task => {
            compile_options.add_macro_definition("__TASK__", None);
        }
        _ => {}
    }

    let compiler = get_shader_compiler();
    let result = compiler.compile_into_spirv(
        &source_content,
        kind,
        &input_file_name,
        entry_point,
        Some(&compile_options),
    );

    let compilation_artifact = match result {
        Ok(artifact) => artifact,
        Err(e) => {
            error!("failed to compile shader (`{}`): {}", input_file_name, e);
            return Err(e.into());
        }
    };

    Ok(compilation_artifact.as_binary().into())
}
