extern crate shaderc;

use anyhow::Result;
use std::env;
use std::fs;
use std::fs::File;
use std::io::{Write, Read};
use std::path::{Path, PathBuf};

const SHADERS_DIR: &str = "shaders/";


fn compile_shader(
    compiler: &mut shaderc::Compiler,
    options: &shaderc::CompileOptions,
    source_file_path: &Path,
    kind: shaderc::ShaderKind,
    out_dir: &Path,
) -> Result<()> {

    let mut output_file_path = out_dir
        .to_owned()
        .with_file_name(format!("{}.spv",source_file_path.file_name().unwrap().to_str().unwrap()));

    let source_file_name = source_file_path.to_string_lossy();

    eprintln!(
        "-- Compiling SPIR-V: {}...",
        output_file_path.display()
    );

    // load source
    let mut source_file = File::open(source_file_path)?;
    let mut source = String::new();
    source_file.read_to_string(&mut source)?;

    let binary_result = compiler.compile_into_spirv(
        &source,
        kind,
        &source_file_name,
        "main",
        Some(&options),
    )?;

    // write the binary result to the output directory
    let mut output_file = File::create(output_file_path)?;
    output_file.write(binary_result.as_binary_u8())?;
    Ok(())
}

fn shader_kind_from_extension(path: &Path) -> Option<shaderc::ShaderKind> {
    if let Some(ext) = path.extension() {
        match ext.to_str() {
            Some("vert") => Some(shaderc::ShaderKind::Vertex),
            Some("frag") => Some(shaderc::ShaderKind::Fragment),
            Some("comp") => Some(shaderc::ShaderKind::Compute),
            Some("geom") => Some(shaderc::ShaderKind::Geometry),
            Some("tese") => Some(shaderc::ShaderKind::TessEvaluation),
            Some("tesc") => Some(shaderc::ShaderKind::TessControl),
            _ => None,
        }
    } else {
        None
    }
}

fn main() {
    // compile every shader under the "shaders/" subdirectory

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let mut compiler = shaderc::Compiler::new().expect("failed to create the compiler");
    let mut options = shaderc::CompileOptions::new().unwrap();

    let mut encountered_errors = false;
    for ent in fs::read_dir(SHADERS_DIR).expect("could not open shaders directory") {
        let r: Result<()> = (|| {
            let ent = ent?;
            if ent.file_type()?.is_file() {
                let file_path = ent.path();
                if let Some(shader_kind) = shader_kind_from_extension(&file_path) {
                    compile_shader(
                        &mut compiler,
                        &mut options,
                        &ent.path(),
                        shader_kind,
                        &out_dir,
                    )?;
                }
            }
            Ok(())
        })();

        if let Err(e) = r {
            eprintln!("error reading directory: {}", e);
            encountered_errors = true;
        }
    }

    if encountered_errors {
        panic!("Errors encountered when compiling shaders.");
    }

}
