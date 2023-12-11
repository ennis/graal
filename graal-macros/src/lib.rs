//! Proc-macro for auto-deriving shader interfaces:
//! - `BufferLayout`
//! - `VertexLayout`
//! - `VertexInputInterface`
//! - `DescriptorSetInterface`
//! - `PushConstantInterface`
//! - `FragmentOutputInterface`
#![recursion_limit = "256"]
//#![feature(proc_macro_diagnostic)]
extern crate darling;
extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens, TokenStreamExt};
use syn::{spanned::Spanned, MetaList};

mod arguments;
mod attachments;
mod push_constants;
mod vertex;

//--------------------------------------------------------------------------------------------------
struct CrateName;
const CRATE: CrateName = CrateName;

impl ToTokens for CrateName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(syn::Ident::new("graal", Span::call_site()))
    }
}

#[derive(Default, PartialEq, Eq)]
struct StageFlags {
    vertex: bool,
    tess_control: bool,
    tess_evaluation: bool,
    geometry: bool,
    fragment: bool,
    compute: bool,
    mesh: bool,
    task: bool,
    all: bool,
    all_graphics: bool,
}

impl StageFlags {
    fn all() -> Self {
        Self {
            vertex: false,
            tess_control: false,
            tess_evaluation: false,
            geometry: false,
            fragment: false,
            compute: false,
            mesh: false,
            task: false,
            all: true,
            all_graphics: false,
        }
    }

    /*fn all_graphics() -> Self {
        Self {
            vertex: false,
            tess_control: false,
            tess_evaluation: false,
            geometry: false,
            fragment: false,
            compute: false,
            mesh: false,
            task: false,
            all: false,
            all_graphics: true,
        }
    }*/

    fn from_meta_list(meta_list: &MetaList) -> syn::Result<StageFlags> {
        let mut stage_flags = StageFlags::default();
        meta_list.parse_nested_meta(|meta| {
            if meta.path.is_ident("vertex") {
                stage_flags.vertex = true;
                return Ok(());
            }
            if meta.path.is_ident("tess_control") {
                stage_flags.tess_control = true;
                return Ok(());
            }
            if meta.path.is_ident("tess_evaluation") {
                stage_flags.tess_evaluation = true;
                return Ok(());
            }
            if meta.path.is_ident("geometry") {
                stage_flags.geometry = true;
                return Ok(());
            }
            if meta.path.is_ident("fragment") {
                stage_flags.fragment = true;
                return Ok(());
            }
            if meta.path.is_ident("compute") {
                stage_flags.compute = true;
                return Ok(());
            }
            if meta.path.is_ident("mesh") {
                stage_flags.mesh = true;
                return Ok(());
            }
            if meta.path.is_ident("task") {
                stage_flags.task = true;
                return Ok(());
            }
            if meta.path.is_ident("all_graphics") {
                stage_flags.all_graphics = true;
                return Ok(());
            }
            if meta.path.is_ident("all") {
                stage_flags.all = true;
                return Ok(());
            }
            Err(meta.error("invalid syntax for `stages(...)`"))
        })?;
        if stage_flags == StageFlags::default() {
            return Err(syn::Error::new(meta_list.span(), "no shader stages specified"));
        }
        Ok(stage_flags)
    }
}

impl ToTokens for StageFlags {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.all {
            tokens.append_all(quote!(#CRATE::vk::ShaderStageFlags::ALL));
        } else if self.all_graphics {
            tokens.append_all(quote!(#CRATE::vk::ShaderStageFlags::ALL_GRAPHICS));
        } else {
            let mut tk = TokenStream::new();
            tk.append_all(quote!(#CRATE::vk::ShaderStageFlags::empty().as_raw()));
            if self.vertex {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::VERTEX.as_raw() });
            }
            if self.tess_control {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::TESSELLATION_CONTROL.as_raw() });
            }
            if self.tess_evaluation {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::TESSELLATION_EVALUATION.as_raw() });
            }
            if self.geometry {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::GEOMETRY.as_raw() });
            }
            if self.fragment {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::FRAGMENT.as_raw() });
            }
            if self.compute {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::COMPUTE.as_raw() });
            }
            if self.mesh {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::MESH_NV.as_raw() });
            }
            if self.task {
                tk.append_all(quote! { | #CRATE::vk::ShaderStageFlags::TASK_NV.as_raw() });
            }
            tokens.append_all(quote!( #CRATE::vk::ShaderStageFlags::from_raw(#tk)));
        }
    }
}

fn expect_struct_fields<'a>(input: &'a syn::DeriveInput, derive_name: &str) -> syn::Result<&'a syn::Fields> {
    match input.data {
        syn::Data::Struct(ref data_struct) => Ok(&data_struct.fields),
        _ => Err(syn::Error::new(
            input.span(),
            format!("`{derive_name}` can only be derived on structs"),
        )),
    }
}

//--------------------------------------------------------------------------------------------------

fn try_derive(
    input: proc_macro::TokenStream,
    f: fn(proc_macro::TokenStream) -> syn::Result<TokenStream>,
) -> proc_macro::TokenStream {
    match f(input) {
        Ok(tokens) => tokens.into(),
        Err(e) => e.into_compile_error().into(),
    }
}

#[proc_macro_derive(Vertex, attributes(normalized))]
pub fn vertex_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, vertex::derive_vertex)
}

#[proc_macro_derive(Attachments, attributes(attachment))]
pub fn attachments_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, attachments::derive_attachments)
}

#[proc_macro_derive(Arguments, attributes(argument))]
pub fn arguments_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, arguments::derive_arguments)
}

#[proc_macro_derive(PushConstants, attributes(stages))]
pub fn push_constants_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, push_constants::derive_push_constants)
}

/*
#[proc_macro_derive(StructLayout)]
pub fn struct_layout_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    struct_layout::derive(input).into()
}*/

/*#[proc_macro_derive(VertexInputInterface, attributes(layout))]
pub fn vertex_input_interface_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "VertexInputInterface",
        input,
        vertex_input_interface::generate,
    )
}

#[proc_macro_derive(FragmentOutputInterface, attributes(attachment))]
pub fn fragment_output_interface_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "FragmentOutputInterface",
        input,
        fragment_output_interface::generate,
    )
}*/

/*
#[proc_macro_derive(PipelineInterface, attributes(descriptor_set))]
pub fn pipeline_interface_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "PipelineInterface",
        input,
        pipeline_interface::generate,
    )
}
*/

/*#[proc_macro_derive(StructuredBufferData)]
pub fn structured_buffer_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "StructuredBufferData",
        input,
        vertex_data::generate_structured_buffer_data,
    )
}*/

/*
#[proc_macro_derive(VertexData)]
pub fn vertex_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse item");

    let result = match ast.data {
        syn::Data::Struct(ref s) => layout::generate_vertex_data(&ast, &s.fields),
        _ => panic!("BufferLayout trait can only be automatically derived on structs."),
    };

    result.into()
}

#[proc_macro_derive(Arguments, attributes(argument))]
pub fn arguments_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse item");

    let result = match ast.data {
        syn::Data::Struct(ref s) => arguments::generate(&ast, &s.fields),
        _ => panic!("PipelineInterface trait can only be derived on structs"),
    };

    result.into()
}
*/
