//! Proc-macro for auto-deriving shader interfaces:
//! - `BufferLayout`
//! - `VertexLayout`
//! - `VertexInputInterface`
//! - `DescriptorSetInterface`
//! - `PushConstantInterface`
//! - `FragmentOutputInterface`
#![recursion_limit = "256"]
#![feature(proc_macro_diagnostic)]

extern crate darling; // this is a _good crate_
extern crate proc_macro;
extern crate quote;
extern crate syn;

use quote::{ToTokens, TokenStreamExt};
use proc_macro2::{TokenStream, Span};

//--------------------------------------------------------------------------------------------------
struct CrateName;
const G: CrateName = CrateName;

impl ToTokens for CrateName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(syn::Ident::new("graal", Span::call_site()))
    }
}

//--------------------------------------------------------------------------------------------------

mod vertex_input_interface;
mod descriptor_set_interface;

#[proc_macro_derive(DescriptorSetInterface, attributes(layout))]
pub fn descriptor_set_interface_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let derive_input: syn::DeriveInput = syn::parse(input).expect("Couldn't parse item");

    let result = match derive_input.data {
        syn::Data::Struct(ref s) => descriptor_set_interface::generate(&derive_input, &s.fields),
        _ => panic!("DescriptorSetInterface trait can only be derived on structs"),
    };

    result.into()
}

/*
#[proc_macro_derive(StructuredBufferData)]
pub fn structured_buffer_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let ast: syn::DeriveInput = syn::parse(input).expect("Couldn't parse item");

    let result = match ast.data {
        syn::Data::Struct(ref s) => layout::generate_structured_buffer_data(&ast, &s.fields),
        _ => panic!("StructuredBufferData trait can only be automatically derived on structs."),
    };

    result.into()
}

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