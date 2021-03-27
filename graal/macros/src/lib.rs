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

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, TokenStreamExt};
use syn::spanned::Spanned;

//--------------------------------------------------------------------------------------------------
struct CrateName;
const G: CrateName = CrateName;

impl ToTokens for CrateName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(syn::Ident::new("graal", Span::call_site()))
    }
}

type FieldList = syn::punctuated::Punctuated<syn::Field, syn::Token![,]>;

//--------------------------------------------------------------------------------------------------

mod buffer_data;
mod descriptor_set_interface;
mod fragment_output_interface;
mod struct_layout;
mod vertex_input_interface;
//mod pipeline_interface;

pub(crate) use struct_layout::{ensure_repr_c, generate_field_offsets_and_sizes, has_repr_c_attr};

fn derive_struct(
    name: &str,
    input: proc_macro::TokenStream,
    generator: fn(&syn::DeriveInput, &FieldList) -> proc_macro2::TokenStream,
) -> proc_macro::TokenStream {
    let derive_input: syn::DeriveInput = syn::parse(input).expect("couldn't parse item");

    let result = match derive_input.data {
        syn::Data::Struct(ref s) => {
            let fields = match s.fields {
                syn::Fields::Named(ref fields_named) => &fields_named.named,
                syn::Fields::Unnamed(ref fields_unnamed) => &fields_unnamed.unnamed,
                syn::Fields::Unit => {
                    derive_input
                        .span()
                        .unwrap()
                        .error("`{}` cannot be derived on unit structs")
                        .emit();
                    return Default::default();
                }
            };

            generator(&derive_input, fields)
        }
        _ => {
            derive_input
                .span()
                .unwrap()
                .error(format!("`{}` can only be derived for struct types", name))
                .emit();
            return Default::default();
        }
    };

    result.into()
}

#[proc_macro_derive(DescriptorSetInterface, attributes(layout))]
pub fn descriptor_set_interface_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "DescriptorSetInterface",
        input,
        descriptor_set_interface::generate,
    )
}

#[proc_macro_derive(VertexData)]
pub fn vertex_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct("VertexData", input, buffer_data::generate_vertex_data)
}

#[proc_macro_derive(VertexInputInterface, attributes(layout))]
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
}

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

#[proc_macro_derive(StructuredBufferData)]
pub fn structured_buffer_data_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    derive_struct(
        "StructuredBufferData",
        input,
        buffer_data::generate_structured_buffer_data,
    )
}

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
