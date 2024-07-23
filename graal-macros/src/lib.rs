#![recursion_limit = "256"]
//#![feature(proc_macro_diagnostic)]
extern crate darling;
extern crate proc_macro;
extern crate quote;
extern crate syn;

use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, TokenStreamExt};
use syn::spanned::Spanned;

//mod arguments;
mod attachments;
mod vertex;

//--------------------------------------------------------------------------------------------------
struct CrateName;
const CRATE: CrateName = CrateName;

impl ToTokens for CrateName {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        tokens.append(syn::Ident::new("graal", Span::call_site()))
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

/*
#[proc_macro_derive(Arguments, attributes(argument))]
pub fn arguments_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, arguments::derive_arguments)
}*/

/*
#[proc_macro_derive(PushConstants, attributes(stages))]
pub fn push_constants_derive(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    try_derive(input, push_constants::derive_push_constants)
}
*/
