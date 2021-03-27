use crate::{FieldList, G};
use darling::{
    util::{Flag, SpannedValue},
    FromDeriveInput, FromField, FromMeta,
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;

#[derive(FromDeriveInput, Debug)]
#[darling(forward_attrs(allow, doc, cfg, repr))]
struct VertexInputStruct {
    ident: syn::Ident,
    generics: syn::Generics,
    vis: syn::Visibility,
    attrs: Vec<syn::Attribute>,
}

#[derive(FromField)]
#[darling(attributes(layout))]
struct LayoutAttr {
    #[darling(default)]
    binding: u32,
    #[darling(default)]
    location: u32,
    #[darling(default)]
    per_instance: Flag,
    #[darling(default)]
    per_vertex: Flag,
    ty: syn::Type,
}

pub fn generate(derive_input: &syn::DeriveInput, fields: &FieldList) -> TokenStream {
    let s: VertexInputStruct =
        <VertexInputStruct as FromDeriveInput>::from_derive_input(derive_input).unwrap();
    let struct_name = &s.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

    let mut bindings = Vec::new();
    let mut attrib_count = Vec::new();
    let mut attrib_concat_stmts = Vec::new();

    for field in fields.iter() {
        match <LayoutAttr as FromField>::from_field(field) {
            Ok(attr) => {
                let attr: &LayoutAttr = &attr;
                let ib = bindings.len();
                let ty = &attr.ty;
                let binding = attr.binding;
                let location = attr.location;

                if attr.per_instance.is_some() && attr.per_vertex.is_some() {
                    field
                        .span()
                        .unwrap()
                        .error("Multiple input rates specified.")
                        .note("Expected either `per_vertex` or `per_instance`")
                        .emit();
                    continue;
                }

                let input_rate = if attr.per_instance.is_some() {
                    quote! {#G::vk::VertexInputRate::INSTANCE}
                } else {
                    quote! {#G::vk::VertexInputRate::VERTEX}
                };

                bindings.push(quote! {
                    #G::vk::VertexInputBindingDescription {
                        binding: #binding,
                        stride: <#ty as #G::VertexBindingInterface>::STRIDE as u32,
                        input_rate: #input_rate,
                    }
                });

                attrib_count.push(quote! { <#ty as #G::VertexBindingInterface>::ATTRIBUTES.len() });

                let concat_ident_0 = syn::Ident::new(&format!("X_{}", ib), Span::call_site());
                let concat_ident_1 = syn::Ident::new(&format!("X_{}", ib + 1), Span::call_site());
                attrib_concat_stmts.push(quote! {
                    const #concat_ident_1: &[#G::vk::VertexInputAttributeDescription] =
                        &#G::vertex_macro_helpers::append_attributes::<{#concat_ident_0.len() + <#ty as #G::VertexBindingInterface>::ATTRIBUTES.len()}>(
                            #concat_ident_0,
                            #binding,
                            #location,
                            <#ty as #G::VertexBindingInterface>::ATTRIBUTES);
                });
            }
            Err(e) => {
                e.write_errors();
            }
        }
    }

    let last_concat_ident = syn::Ident::new(&format!("X_{}", bindings.len()), Span::call_site());

    let q = quote! {
        impl #impl_generics #G::VertexInputInterface for #struct_name #ty_generics #where_clause {
            const BINDINGS: &'static [#G::vk::VertexInputBindingDescription] = &[ #(#bindings,)* ];
            const ATTRIBUTES: &'static [#G::vk::VertexInputAttributeDescription] = {
                const X_0: &'static [#G::vk::VertexInputAttributeDescription] = &[];
                #(#attrib_concat_stmts)*
                #last_concat_ident
            };
        }
    };
    q
}
