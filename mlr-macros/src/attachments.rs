use crate::{FieldList, CRATE};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::ops::Deref;
use syn::spanned::Spanned;

pub fn derive_attachments(input: proc_macro::TokenStream) -> TokenStream {
    match derive_attachments_(input) {
        Ok(tokens) => tokens,
        Err(e) => e.into_compile_error(),
    }
}

fn derive_attachments_(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
    let derive_input: syn::DeriveInput = syn::parse(input)?;

    // check for struct
    let fields = match derive_input.data {
        syn::Data::Struct(ref struct_data) => &struct_data.fields,
        _ => {
            return Err(syn::Error::new(
                derive_input.span(),
                "`Vertex` can only be derived on structs",
            ));
        }
    };

    let mut color_formats = vec![];
    let mut depth_format = None;
    let mut stencil_format = None;

    for (_i, f) in fields.iter().enumerate() {
        let mut is_attachment = false;
        let mut is_color = false;
        let mut is_depth = false;
        let mut is_stencil = false;
        let mut format = None;
        for attr in f.attrs.iter() {
            if attr.path().is_ident("attachment") {
                is_attachment = true;
                attr.meta.require_list()?.parse_nested_meta(|meta| {
                    if meta.path.is_ident("color") {
                        //meta.parse_nested_meta(|meta| Err(meta.error("invalid syntax for `color`")))?;
                        is_color = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("depth") {
                        //meta.parse_nested_meta(|meta| Err(meta.error("invalid syntax for `depth`")))?;
                        is_depth = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("stencil") {
                        //meta.parse_nested_meta(|meta| Err(meta.error("invalid syntax for `stencil`")))?;
                        is_stencil = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("format") {
                        let value = meta.value()?;
                        let f: syn::Ident = value.parse()?;
                        format = Some(f);
                        return Ok(());
                    }

                    Err(meta.error("invalid `attachment` syntax"))
                })?;
            }
        }

        // Checks
        if !is_attachment {
            return Err(syn::Error::new(f.span(), "missing `#[attachment(...)]` attribute"));
        }
        if is_color && is_depth {
            return Err(syn::Error::new(
                f.span(),
                "cannot be both a color and a depth attachment",
            ));
        }
        if is_color && is_stencil {
            return Err(syn::Error::new(
                f.span(),
                "cannot be both a color and a stencil attachment",
            ));
        }
        if !(is_color || is_depth || is_stencil) {
            // If unspecified, assume that this is a color attachment.
            is_color = true;
        }

        let format = if let Some(format) = format {
            format
        } else {
            return Err(syn::Error::new(
                f.span(),
                "missing `format` argument in `#[attachment(...)]`",
            ));
        };

        if is_color {
            color_formats.push(quote!(#CRATE::vk::Format::#format));
        } else if is_depth {
            if depth_format.is_some() {
                return Err(syn::Error::new(f.span(), "more than one depth attachment specified"));
            }
            depth_format = Some(quote!(Some(#CRATE::vk::Format::#format)));
        } else if is_stencil {
            if stencil_format.is_some() {
                return Err(syn::Error::new(f.span(), "more than one stencil attachment specified"));
            }
            stencil_format = Some(quote!(Some(#CRATE::vk::Format::#format)));
        }
    }

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    let depth_format = depth_format.unwrap_or(quote! { None });
    let stencil_format = stencil_format.unwrap_or(quote! { None });

    Ok(quote! {
        impl #impl_generics #CRATE::attachments::Attachments for #struct_name #ty_generics #where_clause {
            const COLOR: &'static [#CRATE::vk::Format] = &[#(#color_formats),*];
            const DEPTH: Option<#CRATE::vk::Format> = #depth_format;
            const STENCIL: Option<#CRATE::vk::Format> = #stencil_format;
        }
    })
}
