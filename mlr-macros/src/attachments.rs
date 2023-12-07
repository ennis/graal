use crate::{expect_struct_fields, CRATE};
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;

pub(crate) fn derive_attachments(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
    let derive_input: syn::DeriveInput = syn::parse(input)?;
    let fields = expect_struct_fields(&derive_input, "Attachments")?;

    let mut color_formats = vec![];
    let mut color_attachments = vec![];
    let mut depth_format = None;
    let mut depth_attachment = None;
    let mut stencil_format = None;
    let mut stencil_attachment = None;

    for (_i, f) in fields.iter().enumerate() {
        let mut is_attachment = false;
        let mut is_color = false;
        let mut is_depth = false;
        let mut is_stencil = false;
        let mut load_op = None;
        let mut store_op = None;
        let mut clear_color = None;
        let mut clear_depth_stencil = None;
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

                    if meta.path.is_ident("load_op") {
                        let value = meta.value()?;
                        let op: syn::Ident = value.parse()?;
                        if op == "clear" {
                            load_op = Some(quote!(#CRATE::vk::AttachmentLoadOp::CLEAR));
                        } else if op == "load" {
                            load_op = Some(quote!(#CRATE::vk::AttachmentLoadOp::LOAD));
                        } else if op == "dont_care" {
                            load_op = Some(quote!(#CRATE::vk::AttachmentLoadOp::DONT_CARE));
                        } else {
                            return Err(meta.error("invalid syntax for `load_op`"));
                        }
                        return Ok(());
                    }

                    if meta.path.is_ident("store_op") {
                        let value = meta.value()?;
                        let op: syn::Ident = value.parse()?;
                        if op == "store" {
                            store_op = Some(quote!(#CRATE::vk::AttachmentStoreOp::STORE));
                        } else if op == "dont_care" {
                            store_op = Some(quote!(#CRATE::vk::AttachmentStoreOp::DONT_CARE));
                        } else {
                            return Err(meta.error("invalid syntax for `store_op`"));
                        }
                        return Ok(());
                    }

                    if meta.path.is_ident("clear_color") {
                        let value = meta.value()?;
                        let color: syn::Expr = value.parse()?;
                        clear_color = Some(
                            quote!(#CRATE::vk::ClearColorValue::from(#CRATE::attachments::ClearColorValue::from(#color))),
                        );
                        is_color = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("clear_depth") {
                        let value = meta.value()?;
                        let depth: syn::Expr = value.parse()?;
                        if clear_depth_stencil.is_some() {
                            return Err(meta.error(
                                "must specify only one of `clear_depth`, `clear_stencil`, or `clear_depth_stencil`",
                            ));
                        }
                        clear_depth_stencil = Some(quote!(#CRATE::vk::ClearDepthStencilValue {
                            depth: #depth,
                            stencil: 0,
                        }));
                        is_depth = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("clear_stencil") {
                        let value = meta.value()?;
                        let stencil: syn::Expr = value.parse()?;
                        if clear_depth_stencil.is_some() {
                            return Err(meta.error(
                                "must specify only one of `clear_depth`, `clear_stencil`, or `clear_depth_stencil`",
                            ));
                        }
                        clear_depth_stencil = Some(quote!(#CRATE::vk::ClearDepthStencilValue {
                            depth: 0.0,
                            stencil: #stencil,
                        }));
                        is_stencil = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("clear_depth_stencil") {
                        let value = meta.value()?;
                        let depth_stencil: syn::Expr = value.parse()?;
                        if clear_depth_stencil.is_some() {
                            return Err(meta.error(
                                "must specify only one of `clear_depth`, `clear_stencil`, or `clear_depth_stencil`",
                            ));
                        }
                        clear_depth_stencil = Some(quote!(#CRATE::vk::ClearDepthStencilValue {
                            depth: #depth_stencil.0,
                            stencil: #depth_stencil.1,
                        }));
                        is_depth = true;
                        is_stencil = true;
                        return Ok(());
                    }

                    Err(meta.error("invalid syntax for `#[attachment]`"))
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

        if is_color && clear_depth_stencil.is_some() {
            return Err(syn::Error::new(
                f.span(),
                "cannot specify `clear_depth` or `clear_depth_stencil` for a color attachment",
            ));
        }

        if (is_depth || is_stencil) && clear_color.is_some() {
            return Err(syn::Error::new(
                f.span(),
                "cannot specify `clear_color` for a depth or stencil attachment",
            ));
        }

        let format = if let Some(format) = format {
            format
        } else {
            return Err(syn::Error::new(
                f.span(),
                "missing `format` argument in `#[attachment(...)]`",
            ));
        };

        let field_name = &f.ident;

        let attachment_wrapper =
            if clear_depth_stencil.is_some() || clear_color.is_some() || store_op.is_some() || load_op.is_some() {
                // handle attachment overrides specified on the field
                let load_op = if let Some(load_op) = load_op {
                    quote!(Some(#load_op))
                } else {
                    quote!(None)
                };
                let store_op = if let Some(store_op) = store_op {
                    quote!(Some(#store_op))
                } else {
                    quote!(None)
                };
                let clear_value = if is_color {
                    if let Some(clear_color) = clear_color {
                        quote!(Some(#CRATE::vk::ClearValue { color: #clear_color }))
                    } else {
                        quote!(None)
                    }
                } else if is_depth || is_stencil {
                    if let Some(clear_depth_stencil) = clear_depth_stencil {
                        quote!(Some(#CRATE::vk::ClearValue { depth_stencil: #clear_depth_stencil }))
                    } else {
                        quote!(None)
                    }
                } else {
                    unreachable!()
                };
                quote! {
                     #CRATE::attachments::AsAttachment::as_attachment(&#CRATE::attachments::AttachmentOverride(
                        self.#field_name,
                        #load_op,
                        #store_op,
                        #clear_value
                    ))
                }
            } else {
                // no overrides
                quote!(#CRATE::attachments::AsAttachment::as_attachment(&self.#field_name))
            };

        let format = quote!(#CRATE::vk::Format::#format);

        if is_color {
            color_formats.push(format);
            color_attachments.push(attachment_wrapper);
        } else if is_depth {
            if depth_format.is_some() {
                return Err(syn::Error::new(f.span(), "more than one depth attachment specified"));
            }
            depth_attachment = Some(quote!(Some(#attachment_wrapper)));
            depth_format = Some(quote!(Some(#format)));
        } else if is_stencil {
            if stencil_format.is_some() {
                return Err(syn::Error::new(f.span(), "more than one stencil attachment specified"));
            }
            stencil_attachment = Some(quote!(Some(#attachment_wrapper)));
            stencil_format = Some(quote!(Some(#format)));
        }
    }

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    let depth_format = depth_format.unwrap_or(quote! { None });
    let stencil_format = stencil_format.unwrap_or(quote! { None });
    let depth_attachment = depth_attachment.unwrap_or(quote! { None });
    let stencil_attachment = stencil_attachment.unwrap_or(quote! { None });

    let color_attachment_seq = 0..color_attachments.len();

    Ok(quote! {
        impl #impl_generics #CRATE::attachments::StaticAttachments for #struct_name #ty_generics #where_clause {
            const COLOR: &'static [#CRATE::vk::Format] = &[#(#color_formats),*];
            const DEPTH: Option<#CRATE::vk::Format> = #depth_format;
            const STENCIL: Option<#CRATE::vk::Format> = #stencil_format;
        }

        impl #impl_generics #CRATE::attachments::Attachments for #struct_name #ty_generics #where_clause {
            fn color_attachments(&self) -> impl Iterator<Item = #CRATE::attachments::Attachment<'_>> + '_ {
                let mut index = 0;
                std::iter::from_fn(move || {
                    let r = match index {
                        #(#color_attachment_seq => Some(#color_attachments),)*
                        _ => None,
                    };
                    index += 1;
                    r
                })
            }

            fn depth_attachment(&self) -> Option<#CRATE::attachments::Attachment> {
                #depth_attachment
            }

            fn stencil_attachment(&self) -> Option<#CRATE::attachments::Attachment> {
                #stencil_attachment
            }
        }
    })
}
