use crate::{FieldList, G};
use darling::{
    util::{Flag, SpannedValue},
    FromDeriveInput, FromField, FromMeta,
};
use proc_macro::{Diagnostic, Level};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use std::ops::Deref;
use syn::spanned::Spanned;

#[derive(FromDeriveInput, Debug)]
#[darling(forward_attrs(allow, doc, cfg, repr))]
struct FragmentOutputStruct {
    ident: syn::Ident,
    generics: syn::Generics,
    vis: syn::Visibility,
    attrs: Vec<syn::Attribute>,
}

#[derive(FromField)]
#[darling(attributes(attachment))]
struct AttachmentAttr {
    format: syn::Ident,
    #[darling(default)]
    load_op: Option<syn::Ident>,
    #[darling(default)]
    store_op: Option<syn::Ident>,
    layout: syn::Ident,
    #[darling(default)]
    samples: SpannedValue<Option<u32>>,
    #[darling(default)]
    color: SpannedValue<Flag>,
    #[darling(default)]
    depth: SpannedValue<Flag>,
    ty: syn::Type,
}

pub fn generate(derive_input: &syn::DeriveInput, fields: &FieldList) -> TokenStream {
    let s: FragmentOutputStruct =
        <FragmentOutputStruct as FromDeriveInput>::from_derive_input(derive_input).unwrap();
    let struct_name = &s.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

    let mut attachments = Vec::new();
    let mut color_attachments = Vec::new();
    let mut depth_attachment = None;

    for field in fields.iter() {
        match <AttachmentAttr as FromField>::from_field(field) {
            Ok(attr) => {
                let attr: &AttachmentAttr = &attr;
                let i_attachment = attachments.len() as u32;
                let ty = &attr.ty;

                let format = &attr.format;
                let layout = &attr.layout;
                let dont_care = &syn::Ident::new("DONT_CARE", Span::call_site());
                let load_op = &attr.load_op.as_ref().unwrap_or(dont_care);
                let store_op = &attr.store_op.as_ref().unwrap_or(dont_care);

                let samples = match attr.samples.deref() {
                    None | Some(1) => syn::Ident::new("TYPE_1", Span::call_site()),
                    Some(2) => syn::Ident::new("TYPE_2", Span::call_site()),
                    Some(4) => syn::Ident::new("TYPE_4", Span::call_site()),
                    Some(8) => syn::Ident::new("TYPE_8", Span::call_site()),
                    Some(16) => syn::Ident::new("TYPE_16", Span::call_site()),
                    Some(32) => syn::Ident::new("TYPE_32", Span::call_site()),
                    Some(64) => syn::Ident::new("TYPE_64", Span::call_site()),
                    _ => {
                        attr.samples
                            .span()
                            .unwrap()
                            .error("Invalid sample count.")
                            .note("Valid values are: `1`, `2` ,`4`, `8`, `16`, `32`, `64`.")
                            .emit();
                        syn::Ident::new("TYPE_1", Span::call_site())
                    }
                };

                attachments.push(quote! {
                    #G::vk::AttachmentDescription {
                        flags: #G::vk::AttachmentDescriptionFlags::empty(),
                        format: #G::vk::Format::#format,
                        samples: #G::vk::SampleCountFlags::TYPE_1,
                        load_op: #G::vk::AttachmentLoadOp::#load_op,
                        store_op: #G::vk::AttachmentStoreOp::#store_op,
                        stencil_load_op: #G::vk::AttachmentLoadOp::DONT_CARE,   // TODO
                        stencil_store_op: #G::vk::AttachmentStoreOp::DONT_CARE,
                        initial_layout: #G::vk::ImageLayout::#layout,
                        final_layout: #G::vk::ImageLayout::#layout,
                    }
                });

                if attr.color.is_some() && attr.depth.is_some() {
                    Diagnostic::spanned(
                        &[attr.color.span().unwrap(), attr.depth.span().unwrap()][..],
                        Level::Error,
                        "Attachment cannot be both a color and a depth attachment within the same subpass.")
                        .emit();
                } else if attr.depth.is_some() {
                    if depth_attachment.is_some() {
                        attr.depth
                            .span()
                            .unwrap()
                            .error("More than one depth attachment specified.")
                            .emit();
                    } else {
                        depth_attachment = Some(quote! {
                             #G::vk::AttachmentReference {
                                attachment: #i_attachment,
                                layout: #G::vk::ImageLayout::#layout,
                             }
                        });
                    }
                } else {
                    color_attachments.push(quote! {
                        #G::vk::AttachmentReference {
                            attachment: #i_attachment,
                            layout: #G::vk::ImageLayout::#layout,
                        }
                    });
                }
            }
            Err(e) => {
                e.write_errors();
            }
        }
    }

    let (depth_attachment, p_depth_stencil_attachment) = if let Some(a) = &depth_attachment {
        (quote! { Some(#a) }, quote! { &#a })
    } else {
        (quote! { None }, quote! { ::std::ptr::null() })
    };

    let q = quote! {
        impl #impl_generics #G::FragmentOutputInterface for #struct_name #ty_generics #where_clause {
            const ATTACHMENTS: &'static [#G::vk::AttachmentDescription] = &[ #(#attachments,)* ];
            const COLOR_ATTACHMENTS: &'static [#G::vk::AttachmentReference] = &[ #(#color_attachments,)* ];
            const DEPTH_ATTACHMENT: Option<#G::vk::AttachmentReference> = #depth_attachment;
            const RENDER_PASS_CREATE_INFO: &'static #G::vk::RenderPassCreateInfo = &#G::vk::RenderPassCreateInfo {
                s_type: #G::vk::StructureType::RENDER_PASS_CREATE_INFO,
                p_next: ::std::ptr::null(),
                flags: #G::vk::RenderPassCreateFlags::empty(),
                attachment_count: Self::ATTACHMENTS.len() as u32,
                p_attachments: Self::ATTACHMENTS.as_ptr(),
                subpass_count: 1,
                p_subpasses: &#G::vk::SubpassDescription {
                    flags: #G::vk::SubpassDescriptionFlags::empty(),
                    pipeline_bind_point: #G::vk::PipelineBindPoint::GRAPHICS,
                    input_attachment_count: 0,
                    p_input_attachments: ::std::ptr::null(),
                    color_attachment_count: Self::COLOR_ATTACHMENTS.len() as u32,
                    p_color_attachments: Self::COLOR_ATTACHMENTS.as_ptr(),
                    p_resolve_attachments: ::std::ptr::null(),
                    p_depth_stencil_attachment: #p_depth_stencil_attachment,
                    preserve_attachment_count: 0,
                    p_preserve_attachments: ::std::ptr::null(),
                },
                dependency_count: 0,
                p_dependencies: ::std::ptr::null(),
            };
        }
    };
    q
}
