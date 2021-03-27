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
struct PipelineInterfaceStruct {
    ident: syn::Ident,
    generics: syn::Generics,
    vis: syn::Visibility,
    attrs: Vec<syn::Attribute>,
}

/*
#[derive(Debug, FromField)]
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
}*/

pub fn generate(derive_input: &syn::DeriveInput, fields: &FieldList) -> TokenStream {
    let s: PipelineInterfaceStruct =
        <PipelineInterfaceStruct as FromDeriveInput>::from_derive_input(derive_input).unwrap();
    let struct_name = &s.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

    let mut attachments = Vec::new();
    let mut color_attachments = Vec::new();
    let mut depth_attachment = None;
    let mut create_transient_image_statements = Vec::new();
    let mut create_image_view_statements = Vec::new();
    let mut field_names = Vec::new();

    // TODO reject tuple structs
    for field in fields.iter() {
        let field_name = field.ident.as_ref().unwrap();
        field_names.push(field_name);

        if let Ok(attr) = <AttachmentAttr as FromField>::from_field(field) {
            let attr: AttachmentAttr = attr;
            let i_attachment = attachments.len() as u32;

            dbg!(&attr);

            let format = attr.format;
            let layout = attr.layout;
            let dont_care = &syn::Ident::new("DONT_CARE", Span::call_site());
            let load_op = attr.load_op.as_ref().unwrap_or(dont_care);
            let store_op = attr.store_op.as_ref().unwrap_or(dont_care);

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
                    samples: #G::vk::SampleCountFlags::#samples,
                    load_op: #G::vk::AttachmentLoadOp::#load_op,
                    store_op: #G::vk::AttachmentStoreOp::#store_op,
                    stencil_load_op: #G::vk::AttachmentLoadOp::DONT_CARE,   // TODO
                    stencil_store_op: #G::vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: #G::vk::ImageLayout::#layout,
                    final_layout: #G::vk::ImageLayout::#layout,
                }
            });

            let n_samples = attr.samples.unwrap_or(1);

            let is_depth = if attr.color.is_some() && attr.depth.is_some() {
                Diagnostic::spanned(
                    &[attr.color.span().unwrap(), attr.depth.span().unwrap()][..],
                    Level::Error,
                    "Attachment cannot be both a color and a depth attachment within the same subpass.")
                    .emit();
                false
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
                true
            } else {
                color_attachments.push(quote! {
                    #G::vk::AttachmentReference {
                        attachment: #i_attachment,
                        layout: #G::vk::ImageLayout::#layout,
                    }
                });
                false
            };

            let debug_name = format!("{}::{}", struct_name.to_string(), field_name.to_string());

            let base_usage = if is_depth {
                quote! { #G::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT }
            } else {
                quote! { #G::vk::ImageUsageFlags::COLOR_ATTACHMENT }
            };

            let aspect_mask = if is_depth {
                quote! { #G::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT }
            } else {
                quote! { #G::vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT }
            };

            create_transient_image_statements.push(quote! {
                let #field_name = batch.context().create_image(
                    #debug_name,
                    &#G::ResourceMemoryInfo::DEVICE_LOCAL,
                    &#G::ImageResourceCreateInfo {
                        image_type: #G::vk::ImageType::TYPE_2D,
                        usage: #base_usage // TODO control usage flags per-image
                            | additional_usage,
                        format: #G::vk::Format::#format,
                        extent,
                        mip_levels: 1,
                        array_layers: 1,
                        samples: #n_samples,
                        tiling: #G::vk::ImageTiling::OPTIMAL,
                    },
                    true
                );
            });

            create_image_view_statements.push(quote! {
                let #field_name = unsafe {
                    cmd_ctx.create_image_view(&#G::vk::ImageViewCreateInfo {
                        flags: #G::vk::ImageViewCreateFlags::empty(),
                        image: self.#field_name.handle,
                        view_type: #G::vk::ImageViewType::TYPE_2D,
                        format: #G::vk::Format::#format,
                        components: #G::vk::ComponentMapping::default(),
                        subresource_range: #G::vk::ImageSubresourceRange {
                            aspect_mask: #G::format_aspect_mask(#G::vk::Format::#format),   // TODO
                            base_mip_level: 0,
                            level_count: #G::vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: #G::vk::REMAINING_ARRAY_LAYERS,
                        },
                        .. Default::default()
                    })
                };
            });
        } else {
            field
                .span()
                .unwrap()
                .error("missing `#[attachment]` attribute")
                .note("all fields of a `FragmentOutputInteface` should represent attachments")
                .emit()
        }
    }

    let (depth_attachment, p_depth_stencil_attachment) = if let Some(a) = &depth_attachment {
        (quote! { Some(#a) }, quote! { &#a })
    } else {
        (quote! { None }, quote! { ::std::ptr::null() })
    };

    let field_names = &field_names[..];

    let q = quote! {
        unsafe impl #impl_generics #G::FragmentOutputInterface for #struct_name #ty_generics #where_clause {
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

            fn get_or_init_render_pass(init: impl FnOnce() -> #G::RenderPassId) -> #G::RenderPassId {
                static RENDER_PASS_ID: #G::internal::OnceCell<#G::RenderPassId> = #G::internal::OnceCell::new();
                *RENDER_PASS_ID.get_or_init(init)
            }

            fn new(batch: &#G::Batch, additional_usage: #G::vk::ImageUsageFlags, size: (u32, u32)) -> Self {
                let extent = #G::vk::Extent3D {
                    width: size.0,
                    height: size.1,
                    depth: 1
                };

                #(#create_transient_image_statements)*

                #struct_name {
                    #(#field_names,)*
                }
            }

            fn create_framebuffer(&self, cmd_ctx: &mut #G::CommandContext, size: (u32,u32)) -> #G::vk::Framebuffer {
                let render_pass = cmd_ctx.get_or_create_render_pass_from_interface::<Self>();

                #(#create_image_view_statements)*

                unsafe {
                    cmd_ctx.create_framebuffer(
                        size.0,
                        size.1,
                        1,
                        render_pass,
                        &[#(#field_names,)*],
                    )
                }
            }
        }
    };
    q
}
