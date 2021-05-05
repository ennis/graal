use crate::{ensure_repr_c, generate_field_offsets_and_sizes, FieldList, G};
use proc_macro2::TokenStream;
use quote::quote;

pub fn generate_structured_buffer_data(
    derive_input: &syn::DeriveInput,
    fields: &FieldList,
) -> TokenStream {
    if let Err(e) = ensure_repr_c("StructuredBufferData", derive_input) {
        return e;
    }

    let struct_name = &derive_input.ident;
    let field_offsets_sizes = generate_field_offsets_and_sizes(derive_input);

    let mut struct_fields = Vec::new();
    let mut layouts = Vec::new();
    let mut offsets = Vec::new();

    for (i, f) in fields.iter().enumerate() {
        let field_ty = &f.ty;
        let offset = &field_offsets_sizes.offsets[i].ident;

        // skip padding fields (with an underscore)
        if f.ident.as_ref().unwrap().to_string().starts_with('_') {
            continue;
        }

        struct_fields.push(quote! {
            #G::typedesc::StructField {
                ty: &<#field_ty as #G::StructuredBufferData>::TYPE,
                .. #G::typedesc::StructField::new()
            }
        });

        offsets.push(quote! { Self::#offset });
        layouts.push(quote! { <#field_ty as #G::StructuredBufferData>::LAYOUT });
    }

    let field_offsets_sizes_impl = field_offsets_sizes.impl_block;

    quote! {
        #field_offsets_sizes_impl

        unsafe impl #G::StructuredBufferData for #struct_name {
            const TYPE: #G::typedesc::TypeDesc<'static> = #G::typedesc::TypeDesc::Struct(
                #G::typedesc::StructType {
                    fields: &[#(#struct_fields),*],
                    .. #G::typedesc::StructType::new()
                }
            );
            const LAYOUT: #G::layout::Layout<'static> = #G::layout::Layout {
                align: std::mem::align_of::<#struct_name>(),
                size: std::mem::size_of::<#struct_name>(),
                inner: #G::layout::InnerLayout::Struct(#G::layout::FieldsLayout {
                    offsets: &[#(#offsets),*],
                    layouts: &[#(&#layouts),*]
                })
            };
        }
    }
}

pub fn generate_vertex_data(derive_input: &syn::DeriveInput, fields: &FieldList) -> TokenStream {
    if let Err(e) = ensure_repr_c("VertexData", derive_input) {
        return e;
    }

    let struct_name = &derive_input.ident;
    let field_offsets_sizes = generate_field_offsets_and_sizes(derive_input);

    let mut attribs = Vec::new();

    for (i, f) in fields.iter().enumerate() {
        let field_ty = &f.ty;
        let offset = &field_offsets_sizes.offsets[i];
        let offset = &offset.ident;

        attribs.push(quote! {
            #G::VertexAttribute {
                format: <#field_ty as #G::VertexAttributeType>::FORMAT,
                offset: Self::#offset as u32,
            }
        });
    }

    let field_offsets_sizes_impl = field_offsets_sizes.impl_block;

    quote! {
        #field_offsets_sizes_impl

        unsafe impl #G::VertexData for #struct_name {
            const ATTRIBUTES: &'static [#G::VertexAttribute] = &[#(#attribs,)*];
        }
    }
}
