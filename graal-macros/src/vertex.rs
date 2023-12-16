use crate::CRATE;
use proc_macro2::TokenStream;
use quote::quote;
use syn::spanned::Spanned;

pub(crate) fn derive_vertex(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
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

    let struct_name = &derive_input.ident;

    let mut attribute_descs = vec![];
    for (i, f) in fields.iter().enumerate() {
        let field_ty = &f.ty;

        let mut normalized_attr = false;
        for attr in f.attrs.iter() {
            if attr.path().is_ident("normalized") {
                attr.meta.require_path_only()?;
                normalized_attr = true;
            }
        }

        let format = if normalized_attr {
            quote!(<#CRATE::vertex::Norm<#field_ty> as #CRATE::VertexAttribute>::FORMAT)
        } else {
            quote!(<#field_ty as #CRATE::VertexAttribute>::FORMAT)
        };

        match f.ident {
            None => {
                let index = syn::Index::from(i);
                attribute_descs.push(quote! {
                    #CRATE::VertexAttributeDescription {
                        format: #format,
                        offset: #CRATE::__offset_of_tuple!(#struct_name, #index) as u32,
                    }
                });
            }
            Some(ref ident) => {
                attribute_descs.push(quote! {
                    #CRATE::VertexAttributeDescription {
                        format: #format,
                        offset: #CRATE::__offset_of!(#struct_name, #ident) as u32,
                    }
                });
            }
        }
    }

    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    Ok(quote! {
        unsafe impl #impl_generics #CRATE::Vertex for #struct_name #ty_generics #where_clause {
            const ATTRIBUTES: &'static [#CRATE::VertexAttributeDescription] = {
                &[#(#attribute_descs,)*]
            };
        }
    })
}
