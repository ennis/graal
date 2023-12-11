use crate::CRATE;
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::{spanned::Spanned, Fields, Ident, ItemConst, ItemStruct};

/// Checks that the derive input has a repr(C) attribute.
pub(crate) fn has_repr_c_attr(ast: &syn::DeriveInput) -> bool {
    ast.attrs.iter().any(|attr| match attr.parse_meta() {
        Ok(meta) => match meta {
            syn::Meta::List(list) => {
                (list
                    .path
                    .get_ident()
                    .map_or(false, |i| i.to_string() == "repr"))
                    && list.nested.iter().next().map_or(false, |n| match n {
                        syn::NestedMeta::Meta(syn::Meta::Path(ref path)) => {
                            path.get_ident().map_or(false, |i| i.to_string() == "C")
                        }
                        _ => false,
                    })
            }
            _ => false,
        },
        Err(_) => false,
    })
}

pub(crate) fn ensure_repr_c_derive_input(ast: &syn::DeriveInput) -> TokenStream {
    if !has_repr_c_attr(ast) {
        syn::Error::new(
            ast.span(),
            format!("Cannot derive trait on a non-`repr(C)` struct"),
        )
        .into_compile_error()
    } else {
        quote! {}
    }
}

/// See [generate_repr_c_struct_layout]
pub(crate) struct ReprCStructLayout {
    ///
    pub(crate) layout_struct: syn::ItemStruct,
    /// A constant expression of the same type as `layout_struct`.
    pub(crate) layout_const_fn: TokenStream,
}

/// Utility function to generate a set of constant items containing the offsets and sizes of each
/// field of a repr(C) struct.
pub(crate) fn generate_repr_c_struct_layout(
    derive_input: &syn::DeriveInput,
    vis: &syn::Visibility,
) -> Result<ReprCStructLayout, syn::Error> {
    let fields = match derive_input.data {
        syn::Data::Struct(ref s) => &s.fields,
        _ => {
            return Err(syn::Error::new(
                derive_input.span(),
                "Expected a struct item",
            ))
        }
    };

    // --- offset and size ---
    let layout_const_fn_stmts: Vec<_> = fields
        .iter()
        .enumerate()
        .map(|(i, f)| {
            let field_ty = &f.ty;
            if i == 0 {
                quote! {
                    offsets[0] = 0;
                    sizes[0] = ::std::mem::size_of::<#field_ty>();
                }
            } else {
                let i0 = i - 1;
                let i1 = i;
                quote! {
                    offsets[#i1] =
                        (offsets[#i0]+sizes[#i0])
                        + (::std::mem::align_of::<#field_ty>() -
                                (offsets[#i0]+sizes[#i0])
                                    % ::std::mem::align_of::<#field_ty>())
                          % ::std::mem::align_of::<#field_ty>();
                    sizes[#i1] = ::std::mem::size_of::<#field_ty>();
                }
            }
        })
        .collect();

    // `<vis> struct __MyStruct_StructLayout { <fields...> }`
    let layout_struct_name = Ident::new(
        &format!("__{}_StructLayout", derive_input.ident.to_string()),
        Span::call_site(),
    );

    let layout_struct: ItemStruct = match fields {
        Fields::Named(_) => {
            // named fields: `struct StructLayout { field_name: FieldLayout ... }`
            let layout_fields = fields.iter().map(|f| {
                let name = f.ident.as_ref().unwrap();
                quote! { #name: #CRATE::utils::FieldLayout }
            });
            syn::parse_quote! {
                #[doc(hidden)]
                #vis struct #layout_struct_name {
                    #(#layout_fields,)*
                }
            }
        }
        Fields::Unnamed(_) => {
            // tuple-like: `struct StructLayout(FieldLayout, ...)`
            let types = fields.iter().map(|_| quote! { #CRATE::utils::FieldLayout });
            syn::parse_quote! {
                #[doc(hidden)]
                #vis struct #layout_struct_name(#(#types,)*);
            }
        }
        Fields::Unit => {
            // empty
            syn::parse_quote! {
                #[doc(hidden)]
                #vis struct #layout_struct_name {};
            }
        }
    };
    let layout_struct_init = fields.iter().enumerate().map(|(i, f)| {
        if let Some(ident) = &f.ident {
            quote! { #ident: #CRATE::utils::FieldLayout { offset: offsets[#i], size: sizes[#i] } }
        } else {
            let index = syn::Index::from(i);
            quote! { #index: #CRATE::utils::FieldLayout { offset: offsets[#i], size: sizes[#i] } }
        }
    });

    let num_fields = fields.len();

    let layout_const_fn = quote! {
        const fn layout() -> #layout_struct_name {
            let mut offsets = [0usize; #num_fields];
            let mut sizes = [0usize; #num_fields];
            #(#layout_const_fn_stmts)*
            #layout_struct_name { #(#layout_struct_init,)* }
        }
    };

    Ok(ReprCStructLayout {
        layout_struct,
        layout_const_fn,
    })
}

// Not exactly a derive, but adds an inherent impl block with a `LAYOUT` associated constant.
pub fn derive(input: proc_macro::TokenStream) -> TokenStream {
    let derive_input: syn::DeriveInput = match syn::parse(input) {
        Ok(input) => input,
        Err(e) => return e.into_compile_error(),
    };

    // check for struct
    let fields = match derive_input.data {
        syn::Data::Struct(ref struct_data) => &struct_data.fields,
        _ => {
            return syn::Error::new(
                derive_input.span(),
                "`StructLayout` can only be derived on structs",
            )
            .into_compile_error()
        }
    };

    // check for `#[repr(C)]`
    let repr_c_check = if !has_repr_c_attr(&derive_input) {
        syn::Error::new(
            derive_input.span(),
            format!("`StructLayout` can only be derived on `repr(C)` structs"),
        )
        .into_compile_error()
    } else {
        quote! {}
    };

    // generate field offset constant items
    let struct_layout =
        match generate_repr_c_struct_layout(&derive_input, &syn::Visibility::Inherited) {
            Ok(struct_layout) => struct_layout,
            Err(e) => return e.into_compile_error(),
        };

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();
    let layout_struct = &struct_layout.layout_struct;
    let layout_const_fn = &struct_layout.layout_const_fn;
    let vis = &derive_input.vis;

    quote! {
        #repr_c_check
        #layout_struct
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #layout_const_fn
        }
    }
}
