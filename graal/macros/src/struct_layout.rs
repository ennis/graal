use proc_macro2::{Span, TokenStream};
use syn::spanned::Spanned;
use syn::Ident;
use quote::quote;

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

pub(crate) fn ensure_repr_c(derive_name: &str, ast: &syn::DeriveInput) -> Result<(), TokenStream> {
    if !has_repr_c_attr(ast) {
        ast.span()
            .unwrap()
            .error(format!(
                "`{}` can only be derived on `repr(C)` structs",
                derive_name
            ))
            .help("add `#[repr(C)]` attribute to the struct")
            .emit();
        Err(Default::default())
    } else {
        Ok(())
    }
}

/// See [generate_struct_layout]
pub(crate) struct FieldOffsetsAndSizes {
    pub(crate) impl_block: TokenStream,
    pub(crate) offsets: Vec<syn::ItemConst>,
    pub(crate) sizes: Vec<syn::ItemConst>,
}

/// Utility function to generate a set of constant items containing the offsets and sizes of each
/// field of a repr(C) struct.
pub(crate) fn generate_field_offsets_and_sizes(
    derive_input: &syn::DeriveInput,
) -> FieldOffsetsAndSizes {
    let fields = match derive_input.data {
        syn::Data::Struct(ref s) => &s.fields,
        _ => panic!("not a struct"),
    };

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    let fields = match *fields {
        syn::Fields::Named(ref fields_named) => &fields_named.named,
        syn::Fields::Unnamed(ref fields_unnamed) => &fields_unnamed.unnamed,
        syn::Fields::Unit => panic!("cannot generate struct layout of unit structs"),
    };

    let mut offsets = Vec::new();
    let mut sizes = Vec::new();
    let mut offset_idents = Vec::new();
    let mut size_idents = Vec::new();

    for (i, f) in fields.iter().enumerate() {
        let field_ty = &f.ty;

        // field offset item
        if i == 0 {
            offsets.push(syn::parse_quote! { pub const OFFSET_0: usize = 0; });
            sizes.push(
                syn::parse_quote! { pub const SIZE_0: usize = ::std::mem::size_of::<#field_ty>(); },
            );
            offset_idents.push(Ident::new("OFFSET_0", Span::call_site()));
            size_idents.push(Ident::new("SIZE_0", Span::call_site()));
        } else {
            let offset0 = &offset_idents[i - 1];
            let offset1 = Ident::new(&format!("OFFSET_{}", i), Span::call_site());
            let size0 = &size_idents[i - 1];
            let size1 = Ident::new(&format!("SIZE_{}", i), Span::call_site());

            offsets.push(syn::parse_quote! {
                pub const #offset1: usize =
                    (Self::#offset0+Self::#size0)
                    + (::std::mem::align_of::<#field_ty>() -
                            (Self::#offset0+Self::#size0)
                                % ::std::mem::align_of::<#field_ty>())
                      % ::std::mem::align_of::<#field_ty>();
            });
            sizes.push(syn::parse_quote! {
                 pub const #size1: usize = ::std::mem::size_of::<#field_ty>();
            });

            offset_idents.push(offset1);
            size_idents.push(size1);
        };
    }

    let impl_block = quote! {
        impl #impl_generics #struct_name #ty_generics #where_clause {
            #(#offsets)*
            #(#sizes)*
        }
    };

    FieldOffsetsAndSizes {
        impl_block,
        offsets,
        sizes,
    }
}
