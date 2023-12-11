use crate::{expect_struct_fields, StageFlags, CRATE};
use proc_macro2::TokenStream;
use quote::{quote, TokenStreamExt};

pub(crate) fn derive_push_constants(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
    let derive_input: syn::DeriveInput = syn::parse(input)?;
    let fields = expect_struct_fields(&derive_input, "PushConstants")?;
    let struct_name = &derive_input.ident;

    let make_push_constant_range = |stage_flags: &StageFlags, fields: &[(syn::Member, &syn::Type)]| -> TokenStream {
        let (first_member, _) = fields.first().unwrap();
        let (last_member, last_ty) = fields.last().unwrap();

        let mut tokens = TokenStream::new();
        match first_member {
            syn::Member::Named(_) => {
                tokens.append_all(quote! {
                    let start_offset = #CRATE::__offset_of!(#struct_name, #first_member);
                    let end_offset = #CRATE::__offset_of!(#struct_name, #last_member) + ::std::mem::size_of::<#last_ty>();
                });
            }
            syn::Member::Unnamed(_) => {
                tokens.append_all(quote! {
                    let start_offset = #CRATE::__offset_of_tuple!(#struct_name, #first_member);
                    let end_offset = #CRATE::__offset_of_tuple!(#struct_name, #last_member) + ::std::mem::size_of::<#last_ty>();
                });
            }
        }
        tokens.append_all(quote! {
            assert!(start_offset % 4 == 0, "push constant offset must be a multiple of 4");
            assert!(end_offset % 4 == 0, "push constant size must be a multiple of 4");
            #CRATE::vk::PushConstantRange {
                stage_flags: #stage_flags,
                offset: start_offset as u32,
                size: (end_offset - start_offset) as u32,
            }
        });

        quote!({
            #tokens
        },)
    };

    let mut prev_stage_flags = StageFlags::default();
    let mut push_constant_ranges = vec![];
    let mut cur_range = vec![];

    for (i, f) in fields.iter().enumerate() {
        let mut stage_flags = None;
        for attr in f.attrs.iter() {
            if attr.path().is_ident("stages") {
                stage_flags = Some(StageFlags::from_meta_list(attr.meta.require_list()?)?);
            }
        }

        let stage_flags = stage_flags.unwrap_or(StageFlags::all());
        if stage_flags != prev_stage_flags && !cur_range.is_empty() {
            push_constant_ranges.push(make_push_constant_range(&prev_stage_flags, &cur_range));
            cur_range.clear();
        }

        let member = if let Some(ref ident) = f.ident {
            syn::Member::from(ident.clone())
        } else {
            syn::Member::from(i)
        };
        cur_range.push((member, &f.ty));
        prev_stage_flags = stage_flags;
    }

    if !cur_range.is_empty() {
        push_constant_ranges.push(make_push_constant_range(&prev_stage_flags, &cur_range));
    }

    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    Ok(quote! {
        impl #impl_generics #CRATE::StaticPushConstants for #struct_name #ty_generics #where_clause {
            const PUSH_CONSTANT_RANGES: &'static [#CRATE::vk::PushConstantRange] = &[
                #(#push_constant_ranges)*
            ];
        }
    })
}
