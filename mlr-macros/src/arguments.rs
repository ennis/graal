use crate::{expect_struct_fields, CRATE};
use darling::usage::{CollectTypeParams, GenericsExt, Purpose};
use proc_macro2::{Ident, Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;

pub(crate) fn derive_arguments(input: proc_macro::TokenStream) -> syn::Result<TokenStream> {
    let derive_input: syn::DeriveInput = syn::parse(input)?;
    let fields = expect_struct_fields(&derive_input, "Arguments")?;

    let data_member_name = |index: usize, ident: &Option<Ident>| {
        if let Some(ref ident) = ident {
            ident.clone()
        } else {
            Ident::new(&format!("__field_{}", index), Span::call_site())
        }
    };

    // `DescriptorSetLayoutBinding { .. }, DescriptorSetLayoutBinding { .. }, ...`
    let mut bindings = vec![];

    // `field: type, field2: type, ...`
    // or
    // `__field_0: type, __field_1: type, ...`

    let mut inline_data_members = vec![];
    // `field: self.field, field2: self.field2`
    // or
    // `__field_0: self.field, __field_1: self.field, ...`
    let mut inline_data_member_init = vec![];

    // `type, type, ...`
    let mut inline_data_member_types = vec![];

    // `visitor.visit(binding0, &self.field); visitor.visit(binding1, &self.field2); ...`
    let mut argument_infos = vec![];

    let mut next_binding_index: u32 = 0;
    let mut has_binding_zero = false;

    for (i, f) in fields.iter().enumerate() {
        let member = if let Some(ref ident) = f.ident {
            syn::Member::from(ident.clone())
        } else {
            syn::Member::from(i)
        };

        //let mut is_argument = false;
        let mut is_binding = false;
        let mut binding_index: Option<u32> = None;

        for attr in f.attrs.iter() {
            if attr.path().is_ident("argument") {
                is_binding = true;
                /*if attr.meta.require_path_only().is_ok() {
                    continue;
                }*/
                attr.meta.require_list()?.parse_nested_meta(|meta| {
                    if meta.path.is_ident("binding") {
                        let value = meta.value()?;
                        let index: syn::LitInt = value.parse()?;
                        binding_index = Some(index.base10_parse()?);
                        return Ok(());
                    }

                    /*if meta.path.is_ident("inline") {
                        return Ok(());
                    }*/

                    /*if meta.path.is_ident("...") {
                       let value = meta.value()?;
                       let f: syn::Ident = value.parse()?;
                       ...
                       return Ok(());
                    */

                    Err(meta.error("invalid syntax for `#[argument]`"))
                })?;
            }
        }

        // determine binding index
        if is_binding {
            let binding_index = binding_index.unwrap_or(next_binding_index);
            if binding_index < next_binding_index {
                return Err(syn::Error::new(
                    f.span(),
                    format!(
                        "binding index must be increasing: last binding index was {}",
                        next_binding_index - 1
                    ),
                ));
            }
            next_binding_index = binding_index + 1;

            // descriptor binding
            if binding_index == 0 {
                has_binding_zero = true;
            }
            let ty = &f.ty;

            bindings.push(quote! {
                #CRATE::vk::DescriptorSetLayoutBinding {
                    binding              : #binding_index,
                    stage_flags          : <#ty as #CRATE::argument::DescriptorBinding>::SHADER_STAGES,
                    descriptor_type      : <#ty as #CRATE::argument::DescriptorBinding>::DESCRIPTOR_TYPE,
                    descriptor_count     : <#ty as #CRATE::argument::DescriptorBinding>::DESCRIPTOR_COUNT,
                    p_immutable_samplers : ::std::ptr::null()
                },
            });
            argument_infos.push(quote! {
                #CRATE::argument::Argument::argument_info(&self.#member, #binding_index);
            });
        } else {
            let ty = &f.ty;
            let data_member_name = data_member_name(i, &f.ident);

            inline_data_members.push(quote! {
                #data_member_name: #ty,
            });
            inline_data_member_init.push(quote! {
                #data_member_name: self.#member,
            });
            inline_data_member_types.push(ty);
        }
    }

    // statements to visit inline data in Arguments::visit
    let return_inline_data;
    let inline_data_struct;

    if !inline_data_members.is_empty() {
        if has_binding_zero {
            return Err(syn::Error::new(
                derive_input.span(),
                "binding number 0 conflicts with inline data members",
            ));
        }

        let inline_data_struct_name = quote::format_ident!("__{}_InlineData", derive_input.ident);

        // Add binding 0 entry to set layout
        bindings.insert(
            0,
            quote! {
                #CRATE::vk::DescriptorSetLayoutBinding {
                    binding              : 0,
                    stage_flags          : #CRATE::vk::ShaderStageFlags::ALL,
                    descriptor_type      : #CRATE::vk::DescriptorType::INLINE_UNIFORM_BLOCK,
                    descriptor_count     : ::std::mem::size_of::<#inline_data_struct_name>() as u32,
                    p_immutable_samplers : ::std::ptr::null()
                },
            },
        );

        //  Collect generics of each inline data field
        let type_params = derive_input.generics.declared_type_params();
        let inline_data_type_param_idents =
            inline_data_member_types.collect_type_params(&Purpose::Declare.into(), &type_params);
        let inline_data_type_params: Vec<_> = inline_data_type_param_idents
            .iter()
            .map(|x| derive_input.generics.type_params().find(|tp| &tp.ident == *x).unwrap())
            .collect();

        inline_data_struct = quote! {
            #[repr(C)]
            #[derive(Copy,Clone)]
            struct #inline_data_struct_name<#(#inline_data_type_params,)*> {
                #(#inline_data_members)*
            }
        };

        return_inline_data = quote! {
            let inline_data = #inline_data_struct_name {
                #(#inline_data_member_init)*
            };
            inline_data
        };
    } else {
        inline_data_struct = quote! { () };
        return_inline_data = quote! {
            ::std::borrow::Cow::Borrowed(&())
        };
    }

    // Check that we have no generics (doesn't work with DescriptorSetLayout caching)
    if derive_input.generics.type_params().count() != 0 || derive_input.generics.const_params().count() != 0 {
        return Err(syn::Error::new(
            derive_input.span(),
            "generic parameters are not supported on structs with `#[derive(Arguments)]`",
        ));
    }

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = derive_input.generics.split_for_impl();

    let arguments_seq = 0..argument_infos.len();

    Ok(quote! {
        #inline_data_struct

        impl #impl_generics #CRATE::argument::StaticArguments for #struct_name #ty_generics #where_clause {
            const LAYOUT: #CRATE::argument::ArgumentsLayout<'static> = #CRATE::argument::ArgumentsLayout {
                bindings: ::std::borrow::Cow::Borrowed(&[
                    #(#bindings)*
                ]),
            };

            fn create_descriptor_set_layout(device: &#CRATE::device::Device) -> #CRATE::vk::DescriptorSetLayout {
                static CACHED: ::std::sync::OnceLock<#CRATE::vk::DescriptorSetLayout> = ::std::sync::OnceLock::new();
                CACHED.get_or_init(move || {
                    #CRATE::argument::create_descriptor_set_layout(device, &Self::LAYOUT)
                }).clone()
            }
        }

        impl #impl_generics #CRATE::argument::Arguments for #struct_name #ty_generics #where_clause {

            /// The type of inline data for this argument.
            type InlineData = #inline_data_struct;

            /// Returns an iterator over all descriptors contained in this object.
            fn arguments(&self) -> impl Iterator<Item = ArgumentInfo> + '_ {
                let mut index = 0;
                std::iter::from_fn(move || {
                    let r = match index {
                        #(#arguments_seq => Some(#argument_infos),)*
                        _ => None,
                    };
                    index += 1;
                    r
                })
            }

            /// Returns the inline data for this argument.
            fn inline_data(&self) -> Cow<Self::InlineData> {
                #return_inline_data
            }
        }
    })
}
