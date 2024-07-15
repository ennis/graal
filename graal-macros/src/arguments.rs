use crate::{expect_struct_fields, CRATE};
use darling::usage::{CollectTypeParams, GenericsExt, Purpose};
use proc_macro2::{Ident, Span, TokenStream};
use quote::{quote, ToTokens};
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
        let mut read_only = false;
        let mut read_write = false;
        let mut storage = false;
        let mut storage_image = false;
        let mut uniform = false;
        let mut sampled_image = false;
        let mut sampler = false;

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

                    if meta.path.is_ident("read_only") {
                        read_only = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("read_write") {
                        read_write = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("storage") {
                        storage = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("storage_image") {
                        storage_image = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("uniform") {
                        uniform = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("sampled_image") {
                        sampled_image = true;
                        return Ok(());
                    }

                    if meta.path.is_ident("sampler") {
                        sampler = true;
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

        if read_only && read_write {
            return Err(syn::Error::new(
                f.span(),
                "`read_only` and `read_write` are mutually exclusive",
            ));
        }
        // default to read only access
        if !(read_only || read_write) {
            read_only = true;
        }

        if [storage, storage_image, uniform, sampled_image, sampler]
            .iter()
            .filter(|x| **x)
            .count()
            > 1
        {
            return Err(syn::Error::new(
                f.span(),
                "`storage`, `storage_image`, `uniform`, `sampled_image` and `sampler` are mutually exclusive",
            ));
        }

        if storage || storage_image || uniform || sampled_image || sampler {
            is_binding = true;
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
            //let ty = &f.ty;

            let mut descriptor_type = quote!();
            let mut access = quote!();
            //let shader_stages: u32 = 0xFF;

            /*
            /// The index buffer used for drawing.
            const INDEX = 1 << 3;
            /// A vertex buffer used for drawing.
            const VERTEX = 1 << 4;
            /// A uniform buffer bound in a bind group.
            const UNIFORM = 1 << 5;
            /// The indirect or count buffer in a indirect draw or dispatch.
            const INDIRECT = 1 << 6;
            /// The argument to a read-only mapping.
            const MAP_READ = 1 << 7;
            /// The argument to a write-only mapping.
            const MAP_WRITE = 1 << 8;
            /// Read-only storage buffer usage. Corresponds to a UAV in d3d, so is exclusive, despite being read only.
            const STORAGE_READ = 1 << 9;
            /// Read-write or write-only storage buffer usage.
            const STORAGE_READ_WRITE = 1 << 10;*/

            // FIXME: we shouldn't use the raw values here; remove this once the values are moved to a separate crate
            if storage_image {
                descriptor_type = quote!(#CRATE::vk::DescriptorType::STORAGE_IMAGE);
                if read_only {
                    access = quote!(#CRATE::ImageAccess::IMAGE_READ);
                } else if read_write {
                    access = quote!(#CRATE::ImageAccess::IMAGE_READ_WRITE);
                }
            } else if sampled_image {
                descriptor_type = quote!(#CRATE::vk::DescriptorType::SAMPLED_IMAGE);
                access = quote!(#CRATE::ImageAccess::SAMPLED_READ);
            } else if uniform {
                descriptor_type = quote!(#CRATE::vk::DescriptorType::UNIFORM_BUFFER);
                access = quote!(#CRATE::BufferAccess::UNIFORM);
            } else if storage {
                descriptor_type = quote!(#CRATE::vk::DescriptorType::STORAGE_BUFFER);
                if read_only {
                    access = quote!(#CRATE::BufferAccess::STORAGE_READ);
                } else if read_write {
                    access = quote!(#CRATE::BufferAccess::STORAGE_READ_WRITE);
                }
            } else if sampler {
                descriptor_type = quote!(#CRATE::vk::DescriptorType::SAMPLER);
            } else {
                return Err(syn::Error::new(
                    f.span(),
                    "invalid argument type; must be `storage`, `storage_image`, `uniform`, `sampled_image` or `sampler`",
                ));
            }

            bindings.push(quote! {
                #CRATE::vk::DescriptorSetLayoutBinding {
                    binding              : #binding_index,
                    stage_flags          : #CRATE::vk::ShaderStageFlags::ALL,
                    descriptor_type      : #descriptor_type,
                    descriptor_count     : 1,   // TODO: arrays
                    p_immutable_samplers : ::std::ptr::null()
                },
            });

            if storage_image || sampled_image || uniform || storage {
                argument_infos.push(quote!(#CRATE::ArgumentDescription {
                    binding: #binding_index,
                    descriptor_type: #descriptor_type,
                    kind: #CRATE::ArgumentKind::from((self.#member, #access)),
                }));
            } else if sampler {
                argument_infos.push(quote!(#CRATE::ArgumentDescription {
                    binding: #binding_index,
                    descriptor_type: #descriptor_type,
                    kind: #CRATE::ArgumentKind::from(self.#member),
                }));
            };
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
    let inline_data_struct_ty;

    if !inline_data_members.is_empty() {
        if has_binding_zero {
            return Err(syn::Error::new(
                derive_input.span(),
                "binding number 0 conflicts with inline data members",
            ));
        }

        // FIXME: generics
        inline_data_struct_ty = quote::format_ident!("__{}_InlineData", derive_input.ident).to_token_stream();

        // Add binding 0 entry to set layout
        bindings.insert(
            0,
            quote! {
                #CRATE::vk::DescriptorSetLayoutBinding {
                    binding              : 0,
                    stage_flags          : #CRATE::vk::ShaderStageFlags::ALL,
                    descriptor_type      : #CRATE::vk::DescriptorType::INLINE_UNIFORM_BLOCK,
                    descriptor_count     : ::std::mem::size_of::<#inline_data_struct_ty>() as u32,
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
            struct #inline_data_struct_ty<#(#inline_data_type_params,)*> {
                #(#inline_data_members)*
            }
        };

        return_inline_data = quote! {
            let inline_data = #inline_data_struct_ty {
                #(#inline_data_member_init)*
            };
            ::std::borrow::Cow::Owned(inline_data)
        };
    } else {
        inline_data_struct = quote! {};
        inline_data_struct_ty = quote! { () };
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

        impl #impl_generics #CRATE::StaticArguments for #struct_name #ty_generics #where_clause {
            const LAYOUT: #CRATE::ArgumentsLayout<'static> = #CRATE::ArgumentsLayout {
                bindings: ::std::borrow::Cow::Borrowed(&[
                    #(#bindings)*
                ]),
            };
        }

        impl #impl_generics #CRATE::Arguments for #struct_name #ty_generics #where_clause {

            /// The type of inline data for this argument.
            type InlineData = #inline_data_struct_ty;

            /// Returns an iterator over all descriptors contained in this object.
            fn arguments(&self) -> impl Iterator<Item = #CRATE::ArgumentDescription> + '_ {
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
            fn inline_data(&self) -> ::std::borrow::Cow<Self::InlineData> {
                #return_inline_data
            }
        }
    })
}
