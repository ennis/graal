use crate::{ensure_repr_c, generate_field_offsets_and_sizes, has_repr_c_attr, FieldList, G};
use darling::{
    util::{Flag, SpannedValue},
    FromDeriveInput, FromField, FromMeta,
};
use proc_macro2::{Span, TokenStream};
use quote::quote;
use syn::spanned::Spanned;

#[derive(FromDeriveInput, Debug)]
#[darling(forward_attrs(allow, doc, cfg, repr))]
struct DescriptorStruct {
    ident: syn::Ident,
    generics: syn::Generics,
    vis: syn::Visibility,
    attrs: Vec<syn::Attribute>,
}

#[derive(Default, FromMeta)]
#[darling(default)]
struct StagesMeta {
    all_graphics: Flag,
    compute: Flag,
    vertex: Flag,
    fragment: Flag,
    geometry: Flag,
    tessellation_control: Flag,
    tessellation_evaluation: Flag,
}

#[derive(Default, FromMeta)]
struct RuntimeArrayMeta {
    max_count: u32,
}

#[derive(FromField)]
#[darling(attributes(layout))]
struct LayoutAttr {
    #[darling(default)]
    binding: u32,
    #[darling(default)]
    runtime_array: Option<SpannedValue<RuntimeArrayMeta>>,
    #[darling(default)]
    array: SpannedValue<Flag>,
    #[darling(default)]
    stages: Option<SpannedValue<StagesMeta>>,
    #[darling(default)]
    sampler: Flag,
    #[darling(default)]
    sampled_image: Flag,
    #[darling(default)]
    storage_image: Flag,
    #[darling(default)]
    uniform_buffer: Flag,
    #[darling(default)]
    uniform_buffer_dynamic: Flag,
    #[darling(default)]
    storage_buffer: Flag,
    #[darling(default)]
    storage_buffer_dynamic: Flag,
}

enum DescriptorClass {
    Buffer,
    Image,
    TexelBufferView,
}

pub fn generate(derive_input: &syn::DeriveInput, fields: &FieldList) -> TokenStream {
    let s: DescriptorStruct =
        <DescriptorStruct as FromDeriveInput>::from_derive_input(derive_input).unwrap();
    let struct_name = &s.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

    if let Err(e) = ensure_repr_c("DescriptorSetInterface", derive_input) {
        return e;
    }

    let field_offsets_sizes = generate_field_offsets_and_sizes(derive_input);

    enum DescriptorType {
        Sampler,
        SampledImage,
        StorageImage,
        UniformBuffer,
        UniformBufferDynamic,
        StorageBuffer,
        StorageBufferDynamic,
    }

    let mut binding_infos = Vec::new();
    let mut update_template_entries = Vec::new();
    let mut runtime_array_writes = Vec::new();

    for (i_field, f) in fields.iter().enumerate() {
        let ty = &f.ty;
        let name = f.ident.as_ref().unwrap();

        match <LayoutAttr as FromField>::from_field(f) {
            Ok(attr) => {
                let attr: &LayoutAttr = &attr;

                // check stages meta
                let stages = if let Some(stages) = &attr.stages {
                    let mut flags = Vec::new();
                    if stages.all_graphics.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::ALL_GRAPHICS});
                    }
                    if stages.vertex.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::VERTEX});
                    }
                    if stages.fragment.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::FRAGMENT});
                    }
                    if stages.geometry.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::GEOMETRY});
                    }
                    if stages.tessellation_control.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::TESSELLATION_CONTROL});
                    }
                    if stages.tessellation_evaluation.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::TESSELLATION_EVALUATION});
                    }
                    if stages.compute.is_some() {
                        flags.push(quote! {#G::vk::ShaderStageFlags::COMPUTE});
                    }

                    if flags.is_empty() {
                        stages.span().unwrap().error("No shader stage specified.")
                            .note("Expected one or more of `all_graphics`, `compute`, `vertex`, `fragment`, `geometry`, `tessellation_control`, `tessellation_evaluation`")
                            .emit();
                        continue;
                    }

                    quote! {
                        #G::vk::ShaderStageFlags::from_raw(0 #(| #flags.as_raw())*)
                    }
                } else {
                    quote! { #G::vk::ShaderStageFlags::from_raw(#G::vk::ShaderStageFlags::ALL_GRAPHICS.as_raw() | #G::vk::ShaderStageFlags::COMPUTE.as_raw()) }
                };

                let binding = attr.binding;

                let mut n = 0;
                let mut descriptor_type = None;

                if attr.sampler.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::Sampler);
                }
                if attr.sampled_image.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::SampledImage);
                }
                if attr.storage_image.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::StorageImage);
                }
                if attr.uniform_buffer.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::UniformBuffer);
                }
                if attr.uniform_buffer_dynamic.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::UniformBufferDynamic);
                }
                if attr.storage_buffer.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::StorageBuffer);
                }
                if attr.storage_buffer_dynamic.is_some() {
                    n += 1;
                    descriptor_type = Some(DescriptorType::StorageBufferDynamic);
                }

                static DESCRIPTOR_TYPE_EXPECTED_TOKENS: &str = "Expected exactly one of `sampled_image`, `storage_image`, `uniform_buffer`, `uniform_buffer_dynamic`, `storage_buffer`, `storage_buffer_dynamic`.";

                if n == 0 {
                    f.span()
                        .unwrap()
                        .error("No descriptor type specified.")
                        .note(DESCRIPTOR_TYPE_EXPECTED_TOKENS)
                        .emit();
                    continue;
                }
                if n > 1 {
                    f.span()
                        .unwrap()
                        .error("More than one descriptor type specified.")
                        .note(DESCRIPTOR_TYPE_EXPECTED_TOKENS)
                        .emit();
                    continue;
                }

                let descriptor_type = descriptor_type.unwrap();

                let descriptor_type_tokens = match descriptor_type {
                    DescriptorType::Sampler => {
                        quote! { #G::vk::DescriptorType::SAMPLER }
                    }
                    DescriptorType::SampledImage => {
                        quote! { #G::vk::DescriptorType::SAMPLED_IMAGE }
                    }
                    DescriptorType::StorageImage => {
                        quote! { #G::vk::DescriptorType::STORAGE_IMAGE }
                    }
                    DescriptorType::UniformBuffer => {
                        quote! { #G::vk::DescriptorType::UNIFORM_BUFFER }
                    }
                    DescriptorType::UniformBufferDynamic => {
                        quote! { #G::vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC }
                    }
                    DescriptorType::StorageBuffer => {
                        quote! { #G::vk::DescriptorType::STORAGE_BUFFER }
                    }
                    DescriptorType::StorageBufferDynamic => {
                        quote! { #G::vk::DescriptorType::STORAGE_BUFFER_DYNAMIC }
                    }
                };

                if let Some(runtime_array) = &attr.runtime_array {
                    let max_count = runtime_array.max_count;
                    binding_infos.push(quote! {
                        #G::DescriptorSetLayoutBindingInfo {
                            binding           : #binding,
                            stage_flags       : #stages,
                            descriptor_type   : #descriptor_type_tokens,
                            descriptor_count  : #max_count,
                            immutable_samplers: [#G::vk::Sampler::null(); 16]
                        }
                    });
                    let p_image_info = match descriptor_type {
                        DescriptorType::Sampler
                        | DescriptorType::SampledImage
                        | DescriptorType::StorageImage => {
                            quote! { self.#name.as_ptr() } // TODO use DescriptorSource?
                        }
                        _ => quote! {::std::ptr::null()},
                    };
                    let p_buffer_info = match descriptor_type {
                        DescriptorType::UniformBuffer
                        | DescriptorType::UniformBufferDynamic
                        | DescriptorType::StorageBuffer
                        | DescriptorType::StorageBufferDynamic => {
                            quote! { self.#name.as_ptr() } // TODO use DescriptorSource?
                        }
                        _ => quote! {::std::ptr::null()},
                    };
                    let p_texel_buffer_view = quote! {::std::ptr::null()};

                    runtime_array_writes.push(quote! {
                        #G::vk::WriteDescriptorSet {
                            dst_set             : set,   // hardcoded
                            dst_binding         : #binding,
                            dst_array_element   : 0,
                            descriptor_count    : self.#name.len() as u32,  // TODO use DescriptorSource?
                            descriptor_type     : #descriptor_type_tokens,
                            p_image_info        : #p_image_info,
                            p_buffer_info       : #p_buffer_info,
                            p_texel_buffer_view : #p_texel_buffer_view,
                            .. Default::default()
                        }
                    });
                } else {
                    binding_infos.push(quote! {
                        #G::DescriptorSetLayoutBindingInfo {
                            binding           : #binding,
                            stage_flags       : #stages,
                            descriptor_type   : #descriptor_type_tokens,
                            descriptor_count  : <#ty as #G::DescriptorSource>::ARRAY_SIZE,
                            immutable_samplers: [#G::vk::Sampler::null(); 16]
                        }
                    });
                    let offset_ident = &field_offsets_sizes.offsets[i_field].ident;
                    let size_ident = &field_offsets_sizes.sizes[i_field].ident;
                    update_template_entries.push(quote! {
                        #G::vk::DescriptorUpdateTemplateEntry {
                            dst_binding       : #binding,
                            dst_array_element : 0,
                            descriptor_count  : <#ty as #G::DescriptorSource>::ARRAY_SIZE,
                            descriptor_type   : #descriptor_type_tokens,
                            offset            : Self::#offset_ident,
                            stride            : Self::#size_ident
                        }
                    });
                }
            }
            Err(e) => {
                e.write_errors();
            }
        }
    }

    let field_offsets_sizes_impl = field_offsets_sizes.impl_block;

    let q = quote! {
        #field_offsets_sizes_impl

        impl #impl_generics #G::DescriptorSetInterface for #struct_name #ty_generics #where_clause {
            const LAYOUT: &'static [#G::DescriptorSetLayoutBindingInfo] = &[ #(#binding_infos,)* ];
            const UPDATE_TEMPLATE_ENTRIES: &'static [#G::vk::DescriptorUpdateTemplateEntry] = &[#(#update_template_entries,)*];

            fn get_or_init_layout(init: impl FnOnce() -> #G::DescriptorSetAllocatorId) -> #G::DescriptorSetAllocatorId {
                static LAYOUT_ID: #G::internal::OnceCell<#G::DescriptorSetAllocatorId> = #G::internal::OnceCell::new();
                *LAYOUT_ID.get_or_init(init)
            }

            unsafe fn update_descriptors(
                &self,
                device: &#G::ash::Device,
                set: #G::vk::DescriptorSet,
                update_template: #G::vk::DescriptorUpdateTemplate)
            {
                use #G::ash::version::DeviceV1_0;
                use #G::ash::version::DeviceV1_1;
                if !Self::UPDATE_TEMPLATE_ENTRIES.is_empty() {
                    device.update_descriptor_set_with_template(set, update_template, self as *const _ as *const ::std::ffi::c_void);
                }
                // write runtime-sized arrays
                let descriptor_writes = &[#(#runtime_array_writes,)*];
                if !descriptor_writes.is_empty() {
                    device.update_descriptor_sets(descriptor_writes, &[]);
                }
            }
        }
    };
    q
}
