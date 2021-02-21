use crate::G;
use darling::util::SpannedValue;
use darling::{util::Flag, FromDeriveInput, FromField, FromMeta};
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

#[derive(FromField)]
#[darling(attributes(layout))]
struct LayoutAttr {
    #[darling(default)]
    max_count: u32,
    #[darling(default)]
    binding: u32,
    #[darling(default)]
    unbounded: Flag,
    #[darling(default)]
    array: Flag,
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

pub fn generate(derive_input: &syn::DeriveInput, fields: &syn::Fields) -> TokenStream {
    let s: DescriptorStruct =
        <DescriptorStruct as FromDeriveInput>::from_derive_input(derive_input).unwrap();
    let struct_name = &s.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();

    let fields = match fields {
        syn::Fields::Named(ref fields_named) => &fields_named.named,
        syn::Fields::Unnamed(ref fields_unnamed) => &fields_unnamed.unnamed,
        syn::Fields::Unit => panic!("`DescriptorSetInterface` cannot be derived on unit structs"),
    };

    enum DescriptorType {
        Sampler,
        SampledImage,
        StorageImage,
        UniformBuffer,
        UniformBufferDynamic,
        StorageBuffer,
        StorageBufferDynamic,
    }

    struct DescriptorCodegenData<'a> {
        image_descriptors: Vec<&'a syn::Ident>,
        buffer_descriptors: Vec<&'a syn::Ident>,
        texel_buffer_view_descriptors: Vec<&'a syn::Ident>,
        field_names: Vec<&'a syn::Ident>,
        image_descriptor_offsets: Vec<usize>,
        buffer_descriptor_offsets: Vec<usize>,
        texel_buffer_view_descriptor_offsets: Vec<usize>,
        image_descriptor_count: usize,
        buffer_descriptor_count: usize,
        texel_buffer_view_descriptor_count: usize,
        binding_infos: Vec<TokenStream>,

        bindings: Vec<u32>,
        descriptor_types: Vec<TokenStream>,
        descriptor_counts: Vec<u32>,
    }

    impl<'a> DescriptorCodegenData<'a> {
        fn new() -> DescriptorCodegenData<'a> {
            DescriptorCodegenData {
                image_descriptors: vec![],
                buffer_descriptors: vec![],
                texel_buffer_view_descriptors: vec![],
                field_names: vec![],
                image_descriptor_offsets: vec![],
                buffer_descriptor_offsets: vec![],
                texel_buffer_view_descriptor_offsets: vec![],
                image_descriptor_count: 0,
                buffer_descriptor_count: 0,
                texel_buffer_view_descriptor_count: 0,
                binding_infos: vec![],
                bindings: vec![],
                descriptor_types: vec![],
                descriptor_counts: vec![]
            }
        }

        fn push(
            &mut self,
            name: &'a syn::Ident,
            descriptor_type: DescriptorType,
            descriptor_count: u32,
            binding: u32,
            stages: &TokenStream,
        ) {
            self.field_names.push(name);
            match descriptor_type {
                DescriptorType::Sampler
                | DescriptorType::SampledImage
                | DescriptorType::StorageImage => {
                    self.image_descriptors.push(name);
                    self.image_descriptor_offsets
                        .push(self.image_descriptor_count);
                    self.buffer_descriptor_offsets.push(0);
                    self.texel_buffer_view_descriptor_offsets.push(0);
                    self.image_descriptor_count += 1; // TODO arrays
                }
                DescriptorType::UniformBuffer
                | DescriptorType::UniformBufferDynamic
                | DescriptorType::StorageBuffer
                | DescriptorType::StorageBufferDynamic => {
                    self.buffer_descriptors.push(name);
                    self.image_descriptor_offsets.push(0);
                    self.buffer_descriptor_offsets
                        .push(self.buffer_descriptor_count);
                    self.texel_buffer_view_descriptor_offsets.push(0);
                    self.buffer_descriptor_count += 1;
                }
            }

            let descriptor_type = match descriptor_type {
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

            self.binding_infos.push(quote! {
                #G::DescriptorSetLayoutBindingInfo {
                    binding           : #binding,
                    stage_flags       : #stages,
                    descriptor_type   : #descriptor_type,
                    descriptor_count  : #descriptor_count,
                    immutable_samplers: [#G::vk::Sampler::null(); 16]
                }
            });

            self.bindings.push(binding);
            self.descriptor_types.push(descriptor_type);
            self.descriptor_counts.push(descriptor_count);
        }
    }

    let mut cg_desc = DescriptorCodegenData::new();

    for f in fields.iter() {
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
                // TODO
                let descriptor_count = attr.max_count;

                let mut n = 0;

                if attr.sampler.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::Sampler,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.sampled_image.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::SampledImage,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.storage_image.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::StorageImage,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.uniform_buffer.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::UniformBuffer,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.uniform_buffer_dynamic.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::UniformBufferDynamic,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.storage_buffer.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::StorageBuffer,
                        descriptor_count,
                        binding,
                        &stages,
                    );
                }
                if attr.storage_buffer_dynamic.is_some() {
                    n += 1;
                    cg_desc.push(
                        name,
                        DescriptorType::StorageBufferDynamic,
                        descriptor_count,
                        binding,
                        &stages,
                    );
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
            }
            Err(e) => {
                e.write_errors();
            }
        }
    }

    let binding_infos = &cg_desc.binding_infos;
    let bindings_count = cg_desc.binding_infos.len();

    let field_names = cg_desc.field_names;
    let image_descriptor_count = cg_desc.image_descriptor_count;
    let buffer_descriptor_count = cg_desc.buffer_descriptor_count;
    let texel_buffer_view_descriptor_count = cg_desc.texel_buffer_view_descriptor_count;

    let image_descriptor_offsets = cg_desc.image_descriptor_offsets;
    let buffer_descriptor_offsets = cg_desc.buffer_descriptor_offsets;
    let texel_buffer_view_descriptor_offsets = cg_desc.texel_buffer_view_descriptor_offsets;

    let bindings = cg_desc.bindings;
    let descriptor_counts = cg_desc.descriptor_counts;
    let descriptor_types = cg_desc.descriptor_types;

    let q = quote! {
        impl #impl_generics #G::DescriptorSetInterface for #struct_name #ty_generics #where_clause {
            const LAYOUT: &'static [#G::DescriptorSetLayoutBindingInfo] = &[ #(#binding_infos,)* ];

            unsafe fn write_descriptors(&self, device: &#G::ash::Device, set: #G::vk::DescriptorSet) {
                //
                let mut image_infos = [#G::vk::DescriptorImageInfo::default(); #image_descriptor_count];
                let mut buffer_infos = [#G::vk::DescriptorBufferInfo::default(); #buffer_descriptor_count];
                let mut texel_buffer_views = [#G::vk::BufferView::default(); #texel_buffer_view_descriptor_count];

                use graal::DescriptorSource;
                #( self.#field_names.write_descriptors(
                        &mut image_infos[#image_descriptor_offsets..],
                        &mut buffer_infos[#buffer_descriptor_offsets..],
                        &mut texel_buffer_views[#texel_buffer_view_descriptor_offsets..]); )*

                let descriptor_writes = [
                    # ( graal::vk::WriteDescriptorSet {
                        dst_set: set,
                        dst_binding: #bindings,
                        dst_array_element: 0,
                        descriptor_count: #descriptor_counts, // TODO
                        descriptor_type: #descriptor_types,
                        p_image_info: image_infos[#image_descriptor_offsets..].as_ptr(),
                        p_buffer_info: buffer_infos[#buffer_descriptor_offsets..].as_ptr(),
                        p_texel_buffer_view: texel_buffer_views[#texel_buffer_view_descriptor_offsets..].as_ptr(),
                        .. Default::default()
                    }, )*
                ];
            }
        }
    };
    q
}
