use crate::{
    struct_layout::{ensure_repr_c_derive_input, has_repr_c_attr},
    CRATE,
};
use darling::{
    usage::{CollectTypeParams, GenericsExt, Purpose},
    util::{Flag, SpannedValue},
    FromDeriveInput, FromField, FromMeta,
};
use proc_macro2::{Span, TokenStream};
use quote::{quote, ToTokens, TokenStreamExt};
use syn::{spanned::Spanned, Data, DeriveInput, Fields, GenericParam, Generics, Token};

const VK_DESCRIPTOR_UPDATE_DATA_LEN: usize = 24;
const VK_DESCRIPTOR_IMAGE_INFO_LEN: usize = 24;
const VK_DESCRIPTOR_BUFFER_INFO_LEN: usize = 24;

#[derive(FromDeriveInput, Debug)]
#[darling(forward_attrs(allow, doc, cfg, repr))]
struct ArgumentsStruct {
    ident: syn::Ident,
    generics: syn::Generics,
    vis: syn::Visibility,
    attrs: Vec<syn::Attribute>,
}

/*#[derive(Default, FromMeta)]
#[darling(default)]
struct StagesMeta {
    all_graphics: Flag,
    compute: Flag,
    vertex: Flag,
    fragment: Flag,
    geometry: Flag,
    tessellation_control: Flag,
    tessellation_evaluation: Flag,
}*/

#[derive(Default, FromMeta)]
struct RuntimeArrayMeta {
    max_count: u32,
}

/*#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum ArgumentKind {
    Uniform,
    Sampler,
    SampledImage,
    StorageImage,
    UniformBuffer,
    UniformBufferDynamic,
    StorageBuffer,
    StorageBufferDynamic,
}

impl ArgumentKind {
    fn descriptor_type(&self) -> Option<TokenStream> {
        match self {
            ArgumentKind::Sampler => Some(quote! { #CRATE::vk::DescriptorType::SAMPLER }),
            ArgumentKind::SampledImage => {
                Some(quote! { #CRATE::vk::DescriptorType::SAMPLED_IMAGE })
            }
            ArgumentKind::StorageImage => {
                Some(quote! { #CRATE::vk::DescriptorType::STORAGE_IMAGE })
            }
            ArgumentKind::UniformBuffer => {
                Some(quote! { #CRATE::vk::DescriptorType::UNIFORM_BUFFER })
            }
            ArgumentKind::UniformBufferDynamic => {
                Some(quote! { #CRATE::vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC })
            }
            ArgumentKind::StorageBuffer => {
                Some(quote! { #CRATE::vk::DescriptorType::STORAGE_BUFFER })
            }
            ArgumentKind::StorageBufferDynamic => {
                Some(quote! { #CRATE::vk::DescriptorType::STORAGE_BUFFER_DYNAMIC })
            }
            ArgumentKind::Uniform => None,
        }
    }
}

enum DescriptorWriteType {
    Image,
    Buffer,
}*/

#[derive(FromField)]
#[darling(attributes(argument))]
struct ArgumentAttr {
    #[darling(default)]
    binding: Option<u32>,
    #[darling(default)]
    runtime_array: Option<SpannedValue<RuntimeArrayMeta>>,
    //#[darling(default)]
    //uniform: Flag,
}

/*fn generate_shader_access_mask(stages: &SpannedValue<StagesMeta>) -> TokenStream {
    let mut flags = Vec::new();
    if stages.all_graphics.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::ALL_GRAPHICS});
    }
    if stages.vertex.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::VERTEX});
    }
    if stages.fragment.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::FRAGMENT});
    }
    if stages.geometry.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::GEOMETRY});
    }
    if stages.tessellation_control.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::TESSELLATION_CONTROL});
    }
    if stages.tessellation_evaluation.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::TESSELLATION_EVALUATION});
    }
    if stages.compute.is_some() {
        flags.push(quote! {#CRATE::vk::ShaderStageFlags::COMPUTE});
    }
    if flags.is_empty() {
        syn::Error::new(stages.span(), "No shader stage specified. Expected one or more of `all_graphics`, `compute`, `vertex`, `fragment`, `geometry`, `tessellation_control`, `tessellation_evaluation`.").to_compile_error()
    } else {
        quote! {
            #CRATE::vk::ShaderStageFlags::from_raw(0 #(| #flags.as_raw())*)
        }
    }
}*/

struct ImplGenericsWithoutLifetimesOrBounds<'a>(&'a Generics); // <const N: usize, T: Copy>
struct TypeGenericsWithoutLifetimes<'a>(&'a Generics); // <N,T>

// modified from syn
impl<'a> ToTokens for ImplGenericsWithoutLifetimesOrBounds<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.params.is_empty() {
            return;
        }

        self.0.lt_token.as_ref().unwrap().to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(tokens);
                trailing_or_empty = true;
            }
            match *param.value() {
                GenericParam::Lifetime(_) => unreachable!(),
                GenericParam::Type(param) => {
                    param.ident.to_tokens(tokens);
                }
                GenericParam::Const(param) => {
                    param.const_token.to_tokens(tokens);
                    param.ident.to_tokens(tokens);
                    param.colon_token.to_tokens(tokens);
                    param.ty.to_tokens(tokens);
                }
            }
            param.punct().to_tokens(tokens);
        }

        self.0.gt_token.as_ref().unwrap().to_tokens(tokens);
    }
}

impl<'a> ToTokens for TypeGenericsWithoutLifetimes<'a> {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        if self.0.params.is_empty() {
            return;
        }

        self.0.lt_token.as_ref().unwrap().to_tokens(tokens);
        let mut trailing_or_empty = true;
        for param in self.0.params.pairs() {
            if let GenericParam::Lifetime(_) = **param.value() {
                continue;
            }
            if !trailing_or_empty {
                <Token![,]>::default().to_tokens(tokens);
                trailing_or_empty = true;
            }
            match *param.value() {
                GenericParam::Lifetime(_) => unreachable!(),
                GenericParam::Type(param) => {
                    // Leave off the type parameter defaults
                    param.ident.to_tokens(tokens);
                }
                GenericParam::Const(param) => {
                    // Leave off the const parameter defaults
                    param.ident.to_tokens(tokens);
                }
            }
            param.punct().to_tokens(tokens);
        }

        self.0.gt_token.as_ref().unwrap().to_tokens(tokens);
    }
}

struct Binding<'a> {
    field: &'a syn::Field,
    field_index: usize,
    binding: u32,
}

struct PrimaryUniform<'a> {
    field: &'a syn::Field,
    field_index: usize,
}

pub fn derive(input: proc_macro::TokenStream) -> TokenStream {
    let derive_input: syn::DeriveInput = match syn::parse(input) {
        Ok(input) => input,
        Err(e) => return e.into_compile_error(),
    };

    // check for struct
    let fields = match derive_input.data {
        Data::Struct(ref struct_data) => &struct_data.fields,
        _ => {
            return syn::Error::new(
                derive_input.span(),
                "`Arguments` can only be derived on structs",
            )
            .into_compile_error()
        }
    };

    // check for `#[repr(C)]`
    let repr_c_check = if !has_repr_c_attr(&derive_input) {
        syn::Error::new(
            derive_input.span(),
            format!("`Arguments` can only be derived on `repr(C)` structs"),
        )
        .into_compile_error()
    } else {
        quote! {}
    };

    // parse struct-level `#[arguments(...)]` attribs.
    let s: ArgumentsStruct =
        match <ArgumentsStruct as FromDeriveInput>::from_derive_input(&derive_input) {
            Ok(s) => s,
            Err(e) => return e.write_errors(),
        };

    //let mut runtime_array_writes = Vec::new();
    let mut bindings = Vec::new();
    let mut primary_uniform_fields = Vec::new();
    //let mut default_uniform_buffer_generics = Vec::new();
    let mut attrib_errors = Vec::new();

    // parse field bindings and inline uniform fields
    for (field_index, field) in fields.iter().enumerate() {
        match <ArgumentAttr as FromField>::from_field(field) {
            Ok(attr) => {
                let attr: &ArgumentAttr = &attr;
                // TODO verify attrs
                if let Some(binding) = attr.binding {
                    // this is an actual binding
                    bindings.push(Binding {
                        field,
                        field_index,
                        binding,
                    })
                } else {
                    // assume this is a primary uniform
                    primary_uniform_fields.push(PrimaryUniform { field, field_index });
                }
            }
            Err(e) => attrib_errors.push(e.write_errors()),
        }
    }

    // --- Collect generics of each primary uniform field ---
    let type_params = derive_input.generics.declared_type_params();
    let primary_uniform_type_param_idents = primary_uniform_fields
        .iter()
        .map(|x| x.field)
        .collect_type_params(&Purpose::Declare.into(), &type_params);
    let primary_uniform_type_params: Vec<_> = primary_uniform_type_param_idents
        .iter()
        .map(|x| {
            derive_input
                .generics
                .type_params()
                .find(|tp| &tp.ident == *x)
                .unwrap()
        })
        .collect();

    // --- Generate descriptor set layout bindings ---
    let descriptor_set_layout_bindings: Vec<_> = bindings
        .iter()
        .map(|b| {
            let ty = &b.field.ty;
            let binding = b.binding;
            if binding == 0 && !primary_uniform_fields.is_empty() {
                // binding #0 is reserved for the primary uniform buffer is there's one: ensure it's not used
                syn::Error::new(
                    b.field.span(),
                    "Binding number 0 is reserved for primary uniforms",
                )
                .into_compile_error()
            } else {
                quote! {
                    #CRATE::vk::DescriptorSetLayoutBinding {
                        binding              : #binding,
                        stage_flags          : <#ty as #CRATE::DescriptorBinding>::SHADER_STAGES,
                        descriptor_type      : <#ty as #CRATE::DescriptorBinding>::DESCRIPTOR_TYPE,
                        descriptor_count     : <#ty as #CRATE::DescriptorBinding>::DESCRIPTOR_COUNT,
                        p_immutable_samplers : ::std::ptr::null()
                    }
                }
            }
        })
        .collect();

    // --- Primary UBO binding (binding=#0) ---
    let primary_ubo = if !primary_uniform_fields.is_empty() {
        quote! {
            #CRATE::vk::DescriptorSetLayoutBinding {
                binding              : 0,
                stage_flags          : #CRATE::vk::ShaderStageFlags::ALL,
                descriptor_type      : #CRATE::vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count     : 1,
                p_immutable_samplers : ::std::ptr::null()
            },
        }
    } else {
        quote! {}
    };


    /* quote! {
        // upload inline uniforms
        #[repr(C)]
        struct PrimaryUniforms < #(#primary_uniform_type_params,)* > {
            #(#primary_uniform_fields,)*
        }
    };*/

    // --- Generate update template entries ---
    let descriptor_update_template_entries: Vec<_> = bindings
        .iter()
        .map(|b| {
            let binding = b.binding;
            let ty = &b.field.ty;
            quote! {
                #CRATE::vk::DescriptorUpdateTemplateEntry {
                    dst_binding       : #binding,
                    dst_array_element : 0,
                    descriptor_count  : <#ty as #CRATE::DescriptorBinding>::DESCRIPTOR_COUNT,
                    descriptor_type   : <#ty as #CRATE::DescriptorBinding>::DESCRIPTOR_TYPE,
                    offset            : <#ty as #CRATE::DescriptorBinding>::UPDATE_OFFSET,
                    stride            : <#ty as #CRATE::DescriptorBinding>::UPDATE_STRIDE
                }
            }
        })
        .collect();

    // --- Descriptor update statements ---
    let descriptor_prepare_update_statements: Vec<_> = bindings
        .iter()
        .map(|b| {
            if let Some(ref ident) = b.field.ident {
                quote! { #CRATE::DescriptorBinding::prepare_descriptors(&mut self.#ident, ctx); }
            } else {
                let index = syn::Index::from(b.field_index);
                quote! { #CRATE::DescriptorBinding::prepare_descriptors(&mut self.#index, ctx); }
            }
        })
        .collect();

    // --- resource visitor statements ---
    let descriptor_visit_stmts: Vec<_> = bindings
        .iter()
        .map(|b| {
            if let Some(ref ident) = b.field.ident {
                quote! { #CRATE::DescriptorBinding::visit(&self.#ident, visitor); }
            } else {
                let index = syn::Index::from(b.field_index);
                quote! { #CRATE::DescriptorBinding::visit(&self.#index, visitor); }
            }
        })
        .collect();

    // --- primary uniform upload block
    let primary_uniform_upload_stmts = if !primary_uniform_fields.is_empty() {

        // --- Primary uniform struct ---
        let primary_uniforms_struct = match fields {
            Fields::Named(named) => {
                let fields_2 = primary_uniform_fields.iter().map(|f| &f.field);
                quote! {
                    #[derive(Copy,Clone)]
                    #[repr(C)]
                    struct PrimaryUniforms < #(#primary_uniform_type_params,)* > {
                        #(#fields_2)*
                    }
                }
            }
            Fields::Unnamed(unnamed) => {
                let fields_2 = primary_uniform_fields.iter().map(|f| &f.field.ty);
                quote! {
                    #[derive(Copy,Clone)]
                    #[repr(C)]
                    struct PrimaryUniforms < #(#primary_uniform_type_params,)* >(#(#fields_2)*)
                }
            }
            Fields::Unit => { quote!{} }
        };

        let primary_uniforms_field_init: Vec<_> = primary_uniform_fields
            .iter()
            .enumerate()
            .map(|(i, f)| {
                if let Some(ref ident) = f.field.ident {
                    quote! { #ident : self.#ident }
                } else {
                    let i = syn::Index::from(i);
                    let field_index = syn::Index::from(f.field_index);
                    quote! { #i: self.#field_index }
                }
            })
            .collect();

        quote!{
            // upload inline uniforms
            #primary_uniforms_struct
            // TODO we could allocate the struct directly in the upload buffer and fill it there
            let primary_uniforms = PrimaryUniforms { #(#primary_uniforms_field_init,)* };
            let (buffer, offset) = ctx.upload_slice(::std::slice::from_ref(&primary_uniforms), #CRATE::vk::BufferUsageFlags::UNIFORM_BUFFER);
        }
    } else {
        quote!{}
    };

    let struct_name = &derive_input.ident;
    let (impl_generics, ty_generics, where_clause) = s.generics.split_for_impl();
    let unique_type_name =
        syn::Ident::new(&format!("__{}_UniqueType", struct_name), Span::call_site());

    // --- generics without lifetimes, to get a unique typeid (because of https://github.com/rust-lang/rust/issues/41875) ---
    // TODO recursively replace inner lifetimes with 'static => PITA
    let impl_generics_without_lifetimes = ImplGenericsWithoutLifetimesOrBounds(&s.generics);
    let type_generics_without_lifetimes = TypeGenericsWithoutLifetimes(&s.generics);
    let type_params: Vec<_> = s.generics.type_params().map(|tp| &tp.ident).collect();

    quote! {
        // potential compile_errors
        #repr_c_check
        #(#attrib_errors)*

        // private type for getting a unique typeid.
        // we don't need the bounds?
        struct #unique_type_name  #impl_generics_without_lifetimes (::std::marker::PhantomData<(#(#type_params,)*)>);

        impl #impl_generics #CRATE::ResourceHolder for #struct_name #ty_generics #where_clause {
            fn walk_resources(&self, visitor: &mut dyn #CRATE::ResourceVisitor) {
                #(#descriptor_visit_stmts)*
            }
        }

        impl #impl_generics #CRATE::Arguments for #struct_name #ty_generics #where_clause {

            fn unique_type_id(&self) -> Option<::std::any::TypeId> {
                Some( ::std::any::TypeId::of::<#unique_type_name #type_generics_without_lifetimes>())
            }

            fn get_descriptor_set_layout_bindings(&self) -> &[#CRATE::vk::DescriptorSetLayoutBinding]
            {
                &[#primary_ubo #(#descriptor_set_layout_bindings,)*]
            }

            fn get_descriptor_set_update_template_entries(
                &self,
            ) -> Option<&[#CRATE::vk::DescriptorUpdateTemplateEntry]>
            {
                Some(&[#(#descriptor_update_template_entries,)*])
            }

            unsafe fn update_descriptor_set(
                &mut self,
                ctx: &mut #CRATE::ContextResources,
                set: #CRATE::vk::DescriptorSet,
                update_template: Option<#CRATE::vk::DescriptorUpdateTemplate>)
            {
                #(#descriptor_prepare_update_statements)*
                let device = ctx.vulkan_device();
                // update with template
                if let Some(update_template) = update_template {
                    device.update_descriptor_set_with_template(set, update_template, self as *const _ as *const ::std::ffi::c_void);
                }
                #primary_uniform_upload_stmts
            }
        }
    }
}
