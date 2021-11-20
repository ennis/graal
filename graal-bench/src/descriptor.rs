use ash::vk;
use graal_spirv as spirv;
use std::{collections::BTreeMap, marker::PhantomData};
use crate::vk;

//const MAX_DESCRIPTOR_SET_LAYOUT_BINDING_DESCRIPTORS: usize = 16;
pub const MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS: usize = 16;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct DescriptorSetLayoutBindingInfo {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    //pub immutable_samplers: Option<[vk::Sampler; 16]>,  // only there so that
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct DescriptorSetLayoutInfo {
    pub binding_count: u32,
    pub bindings: [DescriptorSetLayoutBindingInfo; MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS],
}

impl<'a> From<&'a [DescriptorSetLayoutBindingInfo]> for DescriptorSetLayoutInfo {
    fn from(layout: &'a [DescriptorSetLayoutBindingInfo]) -> Self {
        let mut info = DescriptorSetLayoutInfo::default();
        info.binding_count = layout.len() as u32;
        info.bindings[0..layout.len()].copy_from_slice(layout);
        info
    }
}

/*pub struct OutputAttachment {
    // TODO
}*/

/*pub struct PushConstantLayout {
    pub offset: usize,
    pub size: usize,
}*/

/// Trait implemented by types that describe a _descriptor set interface_: they are types that contain
/// the descriptors needed to build a descriptor set.
///
/// In most cases, you should use the derive macro (`#[derive(DescriptorSetInterface)]`)
/// provided by the library instead of implementing this trait manually.
pub trait DescriptorSetInterface {
    /// The _descriptor set layout_ of the descriptor sets that this type represents.
    const LAYOUT: &'static [DescriptorSetLayoutBindingInfo];

    /// Entries of the _descriptor update template_ for this type. This is used by Vulkan to directly
    /// read descriptors from instances of this type.
    const UPDATE_TEMPLATE_ENTRIES: &'static [vk::DescriptorUpdateTemplateEntry];

    /// Used internally by `Context` to associate a descriptor set layout ID to this type.
    fn get_or_init_layout(
        init: impl FnOnce() -> DescriptorSetAllocatorId,
    ) -> DescriptorSetAllocatorId;

    /// Updates the given descriptor set from the descriptors contained in this object.
    /// `update_template` must have been created according to
    /// `Self::UPDATE_TEMPLATE_ENTRIES`, and the layout of `set` must be the same as the one described
    /// by `Self::LAYOUT`.
    unsafe fn update_descriptors(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        update_template: vk::DescriptorUpdateTemplate,
    );
}

/*pub trait PushConstantInterface {
    const LAYOUT: &'static [PushConstantLayout];
}*/

///
pub enum DescriptorWriteKind {
    Image,
    Buffer,
    TexelBufferView,
}

/// A trait implemented by types that can produce one or more descriptors.
pub unsafe trait DescriptorSource {
    const KIND: DescriptorWriteKind;
    const ARRAY_SIZE: u32 = 1;

    /// Returns the number of descriptors in this binding.
    fn descriptor_count(&self) -> u32 {
        1
    }
}

/// Impl for descriptor arrays
unsafe impl<'a, D: DescriptorSource> DescriptorSource for &'a [D] {
    const KIND: DescriptorWriteKind = D::KIND;

    fn descriptor_count(&self) -> u32 {
        self.len() as u32
    }
}

unsafe impl DescriptorSource for vk::Sampler {
    const KIND: DescriptorWriteKind = DescriptorWriteKind::Image;
}

unsafe impl DescriptorSource for vk::DescriptorBufferInfo {
    const KIND: DescriptorWriteKind = DescriptorWriteKind::Buffer;
}

unsafe impl DescriptorSource for vk::DescriptorImageInfo {
    const KIND: DescriptorWriteKind = DescriptorWriteKind::Image;
}

unsafe impl<T: DescriptorSource, const N: usize> DescriptorSource for [T; N] {
    const KIND: DescriptorWriteKind = DescriptorWriteKind::Image;
    const ARRAY_SIZE: u32 = N as u32;

    fn descriptor_count(&self) -> u32 {
        N as u32
    }
}

/// Strongly-typed buffer descriptor.
#[repr(transparent)]
#[derive(Copy, Clone, Debug)]
pub struct BufferDescriptor<T> {
    pub descriptor: vk::DescriptorBufferInfo,
    pub _phantom: PhantomData<*const T>,
}

unsafe impl<T> DescriptorSource for BufferDescriptor<T> {
    const KIND: DescriptorWriteKind = DescriptorWriteKind::Buffer;
}

impl<T> From<TypedBufferInfo<T>> for BufferDescriptor<T> {
    fn from(b: TypedBufferInfo<T>) -> Self {
        BufferDescriptor {
            descriptor: vk::DescriptorBufferInfo {
                buffer: b.handle,
                offset: 0,
                range: vk::WHOLE_SIZE,
            },
            _phantom: PhantomData,
        }
    }
}

pub struct PipelineShaderStage<'a> {
    pub stage: vk::ShaderStageFlags,
    pub spirv: &'a [u32],
}

/*struct VertexInputVariable {
    location: u32,
}*/

/*struct FragmentOutputVariable {
    location: u32,
}*/

/*struct ShaderInterfaces {
    vertex_input: BTreeMap<u32, VertexInputVariable>,
    fragment_output: BTreeMap<u32, FragmentOutputVariable>,
    resource: BTreeMap<u32, DescriptorSetLayoutInfo>,
    // push_constant:
}*/

pub fn extract_descriptor_set_layouts_from_shader_stages(
    stages: &[PipelineShaderStage],
) -> BTreeMap<u32, DescriptorSetLayoutInfo> {
    let mut bindings: BTreeMap<(u32, u32), DescriptorSetLayoutBindingInfo> = BTreeMap::new();
    let arena = spirv::Arena::new();

    for pipeline_stage in stages.iter() {
        let module = spirv::Module::from_words(&arena, pipeline_stage.spirv).unwrap();

        for v in module.variables {
            if let Some(set) = v.descriptor_set {
                if let Some(binding) = v.binding {
                    use graal_spirv::typedesc::{ImageType, StructType, TypeDesc::*};
                    use spirv::spv::StorageClass;

                    let (descriptor_type, _unbounded_descriptor_array) =
                        match (v.storage_class, v.ty) {
                            // According to https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#interfaces-resources-descset
                            // for SampledImages, the descriptor type could be either VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE or VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER

                            // According to the spec, it's possible to have both a `texture` and a `sampler` in the same binding point:
                            //
                            //      A noteworthy example of using multiple statically-used shader variables sharing the same descriptor set and binding values
                            //      is a descriptor of type VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER that has multiple corresponding shader variables in the UniformConstant storage class,
                            //      where some could be OpTypeImage, some could be OpTypeSampler (Sampled=1), and some could be OpTypeSampledImage.
                            //
                            // We don't support it. Please don't do that.

                            // --- unbounded texture descriptor arrays (descriptor indexing) ---
                            (
                                StorageClass::UniformConstant,
                                &Pointer(&Array {
                                    elem_ty: Image(_), ..
                                }),
                            ) => (vk::DescriptorType::SAMPLED_IMAGE, true),

                            // --- standalone samplers ----
                            (StorageClass::UniformConstant, &Pointer(&Sampler)) => {
                                (vk::DescriptorType::SAMPLER, false)
                            }

                            // --- sampler1D,sampler2D ----
                            (StorageClass::UniformConstant, &Pointer(&SampledImage(_img))) => {
                                (vk::DescriptorType::COMBINED_IMAGE_SAMPLER, false)
                            }

                            // --- textures, images, texel buffers ---
                            (
                                StorageClass::UniformConstant,
                                &Pointer(&Image(ImageType { dim, sampled, .. })),
                            ) => {
                                match dim {
                                    spirv::spv::Dim::DimBuffer => match sampled {
                                        Some(false) => {
                                            (vk::DescriptorType::STORAGE_TEXEL_BUFFER, false)
                                        }
                                        Some(true) => {
                                            (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, false)
                                        }
                                        None => (vk::DescriptorType::UNIFORM_TEXEL_BUFFER, false),
                                    },
                                    _ => match sampled {
                                        // texture1D,texture2D...
                                        Some(true) => (vk::DescriptorType::SAMPLED_IMAGE, false),
                                        // image1D,image2D...
                                        Some(false) => (vk::DescriptorType::STORAGE_IMAGE, false),
                                        // we pick?
                                        None => (vk::DescriptorType::SAMPLED_IMAGE, false),
                                    },
                                }
                            }

                            // --- uniform buffer ---
                            (StorageClass::Uniform, &Pointer(&Struct(sty))) if sty.block => {
                                // handle graal set conventions:
                                // depending on the set, the uniform buffer may or may not be dynamic
                                // sets 0-2 are static, sets > 2 are dynamic by convention
                                if set < 3 {
                                    (vk::DescriptorType::UNIFORM_BUFFER, false)
                                } else {
                                    (vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC, false)
                                }
                            }

                            // --- storage buffer ---
                            (
                                StorageClass::Uniform,
                                &Pointer(&Struct(StructType {
                                    buffer_block: true, ..
                                })),
                            )
                            | (
                                StorageClass::StorageBuffer,
                                &Pointer(&Struct(StructType { block: true, .. })),
                            ) => {
                                if set < 3 {
                                    (vk::DescriptorType::STORAGE_BUFFER, false)
                                } else {
                                    (vk::DescriptorType::STORAGE_BUFFER_DYNAMIC, false)
                                }
                            }
                            _ => continue,
                        };

                    let mut binding_info = bindings.entry((set, binding)).or_default();
                    binding_info.binding = binding;
                    binding_info.descriptor_type = descriptor_type;
                    binding_info.descriptor_count = 1;
                    binding_info.stage_flags |= pipeline_stage.stage;
                    // TODO in-shader sampler declaration?
                }
            }
        }
    }

    let mut set_layouts: BTreeMap<u32, DescriptorSetLayoutInfo> = BTreeMap::new();
    // group by set
    for (&(set, binding), binding_info) in bindings.iter() {
        let set_layout_info = set_layouts.entry(set).or_default();
        set_layout_info.bindings[binding as usize] = *binding_info;
    }

    set_layouts

    // type-safe partial descriptor interfaces
    // -> derive(DescriptorSetInterface)
    // -> the struct contains a bunch of resources to turn into a descriptor set
    //      -> TypedUniformBufferView
    //      -> for dynamic uniforms: UniformBufferView (can't be typed, unfortunately)
    // -> typed descriptor set
    // -> the type contains the layout description (bunch of static DescriptorSetLayoutBindingInfos)

    // "attach" interfaces to pipelines:
    // - pipeline.verify_shader_interface<D: DescriptorSetInterface>(set) -> Result<(), ValidationError> {}

    // how can we refer to a descriptor set allocator?
    // - hash the layout info every time? too costly
    // - borrow the allocator? borrow-lock
    // - by ID, like the rest

    // allocate a descriptor set:
    // - get layout allocator, either by looking up the DescriptorSetInterface or from the shader
    // - allocate the set
    // - write stuff in the set (manually or with auto-generated stuff in the DescriptorSetInterface impl)

    // dynamic uniforms?

    //Vec::new()
}
