use crate::pass::BatchSerialNumber;
use crate::vk::{
    BufferView, DescriptorBufferInfo, DescriptorImageInfo, DescriptorPoolSize, DescriptorType,
    ShaderStageFlags,
};
use crate::Device;
use ash::version::DeviceV1_0;
use ash::vk;
use graal_spirv as spirv;
use slotmap::SlotMap;
use std::collections::{BTreeMap, HashMap};
use std::ffi::CString;
use std::{ptr, mem};

const MAX_DESCRIPTOR_SET_LAYOUT_BINDING_DESCRIPTORS: usize = 16;
const MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS: usize = 16;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Default)]
pub struct DescriptorSetLayoutBindingInfo {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub immutable_samplers: [vk::Sampler; 16],
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

pub struct OutputAttachment {
    // TODO
}

pub struct PushConstantLayout {
    pub offset: usize,
    pub size: usize,
}

pub trait DescriptorSetInterface {
    const LAYOUT: &'static [DescriptorSetLayoutBindingInfo];
    // only for entries with statically known sizes
    const UPDATE_TEMPLATE_ENTRIES: &'static [vk::DescriptorUpdateTemplateEntry];

    unsafe fn update_descriptors(
        &self,
        device: &ash::Device,
        set: vk::DescriptorSet,
        update_template: vk::DescriptorUpdateTemplate,
    );
}

pub trait PushConstantInterface {
    const LAYOUT: &'static [PushConstantLayout];
}

pub trait FragmentOutputInterface {
    const ATTACHMENTS: &'static [OutputAttachment];
}

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

//-----------------------------------------------------------------------------------------
#[derive(Copy, Clone, Debug)]
struct TrackedDescriptorSet {
    /// vulkan handle
    handle: vk::DescriptorSet,
    /// The batch index that last used this descriptor
    batch: BatchSerialNumber,
}

const DESCRIPTOR_POOL_PER_TYPE_COUNT: u32 = 1024;
const DESCRIPTOR_POOL_SET_COUNT: u32 = DESCRIPTOR_POOL_PER_TYPE_COUNT;

#[derive(Debug)]
pub struct DescriptorSetAllocator {
    layout_handle: vk::DescriptorSetLayout,
    pool_size_count: u32,
    pool_sizes: [vk::DescriptorPoolSize; 16],
    full_pools: Vec<vk::DescriptorPool>,
    pool: Option<vk::DescriptorPool>,
    free: Vec<vk::DescriptorSet>,
    used: Vec<TrackedDescriptorSet>,
}

impl DescriptorSetAllocator {
    pub fn new(
        device: &ash::Device,
        layout: &[DescriptorSetLayoutBindingInfo],
    ) -> DescriptorSetAllocator {
        let mut descriptor_set_layout_bindings: [vk::DescriptorSetLayoutBinding;
            MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS] = Default::default();
        for i in 0..layout.len() {
            descriptor_set_layout_bindings[i].binding = layout[i].binding;
            descriptor_set_layout_bindings[i].descriptor_type = layout[i].descriptor_type;
            descriptor_set_layout_bindings[i].descriptor_count = layout[i].descriptor_count;
            descriptor_set_layout_bindings[i].stage_flags = layout[i].stage_flags;
            descriptor_set_layout_bindings[i].p_immutable_samplers =
                layout[i].immutable_samplers.as_ptr();
        }

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: layout.len() as u32,
            p_bindings: descriptor_set_layout_bindings.as_ptr(),
            ..Default::default()
        };
        let layout_handle = unsafe {
            device
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .expect("failed to create descriptor set layout")
        };

        let mut pool_sizes: [DescriptorPoolSize; 16] = Default::default();
        // count the number of each type of descriptor
        let mut sampler_desc_count = 0;
        let mut combined_image_sampler_desc_count = 0;
        let mut sampled_image_desc_count = 0;
        let mut storage_image_desc_count = 0;
        let mut uniform_texel_buffer_desc_count = 0;
        let mut storage_texel_buffer_desc_count = 0;
        let mut uniform_buffer_desc_count = 0;
        let mut storage_buffer_desc_count = 0;
        let mut uniform_buffer_dynamic_desc_count = 0;
        let mut storage_buffer_dynamic_desc_count = 0;
        let mut input_attachment_desc_count = 0;
        let mut acceleration_structure_desc_count = 0;

        for b in layout.iter() {
            match b.descriptor_type {
                vk::DescriptorType::SAMPLER => sampler_desc_count += 1,
                vk::DescriptorType::COMBINED_IMAGE_SAMPLER => {
                    combined_image_sampler_desc_count += 1
                }
                vk::DescriptorType::SAMPLED_IMAGE => sampled_image_desc_count += 1,
                vk::DescriptorType::STORAGE_IMAGE => storage_image_desc_count += 1,
                vk::DescriptorType::UNIFORM_TEXEL_BUFFER => uniform_texel_buffer_desc_count += 1,
                vk::DescriptorType::STORAGE_TEXEL_BUFFER => storage_texel_buffer_desc_count += 1,
                vk::DescriptorType::UNIFORM_BUFFER => uniform_buffer_desc_count += 1,
                vk::DescriptorType::STORAGE_BUFFER => storage_buffer_desc_count += 1,
                vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC => {
                    uniform_buffer_dynamic_desc_count += 1
                }
                vk::DescriptorType::STORAGE_BUFFER_DYNAMIC => {
                    storage_buffer_dynamic_desc_count += 1
                }
                vk::DescriptorType::INPUT_ATTACHMENT => input_attachment_desc_count += 1,
                vk::DescriptorType::ACCELERATION_STRUCTURE_KHR => {
                    acceleration_structure_desc_count += 1
                }
                _ => {}
            }
        }

        let mut pool_size_count = 0;
        if sampler_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::SAMPLER;
            pool_sizes[pool_size_count].descriptor_count =
                sampler_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if combined_image_sampler_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::COMBINED_IMAGE_SAMPLER;
            pool_sizes[pool_size_count].descriptor_count =
                combined_image_sampler_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if sampled_image_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::SAMPLED_IMAGE;
            pool_sizes[pool_size_count].descriptor_count =
                sampled_image_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_image_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_IMAGE;
            pool_sizes[pool_size_count].descriptor_count =
                storage_image_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_texel_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_TEXEL_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_texel_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_texel_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_TEXEL_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                storage_texel_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_buffer_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_BUFFER;
            pool_sizes[pool_size_count].descriptor_count =
                storage_buffer_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if uniform_buffer_dynamic_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::UNIFORM_BUFFER_DYNAMIC;
            pool_sizes[pool_size_count].descriptor_count =
                uniform_buffer_dynamic_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if storage_buffer_dynamic_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::STORAGE_BUFFER_DYNAMIC;
            pool_sizes[pool_size_count].descriptor_count =
                storage_buffer_dynamic_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if input_attachment_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::INPUT_ATTACHMENT;
            pool_sizes[pool_size_count].descriptor_count =
                input_attachment_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }
        if acceleration_structure_desc_count != 0 {
            pool_sizes[pool_size_count].ty = vk::DescriptorType::ACCELERATION_STRUCTURE_KHR;
            pool_sizes[pool_size_count].descriptor_count =
                acceleration_structure_desc_count * DESCRIPTOR_POOL_PER_TYPE_COUNT;
            pool_size_count += 1;
        }

        DescriptorSetAllocator {
            //layout_info: *layout_info,
            layout_handle,
            pool_sizes,
            pool_size_count: pool_size_count as u32,
            full_pools: vec![],
            pool: None,
            free: vec![],
            used: vec![],
        }
    }

    fn retire_descriptor_pool(&mut self) {
        if let Some(pool) = mem::replace(&mut self.pool, None) {
            self.full_pools.push(pool);
        }
    }

    fn get_descriptor_pool(&mut self, device: &ash::Device) -> vk::DescriptorPool {
        if let Some(pool) = self.pool {
            return pool;
        }

        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
            flags: vk::DescriptorPoolCreateFlags::default(),
            max_sets: DESCRIPTOR_POOL_SET_COUNT,
            pool_size_count: self.pool_size_count,
            p_pool_sizes: self.pool_sizes.as_ptr(),
            ..Default::default()
        };

        let pool = unsafe {
            device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        };

        self.pool = Some(pool);
        pool
    }

    /// Gets a descriptor set.
    pub fn allocate_set(
        &mut self,
        device: &ash::Device,
        batch: BatchSerialNumber,
    ) -> vk::DescriptorSet {
        let handle = loop {
            let descriptor_pool = self.get_descriptor_pool(device);
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.layout_handle,
                ..Default::default()
            };

            let result = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) };

            match result {
                Ok(d) => break *d.first().unwrap(),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                    self.retire_descriptor_pool();
                    continue;
                }
                Err(e) => panic!("error allocating descriptor sets: {}", e),
            }
        };

        self.used.push(TrackedDescriptorSet { handle, batch });

        handle
    }

    // TODO "recycle" instead? it doesn't actually free memory
    pub fn cleanup(&mut self, completed_batch: BatchSerialNumber) {
        let mut i = 0;
        while i < self.used.len() {
            if self.used[i].batch <= completed_batch {
                self.free.push(self.used.swap_remove(i).handle);
            } else {
                i += 1;
            }
        }
    }
}

pub struct DescriptorSetLayoutCache {
    entries: SlotMap<DescriptorSetLayoutId, DescriptorSetAllocator>,
    map: HashMap<DescriptorSetLayoutInfo, DescriptorSetLayoutId>,
}

impl DescriptorSetLayoutCache {
    pub fn new() -> DescriptorSetLayoutCache {
        DescriptorSetLayoutCache {
            entries: Default::default(),
            map: Default::default(),
        }
    }

    pub fn create_descriptor_set_layout(
        &mut self,
        device: &ash::Device,
        layout: &[DescriptorSetLayoutBindingInfo],
    ) -> (vk::DescriptorSetLayout, DescriptorSetLayoutId) {
        let mut entries = &mut self.entries;
        let id = *self
            .map
            .entry(DescriptorSetLayoutInfo::from(layout))
            .or_insert_with(|| entries.insert(DescriptorSetAllocator::new(device, layout)));

        (self.entries.get(id).unwrap().layout_handle, id)
    }

    pub fn layout_allocator(&mut self, id: DescriptorSetLayoutId) -> &mut DescriptorSetAllocator {
        self.entries.get_mut(id).unwrap()
    }

    pub fn create_descriptor_set_layout_from_interface<T: DescriptorSetInterface>(
        &mut self,
        device: &ash::Device,
    ) -> (vk::DescriptorSetLayout, DescriptorSetLayoutId) {
        self.create_descriptor_set_layout(device, T::LAYOUT)
    }
}

slotmap::new_key_type! {
    pub struct PipelineLayoutId;
    pub struct DescriptorSetLayoutId;
}

const MAX_DESCRIPTOR_SET_LAYOUTS: usize = 8;

pub struct PipelineShaderStage<'a> {
    pub stage: vk::ShaderStageFlags,
    pub spirv: &'a [u32],
}

struct VertexInputVariable {
    location: u32,
}

struct FragmentOutputVariable {
    location: u32,
}

struct ShaderInterfaces {
    vertex_input: BTreeMap<u32, VertexInputVariable>,
    fragment_output: BTreeMap<u32, FragmentOutputVariable>,
    resource: BTreeMap<u32, DescriptorSetLayoutInfo>,
    // push_constant:
}

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
                    use crate::typedesc::ImageType;
                    use crate::typedesc::StructType;
                    use crate::typedesc::TypeDesc::*;
                    use spirv::spv::StorageClass;

                    let (descriptor_type, unbounded_descriptor_array) =
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
                            (StorageClass::UniformConstant, &Pointer(&SampledImage(img))) => {
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
