use crate::pass::BatchSerialNumber;
use ash::version::DeviceV1_0;
use ash::vk;
use std::collections::HashMap;
use crate::vk::DescriptorPoolSize;

const MAX_DESCRIPTOR_SET_LAYOUT_BINDING_DESCRIPTORS: usize = 16;
const MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS: usize = 16;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DescriptorSetLayoutBindingInfo {
    pub binding: u32,
    pub descriptor_type: vk::DescriptorType,
    pub descriptor_count: u32,
    pub stage_flags: vk::ShaderStageFlags,
    pub immutable_samplers: [vk::Sampler; MAX_DESCRIPTOR_SET_LAYOUT_BINDING_DESCRIPTORS],
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct DescriptorSetLayoutInfo {
    pub binding_count: u32,
    pub bindings: [DescriptorSetLayoutBindingInfo; MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS],
}

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
struct DescriptorSetAllocator {
    layout_info: DescriptorSetLayoutInfo,
    layout_handle: vk::DescriptorSetLayout,
    pool_size_count: u32,
    pool_sizes: [vk::DescriptorPoolSize; 16],
    full_pools: Vec<vk::DescriptorPool>,
    pool: vk::DescriptorPool,
    free: Vec<vk::DescriptorSet>,
    used: Vec<TrackedDescriptorSet>,
}

impl DescriptorSetAllocator {
    pub fn new(
        device: &ash::Device,
        layout_info: &DescriptorSetLayoutInfo,
    ) -> DescriptorSetAllocator {
        let mut descriptor_set_layout_bindings: [vk::DescriptorSetLayoutBinding;
            MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS] = Default::default();
        for i in 0..layout_info.binding_count as usize {
            descriptor_set_layout_bindings[i].binding = layout_info.bindings[i].binding;
            descriptor_set_layout_bindings[i].descriptor_type =
                layout_info.bindings[i].descriptor_type;
            descriptor_set_layout_bindings[i].descriptor_count =
                layout_info.bindings[i].descriptor_count;
            descriptor_set_layout_bindings[i].stage_flags = layout_info.bindings[i].stage_flags;
            descriptor_set_layout_bindings[i].p_immutable_samplers =
                layout_info.bindings[i].immutable_samplers.as_ptr();
        }

        let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: layout_info.binding_count,
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

        for i in 0..layout_info.binding_count {
            let b = &layout_info.bindings[i as usize];
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
            layout_info: *layout_info,
            layout_handle,
            pool_sizes,
            pool_size_count: pool_size_count as u32,
            full_pools: vec![],
            pool: vk::DescriptorPool::null(),
            free: vec![],
            used: vec![],
        }
    }

    /// Gets a descriptor set.
    pub fn allocate_set(
        &mut self,
        device: &ash::Device,
        batch: BatchSerialNumber,
    ) -> vk::DescriptorSet {
        let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: self.pool,
            descriptor_set_count: 1,
            p_set_layouts: &self.layout_handle,
            ..Default::default()
        };

        let handle = loop {
            let result = unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) };

            match result {
                Ok(d) => break *d.first().unwrap(),
                Err(vk::Result::ERROR_OUT_OF_POOL_MEMORY) => {
                    // allocate a new pool and continue
                    let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo {
                        flags: vk::DescriptorPoolCreateFlags::default(),
                        max_sets: DESCRIPTOR_POOL_SET_COUNT,
                        pool_size_count: self.pool_size_count,
                        p_pool_sizes: self.pool_sizes.as_ptr(),
                        ..Default::default()
                    };
                    self.full_pools.push(self.pool);
                    self.pool = unsafe {
                        device
                            .create_descriptor_pool(&descriptor_pool_create_info, None)
                            .expect("failed to create descriptor pool")
                    };
                    continue;
                }
                Err(e) => panic!("error allocating descriptor sets: {}", e),
            }
        };

        self.used.push(TrackedDescriptorSet {
            handle,
            batch
        });

        handle
    }
}

struct DescriptorCache {
    entries: HashMap<DescriptorSetLayoutInfo, DescriptorSetAllocator>,
}
