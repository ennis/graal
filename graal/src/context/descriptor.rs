use crate::{
    descriptor::MAX_DESCRIPTOR_SET_LAYOUT_BINDINGS, vk, Context, DescriptorSetInterface,
    DescriptorSetLayoutBindingInfo, DescriptorSetLayoutInfo,
};
use ash::version::{DeviceV1_0, DeviceV1_1};
use slotmap::{SecondaryMap, SlotMap};
use std::{collections::HashMap, mem};

const DESCRIPTOR_POOL_PER_TYPE_COUNT: u32 = 1024;
const DESCRIPTOR_POOL_SET_COUNT: u32 = DESCRIPTOR_POOL_PER_TYPE_COUNT;
const MAX_DESCRIPTOR_SET_LAYOUTS: usize = 8;

slotmap::new_key_type! {
    pub struct DescriptorSetAllocatorId;
}

#[derive(Debug)]
pub(crate) struct DescriptorSetAllocator {
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) update_template: Option<vk::DescriptorUpdateTemplate>,
    pub(crate) pool_size_count: u32,
    pub(crate) pool_sizes: [vk::DescriptorPoolSize; 16],
    pub(crate) full_pools: Vec<vk::DescriptorPool>,
    pub(crate) pool: Option<vk::DescriptorPool>,
    pub(crate) free: Vec<vk::DescriptorSet>,
}

impl DescriptorSetAllocator {
    pub fn new(
        device: &ash::Device,
        layout: &[DescriptorSetLayoutBindingInfo],
        update_template: Option<&[vk::DescriptorUpdateTemplateEntry]>,
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

        let mut pool_sizes: [vk::DescriptorPoolSize; 16] = Default::default();
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

        // create the update template if one was provided
        let update_template = if let Some(update_template) = update_template {
            let update_template_create_info = vk::DescriptorUpdateTemplateCreateInfo {
                flags: Default::default(),
                descriptor_update_entry_count: update_template.len() as u32,
                p_descriptor_update_entries: update_template.as_ptr(),
                template_type: vk::DescriptorUpdateTemplateType::DESCRIPTOR_SET,
                descriptor_set_layout: layout_handle,
                pipeline_bind_point: Default::default(),
                pipeline_layout: vk::PipelineLayout::null(),
                set: 0,
                ..Default::default()
            };

            let template = unsafe {
                device
                    .create_descriptor_update_template(&update_template_create_info, None)
                    .unwrap()
            };

            Some(template)
        } else {
            None
        };

        DescriptorSetAllocator {
            //layout_info: *layout_info,
            layout: layout_handle,
            pool_sizes,
            pool_size_count: pool_size_count as u32,
            full_pools: vec![],
            pool: None,
            free: vec![],
            update_template,
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

    /// Allocates a descriptor set.
    pub fn alloc(&mut self, device: &ash::Device) -> vk::DescriptorSet {
        let handle = loop {
            let descriptor_pool = self.get_descriptor_pool(device);
            let descriptor_set_allocate_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &self.layout,
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

        handle
    }

    pub(crate) unsafe fn recycle(&mut self, set: vk::DescriptorSet) {
        self.free.push(set)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DescriptorSet {
    pub(crate) alloc_id: DescriptorSetAllocatorId,
    pub(crate) set: vk::DescriptorSet,
}

impl Context {
    /// Creates a descriptor set layout.
    pub fn get_or_create_descriptor_set_allocator(
        &mut self,
        layout: &[DescriptorSetLayoutBindingInfo],
    ) -> DescriptorSetAllocatorId {
        let device = &self.device.device;
        let mut set_allocators = &mut self.set_allocators;
        *self
            .cache
            .entry(DescriptorSetLayoutInfo::from(layout))
            .or_insert_with(|| {
                set_allocators.insert(DescriptorSetAllocator::new(device, layout, None))
            })
    }

    /// Gets or creates the descriptor set layout for the specified interface type.
    pub fn get_or_create_descriptor_set_allocator_from_interface<T: DescriptorSetInterface>(
        &mut self,
    ) -> DescriptorSetAllocatorId {
        T::get_or_init_layout(|| {
            self.set_allocators.insert(DescriptorSetAllocator::new(
                &self.device.device,
                T::LAYOUT,
                Some(T::UPDATE_TEMPLATE_ENTRIES),
            ))
        })
    }

    pub fn get_or_create_descriptor_set_layout_for_interface<T: DescriptorSetInterface>(
        &mut self,
    ) -> vk::DescriptorSetLayout {
        let alloc_id = self.get_or_create_descriptor_set_allocator_from_interface::<T>();
        self.set_allocators.get(alloc_id).unwrap().layout
    }

    ///
    pub fn create_descriptor_set<T: DescriptorSetInterface>(
        &mut self,
        descriptors: &T,
    ) -> DescriptorSet {
        let alloc_id = self.get_or_create_descriptor_set_allocator_from_interface::<T>();
        let set_allocator = self.set_allocators.get_mut(alloc_id).unwrap();
        let set = set_allocator.alloc(&self.device.device);
        unsafe {
            // TODO safety
            descriptors.update_descriptors(
                &self.device.device,
                set,
                set_allocator.update_template.unwrap(),
            );
        }
        DescriptorSet { alloc_id, set }
    }

    pub(crate) unsafe fn recycle_descriptor_sets(&mut self, sets: Vec<DescriptorSet>) {
        // put the sets back in their respective allocators
        for s in sets {
            let set_allocator = self.set_allocators.get_mut(s.alloc_id).unwrap();
            set_allocator.recycle(s.set);
        }
    }
}
