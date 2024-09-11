use crate::{Device, ImageViewId, SamplerId};
use ash::{vk, vk::DescriptorType};
use std::{ffi::c_void, ptr};
use tracing::trace;

type DT = vk::DescriptorType;

unsafe fn create_bindless_layout(
    device: &ash::Device,
    descriptor_type: vk::DescriptorType,
    count: usize,
) -> vk::DescriptorSetLayout {
    let bindings = [vk::DescriptorSetLayoutBinding {
        binding: 0,
        descriptor_type,
        descriptor_count: count as u32,
        stage_flags: vk::ShaderStageFlags::ALL,
        p_immutable_samplers: ptr::null(),
    }];

    let flags = [vk::DescriptorBindingFlags::PARTIALLY_BOUND | vk::DescriptorBindingFlags::UPDATE_UNUSED_WHILE_PENDING];
    let dslbfci = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
        binding_count: flags.len() as u32,
        p_binding_flags: flags.as_ptr(),
        ..Default::default()
    };

    let dslci = vk::DescriptorSetLayoutCreateInfo {
        p_next: &dslbfci as *const _ as *const c_void,
        flags: Default::default(),
        binding_count: bindings.len() as u32,
        p_bindings: bindings.as_ptr(),
        ..Default::default()
    };

    let handle = device
        .create_descriptor_set_layout(&dslci, None)
        .expect("failed to create descriptor set layout");
    handle
}

/// Bindless descriptor table.
#[derive(Debug)]
pub(crate) struct BindlessDescriptorTable {
    pub(crate) layout: vk::DescriptorSetLayout,
    pub(crate) set: vk::DescriptorSet,
    pool: vk::DescriptorPool,
    count: usize,
}

impl BindlessDescriptorTable {
    pub(crate) fn new(device: &ash::Device, descriptor_type: DescriptorType, count: usize) -> BindlessDescriptorTable {
        let layout = unsafe { create_bindless_layout(device, descriptor_type, count) };
        let pool = unsafe {
            device
                .create_descriptor_pool(
                    &vk::DescriptorPoolCreateInfo {
                        max_sets: 1,
                        pool_size_count: 1,
                        p_pool_sizes: &vk::DescriptorPoolSize {
                            ty: descriptor_type,
                            descriptor_count: count as u32,
                        },
                        ..Default::default()
                    },
                    None,
                )
                .expect("failed to create descriptor pool")
        };
        // and allocate a new descriptor set from it, copy old descriptors into it
        let set = unsafe {
            device
                .allocate_descriptor_sets(&vk::DescriptorSetAllocateInfo {
                    descriptor_pool: pool,
                    descriptor_set_count: 1,
                    p_set_layouts: &layout,
                    ..Default::default()
                })
                .expect("failed to allocate descriptor set")[0]
        };

        BindlessDescriptorTable {
            layout,
            pool,
            set,
            count,
        }
    }
}

impl Device {
    pub(super) unsafe fn write_global_texture_descriptor(&self, id: ImageViewId, image_view: vk::ImageView) {
        let d = self.inner.texture_descriptors.lock().unwrap();
        let dst_array_element = id.index();
        assert!(dst_array_element < d.count as u32);
        let write = vk::WriteDescriptorSet {
            dst_set: d.set,
            dst_binding: 0,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: DT::SAMPLED_IMAGE,
            p_image_info: &vk::DescriptorImageInfo {
                image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                ..Default::default()
            },
            ..Default::default()
        };
        trace!("texture_descriptors[{}] = {:?}", dst_array_element, image_view);
        self.update_descriptor_sets(&[write], &[]);
    }

    pub(super) unsafe fn write_global_storage_image_descriptor(&self, id: ImageViewId, image_view: vk::ImageView) {
        let d = self.inner.image_descriptors.lock().unwrap();
        let dst_array_element = id.index();
        assert!(dst_array_element < d.count as u32);
        let write = vk::WriteDescriptorSet {
            dst_set: d.set,
            dst_binding: 0,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: DT::STORAGE_IMAGE,
            p_image_info: &vk::DescriptorImageInfo {
                image_view,
                image_layout: vk::ImageLayout::GENERAL,
                ..Default::default()
            },
            ..Default::default()
        };
        trace!("image_descriptors[{}] = {:?}", dst_array_element, image_view);
        self.update_descriptor_sets(&[write], &[]);
    }

    pub(super) unsafe fn write_global_sampler_descriptor(&self, id: SamplerId, sampler: vk::Sampler) {
        let d = self.inner.sampler_descriptors.lock().unwrap();
        let dst_array_element = id.index();
        assert!(dst_array_element < d.count as u32);
        let write = vk::WriteDescriptorSet {
            dst_set: d.set,
            dst_binding: 0,
            dst_array_element,
            descriptor_count: 1,
            descriptor_type: DT::SAMPLER,
            p_image_info: &vk::DescriptorImageInfo {
                sampler,
                ..Default::default()
            },
            ..Default::default()
        };
        trace!("sampler_descriptors[{}] = {:?}", dst_array_element, sampler);
        self.inner.device.update_descriptor_sets(&[write], &[]);
    }
}
