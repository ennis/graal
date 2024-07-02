//! Memory resources (images and buffers) and resource groups.
use std::{
    ffi::c_void,
    mem, ptr,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
};

use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use tracing::debug;

use crate::{
    device::{Device, GroupId, ResourceAllocation},
    is_write_access, vk, BufferAccess, BufferInner, BufferUntyped, BufferUsage, Image, ImageCreateInfo, ImageInner,
    ImageType, ImageUsage, ImageView, ImageViewInfo, ImageViewInner, MemoryLocation, Size3D,
};

use super::{DeferredDeletionList, DeferredDeletionObject};

pub(crate) struct ResourceGroup {
    // The timeline values that a pass needs to wait for to ensure an execution dependency between the pass
    // and all writers of the resources in the group.
    //pub(crate) wait: TimelineValues,
    // ignored if waiting on multiple queues
    pub(crate) src_stage_mask: vk::PipelineStageFlags2,
    pub(crate) dst_stage_mask: vk::PipelineStageFlags2,
    // ignored if waiting on multiple queues
    pub(crate) src_access_mask: vk::AccessFlags2,
    pub(crate) dst_access_mask: vk::AccessFlags2,
}

pub(crate) fn get_vk_sample_count(count: u32) -> vk::SampleCountFlags {
    match count {
        0 => vk::SampleCountFlags::TYPE_1,
        1 => vk::SampleCountFlags::TYPE_1,
        2 => vk::SampleCountFlags::TYPE_2,
        4 => vk::SampleCountFlags::TYPE_4,
        8 => vk::SampleCountFlags::TYPE_8,
        16 => vk::SampleCountFlags::TYPE_16,
        32 => vk::SampleCountFlags::TYPE_32,
        64 => vk::SampleCountFlags::TYPE_64,
        _ => panic!("unsupported number of samples"),
    }
}

impl Device {
    ////////////////////////////////////////////////////////////////////////////////////////////////
    // BUFFERS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates a new buffer resource.
    pub fn create_buffer(&self, usage: BufferUsage, memory_location: MemoryLocation, byte_size: u64) -> BufferUntyped {
        assert!(byte_size > 0, "buffer size must be greater than zero");

        // create the buffer object first
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: byte_size,
            usage: usage.to_vk_buffer_usage_flags() | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = unsafe {
            self.inner
                .device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };

        // get its memory requirements
        let mem_req = unsafe { self.inner.device.get_buffer_memory_requirements(handle) };

        let allocation_create_desc = AllocationCreateDesc {
            name: "",
            requirements: mem_req,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .inner
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.inner
                .device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
                .unwrap();
        }
        let mapped_ptr = allocation.mapped_ptr();
        let allocation = ResourceAllocation::Allocation { allocation };

        let device_address = unsafe {
            self.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
                buffer: handle,
                ..Default::default()
            })
        };

        let id = self.inner.buffer_ids.lock().unwrap().insert(());
        BufferUntyped {
            inner: Some(Arc::new(BufferInner {
                device: self.clone(),
                id,
                user_ref_count: AtomicU32::new(1),
                last_submission_index: AtomicU64::new(0),
                allocation,
                group: None,
                handle,
                device_address,
            })),
            handle,
            size: byte_size,
            usage,
            mapped_ptr,
        }
    }

    /*/// Registers an existing buffer resource.
    pub(crate) unsafe fn register_buffer(
        &self,
        allocation: ResourceAllocation,
        handle: vk::Buffer,
    ) -> Arc<BufferInner> {
        let device_address = self.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
            buffer: handle,
            ..Default::default()
        });

        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let id = buffer_ids.insert_with_key(|id| {
            Arc::new(BufferInner {
                device: self.clone(),
                id,
                user_ref_count: AtomicU32::new(1),
                last_submission_index: AtomicU64::new(0),
                allocation,
                group: None,
                handle,
                device_address,
            })
        });
        buffer_ids.get(id).unwrap().clone()
    }*/

    pub(crate) fn register_swapchain_image(
        &self,
        handle: vk::Image,
        format: vk::Format,
        width: u32,
        height: u32,
    ) -> Image {
        let id = self.inner.image_ids.lock().unwrap().insert(());
        Image {
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::External,
                group: None,
                handle,
                swapchain_image: true,
                format,
            })),
            handle,
            usage: ImageUsage::TRANSFER_DST | ImageUsage::COLOR_ATTACHMENT,
            type_: ImageType::Image2D,
            format,
            size: Size3D {
                width,
                height,
                depth: 1,
            },
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // IMAGES
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates a new image resource.
    ///
    /// Returns an `ImageInfo` struct containing the image resource ID and the vulkan image handle.
    ///
    /// # Notes
    /// The image might not have any device memory attached when this function returns.
    /// This is because graal may delay the allocation and binding of device memory until the end of the
    /// current frame (see `Context::end_frame`).
    ///
    /// # Examples
    ///
    pub fn create_image(&self, image_info: &ImageCreateInfo) -> Image {
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.type_.into(),
            format: image_info.format,
            extent: vk::Extent3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: vk::ImageTiling::OPTIMAL, // LINEAR tiling not used enough to be exposed
            usage: image_info.usage.into(),
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            ..Default::default()
        };
        let handle = unsafe {
            self.inner
                .device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.inner.device.get_image_memory_requirements(handle) };

        let allocation_create_desc = AllocationCreateDesc {
            name: "",
            requirements: mem_req,
            location: image_info.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .inner
            .allocator
            .lock()
            .unwrap()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.inner
                .device
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();
        }

        let id = self.inner.image_ids.lock().unwrap().insert(());

        Image {
            handle,
            inner: Some(Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation: ResourceAllocation::Allocation { allocation },
                group: None,
                handle,
                swapchain_image: false,
                format: image_info.format,
            })),
            usage: image_info.usage,
            type_: image_info.type_,
            format: image_info.format,
            size: Size3D {
                width: image_info.width,
                height: image_info.height,
                depth: image_info.depth,
            },
        }
    }

    pub fn create_image_view(&self, image: &Image, info: &ImageViewInfo) -> ImageView {
        // FIXME: support non-zero base mip level
        if info.subresource_range.base_mip_level != 0 {
            unimplemented!("non-zero base mip level");
        }

        let create_info = vk::ImageViewCreateInfo {
            flags: vk::ImageViewCreateFlags::empty(),
            image: image.handle,
            view_type: info.view_type,
            format: info.format,
            components: vk::ComponentMapping {
                r: info.component_mapping[0],
                g: info.component_mapping[1],
                b: info.component_mapping[2],
                a: info.component_mapping[3],
            },
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: info.subresource_range.aspect_mask,
                base_mip_level: info.subresource_range.base_mip_level,
                level_count: info.subresource_range.level_count,
                base_array_layer: info.subresource_range.base_array_layer,
                layer_count: info.subresource_range.layer_count,
            },
            ..Default::default()
        };

        // SAFETY: the device is valid, the create info is valid
        let handle = unsafe {
            self.inner
                .device
                .create_image_view(&create_info, None)
                .expect("failed to create image view")
        };

        let id = self.inner.image_view_ids.lock().unwrap().insert(());

        ImageView {
            inner: Some(Arc::new(ImageViewInner {
                image: image.clone(),
                id,
                handle,
                last_submission_index: AtomicU64::new(0),
            })),
            image: image.id(),
            handle,
            image_handle: image.handle,
            format: info.format,
            original_format: image.format,
            // TODO: size of mip level
            size: image.size,
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // RESOURCE GROUPS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /// Creates a resource group.
    ///
    /// Resource group hold a set of static resources that can be synchronized with as a group.
    /// This is useful for large sets of long-lived static resources, like texture maps,
    /// where it would be impractical to synchronize on each of them individually.
    pub fn create_resource_group(
        &self,
        dst_stage_mask: vk::PipelineStageFlags2,
        dst_access_mask: vk::AccessFlags2,
    ) -> GroupId {
        // resource groups are for read-only resources
        assert!(!is_write_access(dst_access_mask));
        self.inner.groups.lock().unwrap().insert(ResourceGroup {
            //wait: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }

    /// Destroys a resource group.
    pub fn destroy_resource_group(&self, group_id: GroupId) {
        self.inner.groups.lock().unwrap().remove(group_id);
    }

    /*
    /// Called when the last user reference to a resource is dropped.
    pub(crate) fn drop_resource<R: Resource>(&self, resource: &Arc<R>) {
        // The last user reference to the resource has been dropped.
        // However, there might still be pending references to the resource in command buffers
        // that have not been submitted yet.
        // In this case, the resource's `last_submission_index` will still be
        //

        // add it to the list of dropped resources
        self.inner.dropped_resources.lock().unwrap().insert(resource.clone());
    }*/

    pub(crate) fn delete_later<T: 'static>(&self, submission_index: u64, object: T) {
        let queue_timeline = self.inner.queues[0].timeline;
        let last_completed_submission_index = unsafe {
            self.get_semaphore_counter_value(queue_timeline)
                .expect("get_semaphore_counter_value failed")
        };
        if submission_index <= last_completed_submission_index {
            // drop the object immediately if the submission has completed
            return;
        }

        // otherwise move it to the deferred deletion list
        self.inner
            .dropped_resources
            .lock()
            .unwrap()
            .push((submission_index, Box::new(object)));
    }

    pub(crate) unsafe fn free_memory(&self, allocation: &mut ResourceAllocation) {
        match mem::replace(allocation, ResourceAllocation::External) {
            ResourceAllocation::Allocation { allocation } => self
                .inner
                .allocator
                .lock()
                .unwrap()
                .free(allocation)
                .expect("failed to free memory"),
            ResourceAllocation::DeviceMemory { device_memory } => unsafe {
                self.inner.device.free_memory(device_memory, None);
            },
            ResourceAllocation::External => {
                unreachable!()
            }
        }
    }

    // Cleanup expired resources.
    pub fn cleanup(&self) {
        // TODO multiple queues
        let queue_timeline = self.inner.queues[0].timeline;
        let last_completed_submission_index = unsafe {
            self.get_semaphore_counter_value(queue_timeline)
                .expect("get_semaphore_counter_value failed")
        };

        let mut tracker = self.inner.tracker.lock().unwrap();
        /*let mut image_ids = self.inner.image_ids.lock().unwrap();
        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let mut image_view_ids = self.inner.image_view_ids.lock().unwrap();*/
        let mut dropped_resources = self.inner.dropped_resources.lock().unwrap();

        dropped_resources.retain(|(submission, object)| *submission > last_completed_submission_index);

        // process all completed submissions, oldest to newest
        //let mut active_submissions = tracker.active_submissions.lock().unwrap();
        let mut free_command_pools = self.inner.free_command_pools.lock().unwrap();

        loop {
            let Some(submission) = tracker.active_submissions.front() else {
                break;
            };
            if submission.index > last_completed_submission_index {
                break;
            }
            debug!("cleaning up submission {}", submission.index);
            let submission = tracker.active_submissions.pop_front().unwrap();
            for mut command_pool in submission.command_pools {
                // SAFETY: command buffers are not in use anymore
                unsafe {
                    command_pool.reset(&self.inner.device);
                }
                free_command_pools.push(command_pool);
            }
        }

        // We cleanup non-memory resources here, even though they're not associated with the queue.
        //
        // We could also do that in an hypothetical `Device::end_frame` method,
        // but we don't want to add another method that the user would need to call periodically
        // when there's already `Queue::end_frame` (which calls cleanup_queue_resources).
        //self.cleanup_objects();
    }

    pub(crate) fn set_current_queue_timestamp(&self, queue: usize, timestamp: u64) {
        self.inner.queues[queue].next_submission_timestamp.set(timestamp);
    }
}
