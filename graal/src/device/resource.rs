//! Memory resources (images and buffers) and resource groups.
use std::{
    ffi::c_void,
    mem, ptr,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
};

use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};
use tracing::debug;

use crate::{
    device::{Device, GroupId, ResourceAllocation},
    is_write_access, vk, BufferAccess, BufferInner, BufferUntyped, BufferUsage, Image, ImageCreateInfo, ImageInner,
    ImageView, ImageViewInfo, ImageViewInner, MemoryLocation, Resource, Size3D,
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
        let inner = unsafe { self.register_buffer(allocation, handle) };

        BufferUntyped {
            inner,
            handle,
            size: byte_size,
            usage,
            mapped_ptr,
        }
    }

    /// Registers an existing buffer resource.
    pub unsafe fn register_buffer(&self, allocation: ResourceAllocation, handle: vk::Buffer) -> Arc<BufferInner> {
        let device_address = self.get_buffer_device_address(&vk::BufferDeviceAddressInfo {
            buffer: handle,
            ..Default::default()
        });

        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let id = buffer_ids.insert_with_key(|id| {
            Arc::new(BufferInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation,
                group: None,
                handle,
                device_address,
            })
        });
        buffer_ids.get(id).unwrap().clone()
    }

    pub unsafe fn register_image(
        &self,
        allocation: ResourceAllocation,
        handle: vk::Image,
        format: vk::Format,
    ) -> Arc<ImageInner> {
        self.register_image_inner(allocation, handle, format, false)
    }

    pub(crate) fn register_swapchain_image(&self, handle: vk::Image, format: vk::Format) -> Arc<ImageInner> {
        self.register_image_inner(ResourceAllocation::External, handle, format, true)
    }

    fn register_image_inner(
        &self,
        allocation: ResourceAllocation,
        handle: vk::Image,
        format: vk::Format,
        swapchain_image: bool,
    ) -> Arc<ImageInner> {
        let mut image_ids = self.inner.image_ids.lock().unwrap();
        let id = image_ids.insert_with_key(|id| {
            Arc::new(ImageInner {
                device: self.clone(),
                id,
                last_submission_index: AtomicU64::new(0),
                allocation,
                group: None,
                handle,
                swapchain_image,
                format,
            })
        });
        image_ids.get(id).unwrap().clone()
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

        // register the resource in the context
        let inner =
            unsafe { self.register_image(ResourceAllocation::Allocation { allocation }, handle, image_info.format) };

        Image {
            handle,
            inner,
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

        let mut image_view_ids = self.inner.image_view_ids.lock().unwrap();
        let id = image_view_ids.insert_with_key(|id| {
            Arc::new(ImageViewInner {
                image: image.inner.clone(),
                id,
                handle,
                last_submission_index: AtomicU64::new(0),
            })
        });
        let inner = image_view_ids.get(id).unwrap().clone();

        ImageView {
            inner,
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

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NON-RESOURCE OBJECTS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    /*pub unsafe fn delete_later(&self, object: impl Into<DeferredDeletionObject>) {
        let object = object.into();

        // get next submission timestamps on all queues
        let mut next_submission_timestamps = vec![0; self.inner.queues.len()];
        for (i, q) in self.inner.queues.iter().enumerate() {
            next_submission_timestamps[i] = q.next_submission_timestamp.get();
        }

        // add the object to the current deletion list if the timestamps are the same
        // otherwise create a new deletion list with the up-to-date timestamps
        let mut deletion_lists = self.inner.deletion_lists.borrow_mut();
        if deletion_lists.is_empty() {
            deletion_lists.push(DeferredDeletionList {
                timestamps: next_submission_timestamps,
                objects: vec![object],
            });
        } else {
            let last = deletion_lists.last_mut().unwrap();
            if last.timestamps != next_submission_timestamps {
                deletion_lists.push(DeferredDeletionList {
                    timestamps: next_submission_timestamps,
                    objects: vec![object],
                });
            } else {
                last.objects.push(object);
            }
        }
    }*/

    fn cleanup_objects(&self) {
        /*let mut deletion_lists = self.inner.deletion_lists.borrow_mut();

        // fetch the completed timestamps on all queues
        let mut completed_timestamps = vec![0; self.inner.queues.len()];
        for (i, q) in self.inner.queues.iter().enumerate() {
            completed_timestamps[i] = unsafe { self.inner.device.get_semaphore_counter_value(q.timeline).unwrap() };
        }

        deletion_lists.retain(|list| {
            let expired = list
                .timestamps
                .iter()
                .zip(completed_timestamps.iter())
                .all(|(list_timestamp, completed_timestamp)| list_timestamp <= completed_timestamp);

            if expired {
                for obj in list.objects.iter() {
                    match obj {
                        DeferredDeletionObject::Sampler(sampler) => unsafe {
                            self.inner.device.destroy_sampler(*sampler, None);
                        },
                        DeferredDeletionObject::PipelineLayout(layout) => unsafe {
                            self.inner.device.destroy_pipeline_layout(*layout, None);
                        },
                        DeferredDeletionObject::Pipeline(pipeline) => unsafe {
                            self.inner.device.destroy_pipeline(*pipeline, None);
                        },
                        DeferredDeletionObject::Semaphore(semaphore) => unsafe {
                            self.inner.device.destroy_semaphore(*semaphore, None);
                        },
                        DeferredDeletionObject::Shader(shader) => unsafe {
                            self.inner.vk_ext_shader_object.destroy_shader(*shader, None);
                        },
                        DeferredDeletionObject::DescriptorSetLayout(layout) => unsafe {
                            self.inner.device.destroy_descriptor_set_layout(*layout, None);
                        },
                    }
                }
            }

            !expired
        });*/
    }

    // Called by `Buffer::drop`
    pub(crate) fn drop_buffer(&self, inner: &Arc<BufferInner>) {
        //self.inner.buffer_ids.lock().unwrap().remove(inner.id);
        //self.inner.tracker.lock().unwrap().buffers.remove(inner.id);
        self.delete_later(inner);
    }

    pub(crate) fn drop_image(&self, inner: &Arc<ImageInner>) {
        //self.inner.image_ids.lock().unwrap().remove(inner.id);
        //self.inner.tracker.lock().unwrap().images.remove(inner.id);
        self.delete_later(inner);
    }

    pub(crate) fn drop_image_view(&self, inner: &Arc<ImageViewInner>) {
        //self.inner.image_view_ids.lock().unwrap().remove(inner.id);
        self.delete_later(inner);
    }

    fn delete_later<R: Resource>(&self, resource: &Arc<R>) {
        // Otherwise, add it to the list of dropped resources. This list is cleaned-up in TODO.
        self.inner.dropped_resources.lock().unwrap().insert(resource.clone());
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
        let mut image_ids = self.inner.image_ids.lock().unwrap();
        let mut buffer_ids = self.inner.buffer_ids.lock().unwrap();
        let mut image_view_ids = self.inner.image_view_ids.lock().unwrap();
        let mut dropped_resources = self.inner.dropped_resources.lock().unwrap();

        // FIXME: we should also make sure that there's no pending reference to the resource in an unsubmitted command stream,
        // otherwise the resource will be destroyed on submission
        dropped_resources.images.retain(|id, image| {
            if image.last_submission_index.load(Ordering::Relaxed) <= last_completed_submission_index {
                image_ids.remove(*id);
                false
            } else {
                true
            }
        });
        dropped_resources.buffers.retain(|id, buffer| {
            if buffer.last_submission_index.load(Ordering::Relaxed) <= last_completed_submission_index {
                buffer_ids.remove(*id);
                false
            } else {
                true
            }
        });
        dropped_resources.image_views.retain(|id, image_view| {
            if image_view.last_submission_index.load(Ordering::Relaxed) <= last_completed_submission_index {
                image_view_ids.remove(*id);
                false
            } else {
                true
            }
        });

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
