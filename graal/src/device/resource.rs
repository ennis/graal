//! Memory resources (images and buffers) and resource groups.
use std::{mem, ptr};

use ash::vk::Handle;
use gpu_allocator::vulkan::{AllocationCreateDesc, AllocationScheme};

use crate::{
    aspects_for_format,
    device::{BufferId, Device, GroupId, ImageId, ResourceAllocation, ResourceId},
    is_write_access, vk, Buffer, BufferCreateInfo, BufferUsage, Image, ImageCreateInfo, MemoryLocation, Size3D,
    TypedBuffer,
};

use super::{
    BufferRegistrationInfo, BufferResource, DeferredDeletionList, DeferredDeletionObject, ImageRegistrationInfo,
    ImageResource, OwnerQueue, RefCount, RefCounted, Resource, ResourceKind, ResourceRegistrationInfo,
};

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
    pub fn create_buffer(
        &self,
        name: &str,
        usage: BufferUsage,
        memory_location: MemoryLocation,
        byte_size: u64,
    ) -> Buffer {
        assert!(byte_size > 0, "buffer size must be greater than zero");

        // create the buffer object first
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: byte_size,
            usage: usage.into(),
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
            name,
            requirements: mem_req,
            location: memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .inner
            .allocator
            .borrow_mut()
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

        let id = unsafe {
            self.register_buffer_resource(BufferRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    initial_wait: None,
                    allocation,
                },
                handle,
            })
        };

        Buffer {
            device: self.clone(),
            id,
            handle,
            size: byte_size,
            usage,
            mapped_ptr,
        }
    }

    /// Registers an existing buffer resource.
    pub unsafe fn register_buffer_resource(&self, info: BufferRegistrationInfo) -> RefCounted<BufferId> {
        self.register_resource(
            info.resource,
            ResourceKind::Buffer(BufferResource { handle: info.handle }),
        )
        .map(BufferId)
    }

    /*/// Destroys a buffer resource.
    pub unsafe fn destroy_buffer(&self, id: BufferId) {
        self.destroy_resource(id.into());
    }*/

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
    pub fn create_image(&self, name: &str, image_info: &ImageCreateInfo) -> Image {
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

        // allocate immediately
        let allocation_create_desc = AllocationCreateDesc {
            name,
            requirements: mem_req,
            location: image_info.memory_location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .inner
            .allocator
            .borrow_mut()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.inner
                .device
                .bind_image_memory(handle, allocation.memory(), allocation.offset() as u64)
                .unwrap();
        }

        // register the resource in the context
        let id = unsafe {
            self.register_image_resource(ImageRegistrationInfo {
                resource: ResourceRegistrationInfo {
                    name,
                    allocation: ResourceAllocation::Allocation { allocation },
                    initial_wait: None,
                },
                handle,
                format: image_info.format,
            })
        };

        Image {
            device: self.clone(),
            id,
            handle,
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

    /// Registers an existing image resource.
    pub unsafe fn register_image_resource(&self, info: ImageRegistrationInfo) -> RefCounted<ImageId> {
        self.register_resource(
            info.resource,
            ResourceKind::Image(ImageResource {
                handle: info.handle,
                format: info.format,
                all_aspects: aspects_for_format(info.format),
            }),
        )
        .map(ImageId)
    }

    /*/// Destroys an image resource.
    pub unsafe fn destroy_image(&self, id: ImageId) {
        self.destroy_resource(id.into());
    }*/

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
        self.inner.resource_groups.borrow_mut().insert(ResourceGroup {
            //wait: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }

    /// Destroys a resource group.
    pub fn destroy_resource_group(&self, group_id: GroupId) {
        self.inner.resource_groups.borrow_mut().remove(group_id);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // NON-RESOURCE OBJECTS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    pub unsafe fn delete_later(&self, object: impl Into<DeferredDeletionObject>) {
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
    }

    fn cleanup_objects(&self) {
        let mut deletion_lists = self.inner.deletion_lists.borrow_mut();

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
                        DeferredDeletionObject::ImageView(view) => unsafe {
                            self.inner.device.destroy_image_view(*view, None);
                        },
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
        });
    }

    // Cleanup expired resources on the specified queue.
    pub(crate) unsafe fn cleanup_queue_resources(&self, queue_index: usize, completed_timestamp: u64) {
        let mut resources = self.inner.resources.borrow_mut();
        resources.retain(|_id, resource| {
            if resource.ref_count.is_unique()
                // TODO: the condition is always true for now
                && resource.owner == OwnerQueue::Exclusive(queue_index)
                && resource.timestamp <= completed_timestamp
            {
                self.destroy_resource_now(resource);
                false
            } else {
                true
            }
        });

        // We cleanup non-memory resources here, even though they're not associated with the queue.
        //
        // We could also do that in an hypothetical `Device::end_frame` method,
        // but we don't want to add another method that the user would need to call periodically
        // when there's already `Queue::end_frame` (which calls cleanup_queue_resources).
        self.cleanup_objects();
    }

    pub(crate) fn set_current_queue_timestamp(&self, queue: usize, timestamp: u64) {
        self.inner.queues[queue].next_submission_timestamp.set(timestamp);
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////
    // INTERNALS
    ////////////////////////////////////////////////////////////////////////////////////////////////

    // Not used anymore, replaced by a refcount
    /*/// Marks a resource for deletion once the queue that currently owns it is done with it.
    unsafe fn destroy_resource(&self, id: ResourceId) {
        let mut resources = self.inner.resources.borrow_mut();
        let resource = resources.get_mut(id).expect("invalid resource id");
        if resource.owner == OwnerQueue::None {
            // FIXME: and delete right now
            resources.remove(id);
        } else {
            resource.discarded = true;
        }
    }*/

    unsafe fn destroy_resource_now(&self, resource: &mut Resource) {
        if let ResourceAllocation::External = resource.allocation {
            return;
        }

        // destroy the object
        match resource.kind {
            ResourceKind::Buffer(ref buf) => self.inner.device.destroy_buffer(buf.handle, None),
            ResourceKind::Image(ref img) => self.inner.device.destroy_image(img.handle, None),
        };

        // release the associated memory
        // the mem::replace is there to move out the allocation object (which isn't Clone)
        match mem::replace(&mut resource.allocation, ResourceAllocation::External) {
            ResourceAllocation::Allocation { allocation } => self
                .inner
                .allocator
                .borrow_mut()
                .free(allocation)
                .expect("failed to free memory"),
            ResourceAllocation::DeviceMemory { device_memory } => {
                self.inner.device.free_memory(device_memory, None);
            }
            ResourceAllocation::External => {
                unreachable!()
            }
        }
    }

    unsafe fn register_resource(&self, info: ResourceRegistrationInfo, kind: ResourceKind) -> RefCounted<ResourceId> {
        let (object_type, object_handle) = match kind {
            ResourceKind::Buffer(ref buf) => (vk::ObjectType::BUFFER, buf.handle.as_raw()),
            ResourceKind::Image(ref img) => (vk::ObjectType::IMAGE, img.handle.as_raw()),
        };

        let ref_count = RefCount::new();
        let id = self.inner.resources.borrow_mut().insert(Resource {
            name: info.name.to_string(),
            ref_count: ref_count.clone(),
            kind,
            allocation: info.allocation,
            group: None,
            owner: OwnerQueue::None,
            timestamp: 0,
        });
        //self.inner.usages.borrow_mut().uses.insert(id, Default::default());
        self.set_debug_object_name(object_type, object_handle, info.name, None);
        RefCounted::new(id, ref_count)
    }
}
