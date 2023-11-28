//! Memory resources (images and buffers) and resource groups.
use super::{
    BufferHandle, BufferRegistrationInfo, BufferResource, BufferResourceCreateInfo, ImageHandle, ImageRegistrationInfo,
    ImageResource, ImageResourceCreateInfo, OwnerQueue, Resource, ResourceKind, ResourceRegistrationInfo,
};
use crate::{
    device::{BufferId, Device, GroupId, ImageId, ResourceAllocation, ResourceId},
    is_write_access, vk, MemoryLocation,
};
use ash::vk::Handle;
use gpu_allocator::vulkan::AllocationScheme;
use std::mem;

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

fn is_depth_and_stencil_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM_S8_UINT | vk::Format::D24_UNORM_S8_UINT | vk::Format::D32_SFLOAT_S8_UINT
    )
}

fn is_depth_only_format(fmt: vk::Format) -> bool {
    matches!(
        fmt,
        vk::Format::D16_UNORM | vk::Format::X8_D24_UNORM_PACK32 | vk::Format::D32_SFLOAT
    )
}

fn is_stencil_only_format(fmt: vk::Format) -> bool {
    matches!(fmt, vk::Format::S8_UINT)
}

fn aspects_for_format(fmt: vk::Format) -> vk::ImageAspectFlags {
    if is_depth_only_format(fmt) {
        vk::ImageAspectFlags::DEPTH
    } else if is_stencil_only_format(fmt) {
        vk::ImageAspectFlags::STENCIL
    } else if is_depth_and_stencil_format(fmt) {
        vk::ImageAspectFlags::DEPTH | vk::ImageAspectFlags::STENCIL
    } else {
        vk::ImageAspectFlags::COLOR
    }
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

unsafe fn destroy_resource(device: &Device, resource: &mut Resource) {
    if let ResourceAllocation::External = resource.allocation {
        return;
    }

    // destroy the object
    match resource.kind {
        ResourceKind::Buffer(ref buf) => device.device.destroy_buffer(buf.handle, None),
        ResourceKind::Image(ref img) => device.device.destroy_image(img.handle, None),
    };

    // release the associated memory
    // the mem::replace is there to move out the allocation object (which isn't Clone)
    match mem::replace(&mut resource.allocation, ResourceAllocation::External) {
        ResourceAllocation::Allocation { allocation } => device
            .allocator
            .borrow_mut()
            .free(allocation)
            .expect("failed to free memory"),
        ResourceAllocation::Transient { .. } => {
            todo!("destroy transient resources")
        }
        ResourceAllocation::DeviceMemory { device_memory } => {
            device.device.free_memory(device_memory, None);
        }
        ResourceAllocation::External => {
            unreachable!()
        }
    }
}

impl Device {
    /*unsafe fn destroy_resource(&self, id: ResourceId) {
        //trace!(?id, name = r.name.as_str(), tracking=?r.tracking, "destroy_resource");
        debug!("destroy_resource {:?}", id);
        let mut resources = self.resources.borrow_mut();
        let res = resources.remove(id).expect("invalid resource id");
        destroy_resource()
    }*/

    /*/// Destroys an image resource immediately.
    pub unsafe fn destroy_image(&self, id: ImageId) {
        self.destroy_resource(id.into());
    }

    /// Destroys a buffer resource immediately.
    pub unsafe fn destroy_buffer(&self, id: BufferId) {
        self.destroy_resource(id.into());
    }*/

    /// Marks a resource for deletion once the queue that currently owns it is done with it.
    unsafe fn destroy_resource(&self, id: ResourceId) {
        let mut resources = self.resources.borrow_mut();
        let resource = resources.get_mut(id).expect("invalid resource id");
        if resource.owner == OwnerQueue::None {
            resources.remove(id);
        } else {
            resource.discarded = true;
        }
    }

    /// Destroys an image resource.
    pub unsafe fn destroy_image(&self, id: ImageId) {
        self.destroy_resource(id.into());
    }

    /// Destroys a buffer resource.
    pub unsafe fn destroy_buffer(&self, id: BufferId) {
        self.destroy_resource(id.into());
    }

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
        self.resource_groups.borrow_mut().insert(ResourceGroup {
            //wait: Default::default(),
            src_stage_mask: Default::default(),
            dst_stage_mask,
            src_access_mask: Default::default(),
            dst_access_mask,
        })
    }

    /// Destroys a resource group.
    pub fn destroy_resource_group(&self, group_id: GroupId) {
        self.resource_groups.borrow_mut().remove(group_id);
    }

    /*/// Returns information about the current state of a frozen image resource.
    pub fn get_image_state(&self, image_id: ImageId) -> Option<ImageResourceState> {
        let objects = self.objects.lock().expect("failed to lock resources");
        let image = objects.resources.get(image_id.0).expect("invalid resource");
        image.group.map(|group_id| ImageResourceState {
            group_id,
            layout: image.tracking.layout,
        })
    }

    /// Returns information about the current state of a frozen buffer resource.
    pub fn get_buffer_state(&self, buffer_id: BufferId) -> Option<BufferResourceState> {
        let objects = self.objects.lock().expect("failed to lock resources");
        let buffer = objects.resources.get(buffer_id.0).expect("invalid resource");
        buffer.group.map(|group_id| BufferResourceState { group_id })
    }*/

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
    pub fn create_image(
        &self,
        name: &str,
        location: MemoryLocation,
        image_info: &ImageResourceCreateInfo,
    ) -> ImageHandle {
        // for now all resources are CONCURRENT, because that's the only way they can
        // be read across multiple queues.
        // Maybe exclusive ownership will be needed at some point, but then we should prevent
        // them from being used across multiple queues. I know that there's the possibility of doing
        // a "queue ownership transfer", but that shit is incomprehensible.
        let create_info = vk::ImageCreateInfo {
            image_type: image_info.type_,
            format: image_info.format,
            extent: image_info.extent,
            mip_levels: image_info.mip_levels,
            array_layers: image_info.array_layers,
            samples: get_vk_sample_count(image_info.samples),
            tiling: image_info.tiling,
            usage: image_info.usage,
            sharing_mode: vk::SharingMode::CONCURRENT,
            queue_family_index_count: self.queues_info.queue_count as u32,
            p_queue_family_indices: self.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .create_image(&create_info, None)
                .expect("failed to create image")
        };
        let mem_req = unsafe { self.device.get_image_memory_requirements(handle) };

        // allocate immediately
        let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .borrow_mut()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.device
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

        ImageHandle { id, vk: handle }
    }

    /// Creates a new buffer resource.
    ///
    /// Returns a `BufferInfo` struct containing the buffer resource ID, the vulkan buffer handle,
    /// and a pointer to the buffer mapped in host memory, if `buffer_create_info.map_on_create == true`.
    pub fn create_buffer(
        &self,
        name: &str,
        location: MemoryLocation,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferHandle {
        // create the buffer object first
        let create_info = vk::BufferCreateInfo {
            flags: Default::default(),
            size: buffer_create_info.byte_size,
            usage: buffer_create_info.usage,
            sharing_mode: if self.queues_info.queue_count == 1 {
                vk::SharingMode::EXCLUSIVE
            } else {
                vk::SharingMode::CONCURRENT
            },
            queue_family_index_count: self.queues_info.queue_count as u32,
            p_queue_family_indices: self.queues_info.families.as_ptr(),
            ..Default::default()
        };
        let handle = unsafe {
            self.device
                .create_buffer(&create_info, None)
                .expect("failed to create buffer")
        };

        // get its memory requirements
        let mem_req = unsafe { self.device.get_buffer_memory_requirements(handle) };

        // caller requested a mapped pointer, must create and allocate immediately
        let allocation_create_desc = gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_req,
            location,
            linear: true,
            allocation_scheme: AllocationScheme::GpuAllocatorManaged,
        };
        let allocation = self
            .allocator
            .borrow_mut()
            .allocate(&allocation_create_desc)
            .expect("failed to allocate device memory");
        unsafe {
            self.device
                .bind_buffer_memory(handle, allocation.memory(), allocation.offset())
                .unwrap();
        }
        let mapped_ptr = allocation.mapped_ptr();
        let allocation = ResourceAllocation::Allocation { allocation };

        /*let mapped_ptr = if buffer_create_info.map_on_create {
            let ptr = allocation.mapped_ptr().expect("failed to map buffer");
            //assert!(!ptr.is_null(), "failed to map buffer");
            ptr.as_ptr() as *mut u8
        } else {
            ptr::null_mut()
        };*/

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

        BufferHandle {
            id,
            vk: handle,
            mapped_ptr,
        }
    }

    /// Registers an existing buffer resource.
    pub unsafe fn register_buffer_resource(&self, info: BufferRegistrationInfo) -> BufferId {
        let id = self.register_resource(
            info.resource,
            ResourceKind::Buffer(BufferResource { handle: info.handle }),
        );
        BufferId(id)
    }

    /// Registers an existing image resource.
    pub unsafe fn register_image_resource(&self, info: ImageRegistrationInfo) -> ImageId {
        let id = self.register_resource(
            info.resource,
            ResourceKind::Image(ImageResource {
                handle: info.handle,
                format: info.format,
                all_aspects: aspects_for_format(info.format),
            }),
        );
        ImageId(id)
    }

    unsafe fn register_resource(&self, info: ResourceRegistrationInfo, kind: ResourceKind) -> ResourceId {
        let (object_type, object_handle) = match kind {
            ResourceKind::Buffer(ref buf) => (vk::ObjectType::BUFFER, buf.handle.as_raw()),
            ResourceKind::Image(ref img) => (vk::ObjectType::IMAGE, img.handle.as_raw()),
        };

        let id = self.resources.borrow_mut().insert(Resource {
            name: info.name.to_string(),
            discarded: false,
            kind,
            allocation: info.allocation,
            group: None,
            flush_mask: Default::default(),
            visible: Default::default(),
            layout: Default::default(),
            owner: OwnerQueue::None,
            stages: Default::default(),
            wait_semaphore: None,
            timestamp: 0,
        });

        self.set_debug_object_name(object_type, object_handle, info.name, None);
        id
    }

    // Cleanup expired resources on the specified queue.
    pub(crate) unsafe fn cleanup_queue_resources(&self, queue_index: usize, completed_timestamp: u64) {
        let mut resources = self.resources.borrow_mut();
        resources.retain(|_id, resource| {
            if resource.discarded
                && resource.owner == OwnerQueue::Exclusive(queue_index)
                && resource.timestamp <= completed_timestamp
            {
                destroy_resource(self, resource);
                false
            } else {
                true
            }
        })
    }
}

// Image { handle, Rc<Queue> }
