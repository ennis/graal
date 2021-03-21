use crate::{context::{
    descriptor::DescriptorSet,
    format_aspect_mask, is_read_access, is_write_access,
    pass::{Pass, PassKind, ResourceAccess},
    resource::{
        BufferId, BufferInfo, ImageId, ResourceAccessDetails, ResourceId, ResourceKind,
        ResourceMemory, TypedBufferInfo,
    },
    BatchSerialNumber, InFlightBatch, SubmissionNumber, SwapchainImage,
}, descriptor::BufferDescriptor, device::QueuesInfo, vk, BufferResourceCreateInfo, Context, DescriptorSetInterface, Device, ImageInfo, ImageResourceCreateInfo, ResourceMemoryInfo, MAX_QUEUES, BufferData};
use ash::version::DeviceV1_0;
use std::{
    cell::{Ref, RefCell, RefMut},
    marker::PhantomData,
    mem,
    mem::MaybeUninit,
    ptr, slice,
};

type TemporarySet = std::collections::BTreeSet<ResourceId>;

/// Adds an execution dependency between a source and destination pass, identified by their submission numbers.
fn add_execution_dependency(
    src_snn: SubmissionNumber,
    src: Option<&mut Pass>,
    dst: &mut Pass,
    dst_stage_mask: vk::PipelineStageFlags,
) {
    if let Some(src) = src {
        // --- Intra-batch synchronization
        if src_snn.queue() != dst.snn.queue() {
            // cross-queue dependency w/ timeline semaphore
            src.signal_after = true;
            let q = src_snn.queue();
            dst.wait_before = true;
            dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
            dst.wait_dst_stages[q as usize] |= dst_stage_mask;
        } else {
            // same-queue dependency, a pipeline barrier is sufficient
            dst.src_stage_mask |= src.output_stage_mask;
        }

        dst.preds.push(src.batch_index);
        src.succs.push(dst.batch_index);
    } else {
        // --- Inter-batch synchronization w/ timeline semaphore
        let q = src_snn.queue();
        dst.wait_before = true;
        dst.wait_serials[q] = dst.wait_serials[q].max(src_snn.serial());
        dst.wait_dst_stages[q as usize] |= dst_stage_mask;
    }
}

/// Helper to find the pass given a submission number.
fn get_pass_mut<'a, 'b>(
    start_serial: u64,
    passes: &'a mut [Pass<'b>],
    snn: SubmissionNumber,
) -> Option<&'a mut Pass<'b>> {
    if snn.serial() <= start_serial {
        None
    } else {
        let pass_index = (snn.serial() - start_serial - 1) as usize;
        Some(&mut passes[pass_index])
    }
}

/// Builder object for passes.
pub struct PassBuilder<'a, 'batch> {
    batch: &'batch mut BatchInner<'a>,
    pass: Pass<'a>,
}

impl<'a, 'batch> PassBuilder<'a, 'batch> {
    /// Registers an image access made by this pass.
    pub fn add_image_usage(
        &mut self,
        id: ImageId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
        layout: vk::ImageLayout,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                layout,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    pub fn register_buffer_access(
        &mut self,
        id: BufferId,
        access_mask: vk::AccessFlags,
        input_stage: vk::PipelineStageFlags,
        output_stage: vk::PipelineStageFlags,
    ) {
        self.batch.add_resource_dependency(
            &mut self.pass,
            id.0,
            &ResourceAccessDetails {
                layout: vk::ImageLayout::UNDEFINED,
                access_mask,
                input_stage,
                output_stage,
            },
        )
    }

    /// Sets the command handler for this pass.
    /// The handler will be called when building the command buffer, on batch submission.
    pub fn set_commands(&mut self, commands: impl FnOnce(&Context, vk::CommandBuffer) + 'a) {
        self.pass.commands = Some(Box::new(commands));
    }
}

struct BatchInner<'a> {
    base_serial: u64,
    context: &'a mut Context,
    /// Map temporary index -> resource
    temporaries: Vec<ResourceId>,
    /// Set of all resources referenced in the batch
    temporary_set: TemporarySet,
    /// List of passes
    passes: Vec<Pass<'a>>,
    /// Image views created in this batch
    image_views: Vec<vk::ImageView>,
    /// Framebuffers created in this batch
    framebuffers: Vec<vk::Framebuffer>,
    descriptor_sets: Vec<DescriptorSet>,
}

impl<'a> BatchInner<'a> {
    /// Registers an access to a resource within the specified pass and updates the dependency graph.
    ///
    /// This is the meat of the automatic synchronization system: given the known state of the resources,
    /// this function infers the necessary execution barriers, memory barriers, and layout transitions,
    /// and updates the state of the resources.
    fn add_resource_dependency(
        &mut self,
        pass: &mut Pass,
        id: ResourceId,
        access: &ResourceAccessDetails,
    ) {
        //------------------------
        // first, add the resource into the set of temporaries used within this batch
        let resource = self.context.resources.get_mut(id).unwrap();
        if self.temporary_set.insert(id) {
            // this is the first time the resource has been used in the batch
            match resource.memory {
                ResourceMemory::Aliasable(_) => {
                    if resource.tracking.has_writer() || resource.tracking.has_readers() {
                        panic!("transient resource was already used in a previous batch")
                    }
                }
                _ => {}
            }

            self.temporaries.push(id);
        }

        //------------------------
        let is_write = !access.output_stage.is_empty() || resource.tracking.layout != access.layout;

        // update input stage mask
        pass.input_stage_mask |= access.input_stage;

        // handle external semaphore dependency
        let semaphore = mem::take(&mut resource.tracking.wait_binary_semaphore);
        if semaphore != vk::Semaphore::null() {
            pass.wait_binary_semaphores.push(semaphore);
            pass.wait_before = true;
        }

        //------------------------
        // infer execution dependencies
        if is_write {
            if !resource.tracking.has_readers() && resource.tracking.has_writer() {
                // write-after-write
                add_execution_dependency(
                    resource.tracking.writer,
                    get_pass_mut(self.base_serial, &mut self.passes, resource.tracking.writer),
                    pass,
                    access.input_stage,
                );
            } else {
                // write-after-read
                for q in 0..MAX_QUEUES as u8 {
                    if resource.tracking.readers[q] != 0 {
                        let src_snn = SubmissionNumber::new(q, resource.tracking.readers[q]);
                        add_execution_dependency(
                            src_snn,
                            get_pass_mut(self.base_serial, &mut self.passes, src_snn),
                            pass,
                            access.input_stage,
                        );
                    }
                }
            }
            // update the resource writer
            pass.output_stage_mask = access.output_stage;
        } else {
            if resource.tracking.has_writer() {
                // read-after-write
                // NOTE a read without a write is probably an uninitialized access
                add_execution_dependency(
                    resource.tracking.writer,
                    get_pass_mut(self.base_serial, &mut self.passes, resource.tracking.writer),
                    pass,
                    access.input_stage,
                );
            }
        }

        //------------------------
        // infer memory barriers

        // Q: do we need a memory barrier?
        // A: we need a memory barrier if
        //      - if the operation needs to see all previous writes to the resource:
        //          - if the resource visibility mask doesn't contain the requested access type
        //      - if a layout transition is necessary
        //
        // Note: if the pass overwrites the resource entirely, then the operation technically doesn't need to
        // see the last version of the resource.

        // are all writes to the resource visible to the requested access type?
        let writes_visible =
            // resource was last written in a previous batch, so all writes are made visible
            // by the semaphore wait inserted by the execution dependency
            resource.tracking.writer.serial() < self.base_serial ||
                // resource visible to all MEMORY_READ, or to the requested mask
                resource
                    .tracking
                    .visibility_mask
                    .contains(vk::AccessFlags::MEMORY_READ | access.access_mask);
        // is the layout of the resource different? do we need a transition?
        let layout_transition = resource.tracking.layout != access.layout;
        // is there a possible write-after-write hazard, that requires a memory dependency?
        let write_after_write_hazard =
            is_write && is_write_access(resource.tracking.availability_mask);

        if !writes_visible || layout_transition || write_after_write_hazard {
            // if the last writer of the serial is in another batch, all writes are made available (FIXME and visible?) because of the semaphore
            // wait inserted by the execution dependency. Otherwise, we need to consider the available writes on the resource.
            let src_access_mask = if resource.tracking.writer.serial() < self.base_serial {
                vk::AccessFlags::empty()
            } else {
                resource.tracking.availability_mask
            };
            // no need to make memory visible if we're only writing to the resource
            let dst_access_mask = if !is_read_access(access.access_mask) {
                vk::AccessFlags::empty()
            } else {
                access.access_mask
            };
            // the resource access needs a memory barrier
            match &resource.kind {
                ResourceKind::Image(img) => {
                    let subresource_range = vk::ImageSubresourceRange {
                        aspect_mask: format_aspect_mask(img.format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    };

                    pass.image_memory_barriers.push(vk::ImageMemoryBarrier {
                        src_access_mask,
                        dst_access_mask,
                        old_layout: resource.tracking.layout,
                        new_layout: access.layout,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        image: img.handle,
                        subresource_range,
                        ..Default::default()
                    })
                }
                ResourceKind::Buffer(buf) => {
                    pass.buffer_memory_barriers.push(vk::BufferMemoryBarrier {
                        src_access_mask,
                        dst_access_mask,
                        src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                        buffer: buf.handle,
                        offset: 0,
                        size: vk::WHOLE_SIZE,
                        ..Default::default()
                    })
                }
            }
            // all previous writes to the resource have been made available by the barrier ...
            resource.tracking.availability_mask = vk::AccessFlags::empty();
            // ... but not *made visible* to all access types: update the access types that can now see the resource
            resource.tracking.visibility_mask |= access.access_mask;
            resource.tracking.layout = access.layout;
        }

        // all previous writes are flushed
        if is_write_access(access.access_mask) {
            resource.tracking.availability_mask |= access.access_mask;
        }

        // update output stage
        // FIXME doubt
        if is_write {
            resource.tracking.stages = access.output_stage;
            resource.tracking.clear_readers();
            resource.tracking.writer = pass.snn;
        } else {
            // update the resource readers
            resource
                .tracking
                .readers
                .assign_max_serial(pass.snn.queue(), pass.snn.serial());
        }

        pass.accesses.push(ResourceAccess {
            id,
            access_mask: access.access_mask,
        });
    }
}

pub struct Batch<'a> {
    base_serial: u64,
    batch_serial: BatchSerialNumber,
    inner: RefCell<BatchInner<'a>>,
}

impl<'a> Batch<'a> {
    pub(crate) fn new(context: &'a mut Context) -> Batch<'a> {
        let base_serial = context.next_serial;
        Batch {
            base_serial,
            batch_serial: BatchSerialNumber(context.submitted_batch_count + 1),
            inner: RefCell::new(BatchInner {
                base_serial,
                context,
                temporaries: vec![],
                temporary_set: TemporarySet::new(),
                passes: vec![],
                image_views: vec![],
                framebuffers: vec![],
                descriptor_sets: vec![],
            }),
        }
    }

    /// Returns the context from which this batch was started
    pub fn context(&self) -> RefMut<Context> {
        RefMut::map(self.inner.borrow_mut(), |inner| inner.context)
    }

    /// Returns this batch's serial
    pub fn serial(&self) -> BatchSerialNumber {
        self.batch_serial
    }

    /// Creates a transient image.
    pub fn create_transient_image(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        image_create_info: &ImageResourceCreateInfo,
    ) -> ImageInfo {
        let mut inner = self.inner.borrow_mut();
        inner
            .context
            .create_image(name, memory_info, image_create_info, true)
    }

    /// Creates a transient buffer.
    pub fn create_transient_buffer(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {
        let mut inner = self.inner.borrow_mut();
        inner
            .context
            .create_buffer(name, memory_info, buffer_create_info, true)
    }

    /// Creates a transient mapped buffer. The caller must ensure that `memory_info` describes
    /// a mappable buffer (HOST_VISIBLE).
    pub fn create_mapped_buffer(
        &self,
        name: &str,
        memory_info: &ResourceMemoryInfo,
        buffer_create_info: &BufferResourceCreateInfo,
    ) -> BufferInfo {
        assert!(memory_info
            .required_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE));
        let mut inner = self.inner.borrow_mut();
        let buffer_info = inner
            .context
            .create_buffer(name, memory_info, buffer_create_info, true);
        assert!(!buffer_info.mapped_ptr.is_null());
        buffer_info
    }

    pub fn add_graphics_pass(&self, name: &str, handler: impl FnOnce(&mut PassBuilder<'a, '_>)) {
        self.add_pass(name, PassKind::Render, false, handler)
    }

    /// Starts building a compute pass
    pub fn add_compute_pass(
        &self,
        name: &str,
        async_compute: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassKind::Compute, async_compute, handler)
    }

    /// Starts building a transfer pass
    pub fn add_transfer_pass(
        &self,
        name: &str,
        async_transfer: bool,
        handler: impl FnOnce(&mut PassBuilder),
    ) {
        self.add_pass(name, PassKind::Transfer, async_transfer, handler)
    }

    /// Presents a swapchain image to the associated swapchain.
    pub fn present(&self, name: &str, image: &SwapchainImage) {
        self.add_pass(
            name,
            PassKind::Present {
                swapchain: image.swapchain_handle,
                image_index: image.image_index,
            },
            false,
            |builder| {
                builder.add_image_usage(
                    image.image_info.id,
                    vk::AccessFlags::MEMORY_READ,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::TOP_OF_PIPE,
                    vk::ImageLayout::PRESENT_SRC_KHR,
                );
            },
        );
    }

    /// Common code for `build_xxx_pass`
    fn add_pass(
        &self,
        name: &str,
        kind: PassKind,
        async_pass: bool,
        handler: impl FnOnce(&mut PassBuilder<'a, '_>),
    ) {
        let mut inner = self.inner.borrow_mut();
        let serial = inner.context.get_next_serial();
        let batch_index = inner.passes.len();
        let queue_index = match kind {
            PassKind::Compute if async_pass => inner.context.device.queues_info.indices.compute,
            PassKind::Transfer if async_pass => inner.context.device.queues_info.indices.transfer,
            PassKind::Present { .. } => inner.context.device.queues_info.indices.present,
            _ => inner.context.device.queues_info.indices.graphics,
        };
        let snn = SubmissionNumber::new(queue_index, serial);

        let mut builder = PassBuilder {
            batch: &mut *inner,
            pass: Pass::new(name, batch_index, snn, kind),
        };

        handler(&mut builder);

        let pass = builder.pass;
        inner.passes.push(pass);
    }

    /// Uploads the given object to a transient buffer with the given usage flags and returns the
    /// created buffer resource.
    pub fn upload<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &T,
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of::<T>();
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);
        unsafe {
            ptr::write(mapped_ptr as *mut T, *data);
        }
        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Uploads the given slice to a transient buffer with the given usage flags and returns the created buffer resource.
    ///
    /// Example:
    /// ```
    /// let vertices : [[f32;2];3] = [[0.0,0.0],[0.0,1.0],[1.0,1.0]];
    /// batch.upload_slice(graal::vk::BufferUsageFlags::VERTEX_BUFFER, &vertices, "vertices");
    /// ```
    pub fn upload_slice<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        data: &[T],
        name: Option<&str>,
    ) -> TypedBufferInfo<[T]> {
        let byte_size = mem::size_of_val(data);
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);

        unsafe {
            ptr::copy_nonoverlapping(data.as_ptr(), mapped_ptr as *mut T, data.len());
        }

        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: ptr::slice_from_raw_parts_mut(mapped_ptr as *mut T, data.len()),
        }
    }

    /// Allocates a buffer to hold a value of type `T` and returns a reference to the uninitialized
    /// value inside the buffer.
    ///
    /// Example:
    /// ```
    /// let (buf, val) = batch.alloc_upload::<[f32;4]>(graal::vk::BufferUsageFlags::UNIFORM_BUFFER, None);
    /// unsafe { val.as_mut_ptr().write(Default::default()); }
    /// ```
    pub fn alloc_upload<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        name: Option<&str>,
    ) -> TypedBufferInfo<T> {
        let byte_size = mem::size_of::<T>();
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);

        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: mapped_ptr as *mut T,
        }
    }

    /// Allocates a buffer to hold `size` values of type `T` and returns the slice of the uninitialized
    /// values inside the buffer.
    ///
    /// Example:
    /// ```
    /// let (buf, val) = batch.alloc_upload::<[f32;4]>(graal::vk::BufferUsageFlags::UNIFORM_BUFFER, None);
    /// unsafe { val.as_mut_ptr().write(Default::default()); }
    /// ```
    pub fn alloc_upload_slice<T: Copy>(
        &self,
        usage: vk::BufferUsageFlags,
        size: usize,
        name: Option<&str>,
    ) -> TypedBufferInfo<[T]> {
        let byte_size = mem::size_of::<T>() * size;
        let BufferInfo {
            id,
            handle,
            mapped_ptr,
        } = self.create_upload_buffer(usage, byte_size, name);
        TypedBufferInfo {
            id,
            handle,
            mapped_ptr: ptr::slice_from_raw_parts_mut(mapped_ptr as *mut T, size),
        }
    }

    /// Creates a transient buffer mapped in host-coherent memory.
    fn create_upload_buffer(
        &self,
        usage: vk::BufferUsageFlags,
        byte_size: usize,
        name: Option<&str>,
    ) -> BufferInfo {
        self.inner.borrow_mut().context.create_buffer(
            name.unwrap_or("upload buffer"),
            &ResourceMemoryInfo::HOST_VISIBLE_COHERENT,
            &BufferResourceCreateInfo {
                usage,
                byte_size: byte_size as u64,
                map_on_create: true,
            },
            true,
        )
    }

    /// Creates a descriptor set from an instance of a DescriptorSetInterface type.
    ///
    /// The returned descriptor set lives only for the duration of this batch.
    pub unsafe fn create_descriptor_set<'b, T: DescriptorSetInterface + 'b>(
        &'b self,
        descriptors: &T,
    ) -> vk::DescriptorSet {
        let serial = self.serial();
        let mut inner = self.inner.borrow_mut();
        let set = inner.context.create_descriptor_set(descriptors);
        inner.descriptor_sets.push(set);
        set.set
    }

    /// Creates a transient image view.
    ///
    /// Preconditions:
    /// - the provided `vk::ImageViewCreateInfo` must be valid.
    ///
    /// The returned `vk::ImageView` can only be used in this current batch and is automatically
    /// reclaimed after that.
    pub unsafe fn create_image_view(&self, create_info: &vk::ImageViewCreateInfo) -> vk::ImageView {
        let mut inner = self.inner.borrow_mut();
        let handle = inner
            .context
            .device
            .device
            .create_image_view(&create_info, None)
            .unwrap();
        inner.image_views.push(handle);
        handle
    }

    /// Creates a transient framebuffer.
    ///
    /// Preconditions:
    /// - render_pass must be a valid render pass object
    /// - attachment must contain only valid image views
    ///
    /// The returned framebuffer lives only for the duration of this batch.
    pub unsafe fn create_framebuffer(
        &self,
        width: u32,
        height: u32,
        layers: u32,
        render_pass: vk::RenderPass,
        attachments: &[vk::ImageView],
    ) -> vk::Framebuffer {
        unsafe {
            let framebuffer_create_info = vk::FramebufferCreateInfo {
                flags: Default::default(),
                render_pass,
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                width,
                height,
                layers,
                ..Default::default()
            };

            let mut inner = self.inner.borrow_mut();
            let handle = inner
                .context
                .device
                .device
                .create_framebuffer(&framebuffer_create_info, None)
                .unwrap();
            inner.framebuffers.push(handle);
            handle
        }
    }

    /// Finishes building the batch and submits all the passes to the command queues.
    pub fn finish(mut self) {
        println!("====== Batch #{} ======", self.batch_serial.0);
        /*println!("Passes:");
        for p in self.passes.iter() {
            println!("- `{}` ({:?})", p.name, p.snn);
            if p.wait_before {
                println!("    semaphore wait:");
                if p.wait_serials[0] != 0 {
                    println!("        0:{}|{:?}", p.wait_serials[0], p.wait_dst_stages[0]);
                }
                if p.wait_serials[1] != 0 {
                    println!("        1:{}|{:?}", p.wait_serials[1], p.wait_dst_stages[1]);
                }
                if p.wait_serials[2] != 0 {
                    println!("        2:{}|{:?}", p.wait_serials[2], p.wait_dst_stages[2]);
                }
                if p.wait_serials[3] != 0 {
                    println!("        3:{}|{:?}", p.wait_serials[3], p.wait_dst_stages[3]);
                }
            }
            println!(
                "    input execution barrier: {:?}->{:?}",
                p.src_stage_mask, p.input_stage_mask
            );
            println!("    input memory barriers:");
            for imb in p.image_memory_barriers.iter() {
                let id = self.context.image_resource_by_handle(imb.image);
                print!("        image handle={:?} ", imb.image);
                if !id.is_null() {
                    print!(
                        "(id={:?}, name={})",
                        id,
                        self.context.resources.get(id).unwrap().name
                    );
                } else {
                    print!("(unknown resource)");
                }
                println!(
                    " access_mask:{:?}->{:?} layout:{:?}->{:?}",
                    imb.src_access_mask, imb.dst_access_mask, imb.old_layout, imb.new_layout
                );
            }

            println!("    output stage: {:?}", p.output_stage_mask);
            if p.signal_after {
                println!("    semaphore signal: {:?}", p.snn);
            }
        }

        println!("Final resource states: ");
        for &id in self.temporaries.iter() {
            let resource = self.context.resources.get(id).unwrap();
            println!("`{}`", resource.name);
            println!("    stages={:?}", resource.tracking.stages);
            println!("    avail={:?}", resource.tracking.availability_mask);
            println!("    vis={:?}", resource.tracking.visibility_mask);
            println!("    layout={:?}", resource.tracking.layout);

            if resource.tracking.has_readers() {
                println!("    readers: ");
                if resource.tracking.readers[0] != 0 {
                    println!("        0:{}", resource.tracking.readers[0]);
                }
                if resource.tracking.readers[1] != 0 {
                    println!("        1:{}", resource.tracking.readers[1]);
                }
                if resource.tracking.readers[2] != 0 {
                    println!("        2:{}", resource.tracking.readers[2]);
                }
                if resource.tracking.readers[3] != 0 {
                    println!("        3:{}", resource.tracking.readers[3]);
                }
            }
            if resource.tracking.has_writer() {
                println!("    writer: {:?}", resource.tracking.writer);
            }
        }*/

        let mut inner = self.inner.into_inner();

        let context = inner.context;
        // First, wait for the frames submitted before the last one to finish, for pacing.
        // This also reclaims the resources referenced by the batch that are not in use anymore.
        context.wait_for_batches_in_flight();
        // Allocate and assign memory for all transient resources of this batch.
        let transient_allocations = context.allocate_memory_for_transients(
            self.base_serial,
            &inner.temporaries,
            &inner.passes,
        );
        // All resources now have a block of device memory assigned. We're ready to
        // build the command buffers and submit them to the device queues.
        let submission_result = context.submit_batch(&mut inner.passes);
        // Add this batch to the list of "batches in flight": batches that might be executing on the device.
        // When this batch is completed, all resources of the batch will be automatically recycled.
        // This includes:
        // - device memory blocks for transient allocations
        // - command buffers (in command pools)
        // - image views
        // - framebuffers
        // - descriptor sets
        context.in_flight.push_back(InFlightBatch {
            signalled_serials: submission_result.signalled_serials,
            transient_allocations,
            command_pools: submission_result.command_pools,
            image_views: vec![],
            framebuffers: vec![],
            descriptor_sets: inner.descriptor_sets,
            semaphores: submission_result.semaphores,
        });

        context.submitted_batch_count += 1;
        context.dump_state();
    }
}
