use std::{
    ffi::{c_char, c_void, CString},
    mem,
    mem::MaybeUninit,
    ops::Deref,
    ptr,
    sync::Arc,
    time::Duration,
};

use ash::prelude::VkResult;
use fxhash::FxHashMap;

pub use compute::ComputeEncoder;
pub use render::{RenderEncoder, RenderPassInfo};

use crate::{
    aspects_for_format,
    device::{ActiveSubmission, QueueShared},
    vk, vk_ext_debug_utils, BufferAccess, BufferUntyped, CommandPool, Descriptor, Device, GpuResource, Image,
    ImageAccess, ImageId, ImageView, ImageViewId, MemoryAccess, Swapchain, SwapchainImage,
};

mod blit;
mod compute;
mod render;

////////////////////////////////////////////////////////////////////////////////////////////////////

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

/// Describes a pipeline barrier.
pub struct Barrier {
    access: MemoryAccess,
    transitions: Vec<(Image, MemoryAccess)>,
}

impl Barrier {
    pub fn new() -> Self {
        Barrier {
            access: MemoryAccess::empty(),
            transitions: vec![],
        }
    }

    pub fn color_attachment_write(mut self, image: &Image) -> Self {
        self.transitions
            .push((image.clone(), MemoryAccess::COLOR_ATTACHMENT_WRITE));
        self.access |= MemoryAccess::COLOR_ATTACHMENT_WRITE;
        self
    }

    pub fn depth_stencil_attachment_write(mut self, image: &Image) -> Self {
        self.transitions
            .push((image.clone(), MemoryAccess::DEPTH_STENCIL_ATTACHMENT_WRITE));
        self.access |= MemoryAccess::DEPTH_STENCIL_ATTACHMENT_WRITE;
        self
    }

    pub fn shader_storage_read(mut self) -> Self {
        self.access |= MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_storage_write(mut self) -> Self {
        self.access |= MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_read_image(mut self, image: &Image) -> Self {
        self.transitions.push((
            image.clone(),
            MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES,
        ));
        self.access |= MemoryAccess::SHADER_STORAGE_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn shader_write_image(mut self, image: &Image) -> Self {
        self.transitions.push((
            image.clone(),
            MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES,
        ));
        self.access |= MemoryAccess::SHADER_STORAGE_WRITE | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn present(mut self, image: &Image) -> Self {
        self.transitions.push((image.clone(), MemoryAccess::PRESENT));
        self.access |= MemoryAccess::PRESENT;
        self
    }

    pub fn sample_read_image(mut self, image: &Image) -> Self {
        self.transitions
            .push((image.clone(), MemoryAccess::SAMPLED_READ | MemoryAccess::ALL_STAGES));
        self.access |= MemoryAccess::SAMPLED_READ | MemoryAccess::ALL_STAGES;
        self
    }

    pub fn transfer_read(mut self) -> Self {
        self.access |= MemoryAccess::TRANSFER_READ;
        self
    }

    pub fn transfer_write(mut self) -> Self {
        self.access |= MemoryAccess::TRANSFER_WRITE;
        self
    }

    pub fn transfer_read_image(mut self, image: &Image) -> Self {
        self.transitions.push((image.clone(), MemoryAccess::TRANSFER_READ));
        self.access |= MemoryAccess::TRANSFER_READ;
        self
    }

    pub fn transfer_write_image(mut self, image: &Image) -> Self {
        self.transitions.push((image.clone(), MemoryAccess::TRANSFER_WRITE));
        self.access |= MemoryAccess::TRANSFER_WRITE;
        self
    }
}

///
pub struct CommandStream {
    pub(crate) device: Device,
    /// The queue on which we're submitting work.
    ///
    /// NOTE: for now, and most likely for the foreseeable future, we assume that
    /// there's only one command stream, so we can assume that we have exclusive access to
    /// the queue.
    queue: Arc<QueueShared>,
    command_pool: CommandPool,
    submission_index: u64,
    /// Binary semaphores for which we've submitted a wait operation.
    semaphores: Vec<UnsignaledSemaphore>,
    /// Command buffers waiting to be submitted.
    command_buffers_to_submit: Vec<vk::CommandBuffer>,
    /// Current command buffer.
    command_buffer: Option<vk::CommandBuffer>,

    // Buffer writes that need to be made available
    tracked_writes: MemoryAccess,
    tracked_images: FxHashMap<ImageId, CommandBufferImageState>,
    pub(crate) tracked_image_views: FxHashMap<ImageViewId, ImageView>,

    seen_initial_barrier: bool,
    //initial_writes: MemoryAccess,
    initial_access: MemoryAccess,
}

pub(crate) struct CommandBufferImageState {
    pub image: Image,
    pub first_access: MemoryAccess,
    pub last_access: MemoryAccess,
}

/// Describes how a resource will be used during a render or compute pass.
///
/// For now this library deals only with whole resources.
#[derive(Clone)]
pub enum ResourceUse<'a> {
    Image(&'a Image, ImageAccess),
    Buffer(&'a BufferUntyped, BufferAccess),
}

// A wrapper around a signaled binary semaphore.
//#[derive(Debug)]
//pub struct SignaledSemaphore(pub(crate) vk::Semaphore);

/// A wrapper around an unsignaled binary semaphore.
#[derive(Debug)]
pub struct UnsignaledSemaphore(pub(crate) vk::Semaphore);

/// Describes the kind of semaphore wait operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreWaitKind {
    /// Binary semaphore wait.
    Binary {
        /// The semaphore to wait on.
        semaphore: vk::Semaphore,
        /// Whether to transfer ownership of the semaphore to the queue.
        transfer_ownership: bool,
    },
    /// Timeline semaphore wait.
    Timeline { semaphore: vk::Semaphore, value: u64 },
    /// D3D12 fence wait.
    D3D12Fence {
        semaphore: vk::Semaphore,
        fence: vk::Fence,
        value: u64,
    },
}

/// Describe the kind of semaphore signal operation.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum SemaphoreSignal {
    /// Binary semaphore signal.
    Binary {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
    },
    /// Timeline semaphore signal.
    Timeline {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The value to signal.
        value: u64,
    },
    /// D3D12 fence signal.
    D3D12Fence {
        /// The semaphore to signal.
        semaphore: vk::Semaphore,
        /// The fence to signal.
        fence: vk::Fence,
        /// The value to signal.
        value: u64,
    },
}

/// Represents a semaphore wait operation.
#[derive(Clone, Debug)]
pub struct SemaphoreWait {
    /// The kind of wait operation.
    pub kind: SemaphoreWaitKind,
    /// Destination stage
    pub dst_stage: vk::PipelineStageFlags,
}

impl CommandStream {
    pub(super) fn new(device: Device, command_pool: CommandPool, queue: Arc<QueueShared>) -> CommandStream {
        //let submission_index = device.inner.tracker.lock().
        CommandStream {
            device,
            queue,
            command_pool,
            submission_index: 1,
            semaphores: vec![],
            command_buffers_to_submit: vec![],
            command_buffer: None,
            tracked_images: Default::default(),
            tracked_image_views: Default::default(),
            seen_initial_barrier: false,
            initial_access: MemoryAccess::empty(),
            tracked_writes: MemoryAccess::empty(),
        }
    }

    unsafe fn do_cmd_push_descriptor_set(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        set: u32,
        bindings: &[(u32, Descriptor)],
    ) {
        let mut descriptors = Vec::with_capacity(bindings.len());
        let mut descriptor_writes = Vec::with_capacity(bindings.len());

        for (binding, descriptor) in bindings {
            match *descriptor {
                Descriptor::SampledImage { ref image_view, layout } => {
                    self.reference_resource(image_view);
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.handle(),
                            image_layout: layout,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::StorageImage { ref image_view, layout } => {
                    self.reference_resource(image_view);
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: Default::default(),
                            image_view: image_view.handle(),
                            image_layout: layout,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
                Descriptor::UniformBuffer {
                    ref buffer,
                    offset,
                    size,
                } => {
                    self.reference_resource(buffer);
                    descriptors.push(DescriptorBufferOrImage {
                        buffer: vk::DescriptorBufferInfo {
                            buffer: buffer.handle(),
                            offset,
                            range: size,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                        p_buffer_info: &descriptors.last().unwrap().buffer,
                        ..Default::default()
                    });
                }
                Descriptor::StorageBuffer {
                    ref buffer,
                    offset,
                    size,
                } => {
                    self.reference_resource(buffer);
                    descriptors.push(DescriptorBufferOrImage {
                        buffer: vk::DescriptorBufferInfo {
                            buffer: buffer.handle(),
                            offset,
                            range: size,
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                        p_buffer_info: &descriptors.last().unwrap().buffer,
                        ..Default::default()
                    });
                }
                Descriptor::Sampler { ref sampler } => {
                    descriptors.push(DescriptorBufferOrImage {
                        image: vk::DescriptorImageInfo {
                            sampler: sampler.handle(),
                            image_view: Default::default(),
                            image_layout: Default::default(),
                        },
                    });
                    descriptor_writes.push(vk::WriteDescriptorSet {
                        dst_binding: *binding,
                        dst_array_element: 0,
                        descriptor_count: 1,
                        descriptor_type: vk::DescriptorType::SAMPLER,
                        p_image_info: &descriptors.last().unwrap().image,
                        ..Default::default()
                    });
                }
            }
        }

        unsafe {
            self.device.khr_push_descriptor().cmd_push_descriptor_set(
                command_buffer,
                bind_point,
                pipeline_layout,
                set,
                &descriptor_writes,
            );
        }
    }

    /// Binds push constants.
    fn do_cmd_push_constants(
        &mut self,
        command_buffer: vk::CommandBuffer,
        bind_point: vk::PipelineBindPoint,
        pipeline_layout: vk::PipelineLayout,
        data: &[MaybeUninit<u8>],
    ) {
        let size = mem::size_of_val(data);

        // Minimum push constant size guaranteed by Vulkan is 128 bytes.
        assert!(size <= 128, "push constant size must be <= 128 bytes");
        assert!(size % 4 == 0, "push constant size must be a multiple of 4 bytes");

        // None of the relevant drivers on desktop care about the actual stages,
        // only if it's graphics, compute, or ray tracing.
        let stages = match bind_point {
            vk::PipelineBindPoint::GRAPHICS => {
                vk::ShaderStageFlags::ALL_GRAPHICS | vk::ShaderStageFlags::MESH_EXT | vk::ShaderStageFlags::TASK_EXT
            }
            vk::PipelineBindPoint::COMPUTE => vk::ShaderStageFlags::COMPUTE,
            _ => panic!("unsupported bind point"),
        };

        // Use the raw function pointer because the wrapper takes a `&[u8]` slice which we can't
        // get from `&[MaybeUninit<u8>]` safely (even if we won't read uninitialized data).
        unsafe {
            (self.device.deref().fp_v1_0().cmd_push_constants)(
                command_buffer,
                pipeline_layout,
                stages,
                0,
                size as u32,
                data as *const _ as *const c_void,
            );
        }
    }

    /// Tells the command stream that an operation has made writes that are not available to
    /// subsequent operations.
    pub fn invalidate(&mut self, scope: MemoryAccess) {
        self.tracked_writes |= scope;
    }

    /// Emits a pipeline barrier (if necessary) that ensures that all previous writes are
    /// visible to subsequent operations for the given memory access type.
    ///
    /// Note that it's not possible to make only one specific type of write available. All pending
    /// writes are made available unconditionally.
    pub fn barrier(&mut self, barrier: Barrier) {
        let mut global_memory_barrier = None;
        let mut image_barriers = vec![];

        if !self.seen_initial_barrier {
            self.initial_access = barrier.access;
            self.seen_initial_barrier = true;
        } else {
            let (src_stage_mask, src_access_mask) = self.tracked_writes.to_vk_scope_flags();
            let (dst_stage_mask, dst_access_mask) = barrier.access.to_vk_scope_flags();
            global_memory_barrier = Some(vk::MemoryBarrier2 {
                src_access_mask,
                dst_access_mask,
                src_stage_mask,
                dst_stage_mask,
                ..Default::default()
            });
        }

        for (image, access) in barrier.transitions {
            if let Some(entry) = self.tracked_images.get_mut(&image.id()) {
                if entry.last_access != access {
                    let (src_stage_mask, src_access_mask) = entry.last_access.to_vk_scope_flags();
                    let (dst_stage_mask, dst_access_mask) = access.to_vk_scope_flags();
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask,
                        src_access_mask,
                        dst_stage_mask,
                        dst_access_mask,
                        image: image.handle(),
                        old_layout: entry.last_access.to_vk_image_layout(image.format),
                        new_layout: access.to_vk_image_layout(image.format),
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspects_for_format(image.format),
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        },
                        ..Default::default()
                    });
                }
                entry.last_access = access;
            } else {
                self.tracked_images.insert(
                    image.id(),
                    CommandBufferImageState {
                        image: image.clone(),
                        first_access: access,
                        last_access: access,
                    },
                );
            }
        }

        if global_memory_barrier.is_some() || !image_barriers.is_empty() {
            // a global memory barrier is needed or there are image layout transitions
            let command_buffer = self.get_or_create_command_buffer();
            unsafe {
                self.device.cmd_pipeline_barrier2(
                    command_buffer,
                    &vk::DependencyInfo {
                        dependency_flags: Default::default(),
                        memory_barrier_count: global_memory_barrier.iter().len() as u32,
                        p_memory_barriers: global_memory_barrier
                            .as_ref()
                            .map(|b| b as *const vk::MemoryBarrier2)
                            .unwrap_or(ptr::null()),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: ptr::null(),
                        image_memory_barrier_count: image_barriers.len() as u32,
                        p_image_memory_barriers: image_barriers.as_ptr(),
                        ..Default::default()
                    },
                );
            }
        }

        self.tracked_writes = barrier.access.write_flags();
    }

    /// Returns the device associated with this queue.
    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn push_debug_group(&mut self, label: &str) {
        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            let label = CString::new(label).unwrap();
            vk_ext_debug_utils().cmd_begin_debug_utils_label(
                command_buffer,
                &vk::DebugUtilsLabelEXT {
                    p_label_name: label.as_ptr(),
                    color: [0.0, 0.0, 0.0, 0.0],
                    ..Default::default()
                },
            );
        }
    }

    pub fn pop_debug_group(&mut self) {
        // TODO check that push/pop calls are balanced
        let command_buffer = self.get_or_create_command_buffer();
        unsafe {
            vk_ext_debug_utils().cmd_end_debug_utils_label(command_buffer);
        }
    }

    pub fn debug_group(&mut self, label: &str, f: impl FnOnce(&mut Self)) {
        self.push_debug_group(label);
        f(self);
        self.pop_debug_group();
    }

    pub fn reference_resource<R: GpuResource>(&mut self, resource: &R) {
        resource.set_last_submission_index(self.submission_index);
    }

    /// Creates a new, or returns an existing, binary semaphore that is in the unsignaled state,
    /// or for which we've submitted a wait operation on this queue and that will eventually be unsignaled.
    pub fn get_or_create_semaphore(&mut self) -> UnsignaledSemaphore {
        // Try to recycle one
        if let Some(semaphore) = self.semaphores.pop() {
            return semaphore;
        }

        // Otherwise create a new one
        unsafe {
            let create_info = vk::SemaphoreCreateInfo { ..Default::default() };
            UnsignaledSemaphore(
                self.device
                    .raw()
                    .create_semaphore(&create_info, None)
                    .expect("vkCreateSemaphore failed"),
            )
        }
    }

    /*pub fn use_image(&mut self, image: &Image, access: ImageAccess) {
        self.reference_resource(image);
        let id = image.id();
        if let Some(entry) = self.tracked_images.get_mut(&id) {
            if entry.last_state != access || !access.all_ordered() {
                self.pending_image_barriers.push(make_image_barrier(
                    entry.handle,
                    entry.format,
                    entry.last_state,
                    access,
                ));
            }
            entry.last_state = access;
        } else {
            self.tracked_images.insert(
                id,
                CommandBufferImageState {
                    image: image.clone(),
                    handle: image.handle,
                    format: image.format,
                    first_layout: access,
                    last_state: access,
                },
            );
        }
    }

    pub fn use_buffer(&mut self, buffer: &BufferUntyped, access: BufferAccess) {
        self.reference_resource(buffer);
        let id = buffer.id();
        if let Some(entry) = self.tracked_buffers.get_mut(&id) {
            if entry.last_state != access || !access.all_ordered() {
                self.pending_buffer_barriers
                    .push(make_buffer_barrier(entry.handle, entry.last_state, access));
            }
            entry.last_state = access;
        } else {
            self.tracked_buffers.insert(
                id,
                CommandBufferBufferState {
                    buffer: buffer.clone(),
                    handle: buffer.handle,
                    first_state: access,
                    last_state: access,
                },
            );
        }
    }*/

    /*pub fn use_image_view(&mut self, image_view: &ImageView, state: ImageAccess) {
        self.reference_resource(image_view);
        //self.use_image(image_view.image(), state);
    }*/

    pub(crate) fn create_command_buffer_raw(&mut self) -> vk::CommandBuffer {
        let raw_device = self.device.raw();
        let cb = self.command_pool.alloc(raw_device);

        unsafe {
            raw_device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .unwrap();
            self.device
                .set_object_name(cb, &format!("command buffer at submission {}", self.submission_index));
        }
        cb
    }

    unsafe fn set_current_command_buffer(&mut self, command_buffer: vk::CommandBuffer) {
        /*unsafe {
            self.device.raw().end_command_buffer(command_buffer).unwrap();
        }*/
        assert!(
            self.command_buffer.is_none(),
            "there is already a current command buffer"
        );
        self.command_buffer = Some(command_buffer);
    }

    /// Returns the current command buffer, creating a new one if necessary.
    ///
    /// The returned command buffer is ready to record commands.
    pub(crate) fn get_or_create_command_buffer(&mut self) -> vk::CommandBuffer {
        if let Some(cb) = self.command_buffer {
            cb
        } else {
            let cb = self.create_command_buffer_raw();
            self.command_buffer = Some(cb);
            cb
        }
    }

    /// Closes the current command buffer.
    ///
    /// This does nothing if there is no current command buffer.
    pub(crate) fn close_command_buffer(&mut self) {
        if let Some(cb) = self.command_buffer.take() {
            unsafe {
                self.device.raw().end_command_buffer(cb).unwrap();
            }
            self.command_buffers_to_submit.push(cb);
        }
    }

    /*/// Emits all pending pipeline barriers to the current command buffer,
    /// and clears the list of pending barriers.
    ///
    /// This is typically called before a draw or dispatch command.
    pub(crate) fn flush_barriers(&mut self) {

        /*if self.pending_image_barriers.is_empty() && self.pending_buffer_barriers.is_empty() {
            return;
        }*/

        if self.pending_image_barriers.is_empty() && self.pending_barrier.dst_stage_mask == vk::PipelineStageFlags::NONE && self.pending_barrier.dst_access_mask == vk::AccessFlags::empty() {
            return;
        }

        let command_buffer = self.get_or_create_command_buffer();
        // SAFETY: barriers are sound, the command buffer is valid, and the raw pointers are valid
        unsafe {
            self.device.cmd_pipeline_barrier2(
                command_buffer,
                &vk::DependencyInfo {
                    dependency_flags: Default::default(),
                    memory_barrier_count: 0,
                    p_memory_barriers: ptr::null(),
                    buffer_memory_barrier_count: self.pending_buffer_barriers.len() as u32,
                    p_buffer_memory_barriers: self.pending_buffer_barriers.as_ptr(),
                    image_memory_barrier_count: self.pending_image_barriers.len() as u32,
                    p_image_memory_barriers: self.pending_image_barriers.as_ptr(),
                    ..Default::default()
                },
            );
        }
        self.pending_buffer_barriers.clear();
        self.pending_image_barriers.clear();
    }*/

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(
        &mut self,
        swapchain: &Swapchain,
        timeout: Duration,
    ) -> Result<SwapchainImage, vk::Result> {
        // We can't use `get_or_create_semaphore` because according to the spec the semaphore
        // passed to `vkAcquireNextImage` must not have any pending operations, whereas
        // `get_or_create_semaphore` only guarantees that a wait operation has been submitted
        // on the semaphore (not that the wait has completed).
        let semaphore = {
            self.device
                .create_semaphore(&vk::SemaphoreCreateInfo { ..Default::default() }, None)
                .expect("vkCreateSemaphore failed")
        };
        let (image_index, _suboptimal) = match self.device.khr_swapchain().acquire_next_image(
            swapchain.handle,
            timeout.as_nanos() as u64,
            semaphore,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // delete the semaphore before returning
                unsafe {
                    self.device.destroy_semaphore(semaphore, None);
                }
                return Err(err);
            }
        };

        // Schedule deletion of the semaphore after the wait is over
        let device_clone = self.device.clone();
        self.device.call_later(self.submission_index, move || {
            // SAFETY: the semaphore is not used after this point
            unsafe {
                device_clone.destroy_semaphore(semaphore, None);
            }
        });

        // Wait for the image to be available.
        self.flush(
            &[SemaphoreWait {
                kind: SemaphoreWaitKind::Binary {
                    semaphore,
                    transfer_ownership: false,
                },
                // This is overly pessimistic, but we don't know what the user will do with the image at this point.
                dst_stage: vk::PipelineStageFlags::ALL_COMMANDS,
            }],
            &[],
        )?;

        // FIXME: why not register swapchain images once when creating the swap chain?
        let handle = swapchain.images[image_index as usize];
        let image =
            self.device
                .register_swapchain_image(handle, swapchain.format.format, swapchain.width, swapchain.height);

        Ok(SwapchainImage {
            swapchain: swapchain.handle,
            image,
            index: image_index,
        })
    }

    pub fn present(&mut self, swapchain_image: &SwapchainImage) -> VkResult<bool> {
        self.barrier(Barrier::new().present(&swapchain_image.image));

        // Signal a semaphore when rendering is finished
        let render_finished = self.get_or_create_semaphore().0;
        unsafe {
            self.device
                .set_object_name(render_finished, "render finished semaphore");
        }
        self.flush(
            &[],
            &[SemaphoreSignal::Binary {
                semaphore: render_finished,
            }],
        )?;

        // present the swapchain image
        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: &render_finished,
            swapchain_count: 1,
            p_swapchains: &swapchain_image.swapchain,
            p_image_indices: &swapchain_image.index,
            p_results: ptr::null_mut(),
            ..Default::default()
        };

        // SAFETY: ???
        let result = unsafe {
            self.device
                .khr_swapchain()
                .queue_present(self.queue.queue, &present_info)
        };

        // we signalled and waited on the semaphore, consider it consumed
        self.semaphores.push(UnsignaledSemaphore(render_finished));
        result
    }

    pub fn flush(&mut self, waits: &[SemaphoreWait], signals: &[SemaphoreSignal]) -> VkResult<()> {
        // Close the current command buffer if there is one
        self.close_command_buffer();

        let mut tracker = self.device.inner.tracker.lock().unwrap();

        // The complete list of command buffers to submit, including fixup command buffers between the ones passed to this function.
        let mut command_buffers = mem::take(&mut self.command_buffers_to_submit);

        // Update the state of each resource used by the command buffer in the device tracker,
        // and insert pipeline barriers if necessary.
        {
            let (src_stage_mask, src_access_mask) = tracker.writes.to_vk_scope_flags();
            let (dst_stage_mask, dst_access_mask) = self.initial_access.to_vk_scope_flags();
            // TODO: verify that a barrier is necessary
            let global_memory_barrier = Some(vk::MemoryBarrier2 {
                src_stage_mask,
                src_access_mask,
                dst_stage_mask,
                dst_access_mask,
                ..Default::default()
            });

            let mut image_barriers = Vec::new();
            for (_, state) in self.tracked_images.drain() {
                let prev_access = tracker
                    .images
                    .insert(state.image.id(), state.last_access)
                    .unwrap_or(MemoryAccess::UNINITIALIZED); // if the image was not previously tracked, the contents are undefined
                if prev_access != state.first_access {
                    let format = state.image.format;
                    image_barriers.push(vk::ImageMemoryBarrier2 {
                        src_stage_mask,
                        src_access_mask,
                        dst_stage_mask,
                        dst_access_mask,
                        old_layout: prev_access.to_vk_image_layout(format),
                        new_layout: state.first_access.to_vk_image_layout(format),
                        image: state.image.handle,
                        subresource_range: vk::ImageSubresourceRange {
                            aspect_mask: aspects_for_format(format),
                            base_mip_level: 0,
                            level_count: vk::REMAINING_MIP_LEVELS,
                            base_array_layer: 0,
                            layer_count: vk::REMAINING_ARRAY_LAYERS,
                        },
                        ..Default::default()
                    });
                }
            }

            tracker.writes = self.tracked_writes;
            self.tracked_writes = MemoryAccess::empty();
            self.seen_initial_barrier = false;

            // If we need a pipeline barrier before submitting the command buffers, we insert a "fixup" command buffer
            // containing the pipeline barrier, before the others.
            //if !buffer_barriers.is_empty() || !image_barriers.is_empty() {
            if global_memory_barrier.is_some() || !image_barriers.is_empty() {
                let fixup_command_buffer = self.command_pool.alloc(&self.device.raw());
                unsafe {
                    self.device
                        .begin_command_buffer(
                            fixup_command_buffer,
                            &vk::CommandBufferBeginInfo {
                                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                                ..Default::default()
                            },
                        )
                        .unwrap();
                    vk_ext_debug_utils().cmd_begin_debug_utils_label(
                        fixup_command_buffer,
                        &vk::DebugUtilsLabelEXT {
                            p_label_name: b"barrier fixup\0".as_ptr() as *const c_char,
                            color: [0.0, 0.0, 0.0, 0.0],
                            ..Default::default()
                        },
                    );
                    self.device.cmd_pipeline_barrier2(
                        fixup_command_buffer,
                        &vk::DependencyInfo {
                            dependency_flags: Default::default(),
                            memory_barrier_count: global_memory_barrier.iter().len() as u32,
                            p_memory_barriers: global_memory_barrier
                                .as_ref()
                                .map(|b| b as *const vk::MemoryBarrier2)
                                .unwrap_or(ptr::null()),
                            buffer_memory_barrier_count: 0,
                            p_buffer_memory_barriers: ptr::null(),
                            image_memory_barrier_count: image_barriers.len() as u32,
                            p_image_memory_barriers: image_barriers.as_ptr(),
                            ..Default::default()
                        },
                    );
                    vk_ext_debug_utils().cmd_end_debug_utils_label(fixup_command_buffer);
                    self.device.end_command_buffer(fixup_command_buffer).unwrap();
                }
                command_buffers.insert(0, fixup_command_buffer);
            }
        }

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        signal_semaphores.push(self.queue.timeline);

        // FIXME (!!!) if there are concurrent command streams, there is no guarantee that
        // they will be submitted in the order they were created. This means that`submission_index`
        // below is not necessarily increasing.
        //
        // TODO: probably disallow concurrent command streams entirely
        signal_semaphore_values.push(self.submission_index);

        // setup semaphore signal operations
        for signal in signals.iter() {
            match signal {
                SemaphoreSignal::Binary { semaphore } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                }
                SemaphoreSignal::Timeline { semaphore, value } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(*value);
                }
                SemaphoreSignal::D3D12Fence {
                    semaphore,
                    fence: _,
                    value: _,
                } => {
                    signal_semaphores.push(*semaphore);
                    signal_semaphore_values.push(0);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup semaphore wait operations
        for (_i, w) in waits.iter().enumerate() {
            wait_semaphore_dst_stages.push(w.dst_stage);
            match w.kind {
                SemaphoreWaitKind::Binary {
                    semaphore,
                    transfer_ownership,
                } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(0);
                    if transfer_ownership {
                        self.semaphores.push(UnsignaledSemaphore(semaphore));
                    }
                }
                SemaphoreWaitKind::Timeline { semaphore, value } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                }
                SemaphoreWaitKind::D3D12Fence { semaphore, value, .. } => {
                    wait_semaphores.push(semaphore);
                    wait_semaphore_values.push(value);
                    d3d12_fence_submit = true;
                }
            }
        }

        // setup D3D12 fence submissions
        let d3d12_fence_submit_info_ptr;
        let d3d12_fence_submit_info;

        if d3d12_fence_submit {
            d3d12_fence_submit_info = vk::D3D12FenceSubmitInfoKHR {
                wait_semaphore_values_count: wait_semaphore_values.len() as u32,
                p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
                signal_semaphore_values_count: signal_semaphore_values.len() as u32,
                p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
                ..Default::default()
            };
            d3d12_fence_submit_info_ptr = &d3d12_fence_submit_info as *const _ as *const c_void;
        } else {
            d3d12_fence_submit_info_ptr = ptr::null();
        }

        let timeline_submit_info = vk::TimelineSemaphoreSubmitInfo {
            p_next: d3d12_fence_submit_info_ptr,
            wait_semaphore_value_count: wait_semaphore_values.len() as u32,
            p_wait_semaphore_values: wait_semaphore_values.as_ptr(),
            signal_semaphore_value_count: signal_semaphore_values.len() as u32,
            p_signal_semaphore_values: signal_semaphore_values.as_ptr(),
            ..Default::default()
        };

        let submit_info = vk::SubmitInfo {
            p_next: &timeline_submit_info as *const _ as *const c_void,
            wait_semaphore_count: wait_semaphores.len() as u32,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            p_wait_dst_stage_mask: wait_semaphore_dst_stages.as_ptr(),
            command_buffer_count: command_buffers.len() as u32,
            p_command_buffers: command_buffers.as_ptr(),
            signal_semaphore_count: signal_semaphores.len() as u32,
            p_signal_semaphores: signal_semaphores.as_ptr(),
            ..Default::default()
        };

        // Do the submission
        let result = unsafe {
            self.device
                .queue_submit(self.queue.queue, &[submit_info], vk::Fence::null())
        };

        // We use one command pool per submission, so we retire the current one and create a new one.
        // The retired command pool will be reused once this submission has completed.
        let retired_command_pool = mem::replace(
            &mut self.command_pool,
            self.device.get_or_create_command_pool(self.queue.family),
        );

        tracker.active_submissions.push_back(ActiveSubmission {
            index: self.submission_index,
            //queue: self.queue.index,
            command_pools: vec![retired_command_pool],
        });

        // get the index of the next submission
        // TODO: just store the submission locally and assume that there's only one CommandStream
        self.submission_index = self.device.next_submission_index();

        result
    }
}
