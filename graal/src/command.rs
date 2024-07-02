use ash::prelude::VkResult;
use std::{
    ffi::{c_char, c_void, CString},
    mem,
    mem::MaybeUninit,
    ops::{Deref, DerefMut},
    ptr,
    sync::Arc,
    time::Duration,
};

use fxhash::FxHashMap;

pub use blit::BlitCommandEncoder;
pub use compute::ComputeEncoder;
pub use render::RenderEncoder;

use crate::{
    device::{ActiveSubmission, QueueShared},
    make_buffer_barrier, make_image_barrier, map_image_access_to_layout, vk, vk_ext_debug_utils, ArgumentKind,
    Arguments, BufferAccess, BufferId, BufferInner, BufferUntyped, CommandPool, Device, Format, Image, ImageAccess,
    ImageId, ImageInner, ImageType, ImageUsage, ImageView, ImageViewId, ImageViewInner, Size3D, Swapchain,
    SwapchainImage, VertexInput,
};

mod blit;
mod compute;
mod render;

////////////////////////////////////////////////////////////////////////////////////////////////////

enum DescriptorWrite {
    Image {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        image_view: vk::ImageView,
        format: Format,
        access: ImageAccess,
    },
    Buffer {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        buffer: vk::Buffer,
        access: BufferAccess,
        offset: u64,
        size: u64,
    },
    Sampler {
        binding: u32,
        descriptor_type: vk::DescriptorType,
        sampler: vk::Sampler,
    },
}

union DescriptorBufferOrImage {
    image: vk::DescriptorImageInfo,
    buffer: vk::DescriptorBufferInfo,
}

unsafe fn do_cmd_push_descriptor_sets(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    bind_point: vk::PipelineBindPoint,
    pipeline_layout: vk::PipelineLayout,
    set: u32,
    desc_writes: &[DescriptorWrite],
) {
    let mut descriptors = Vec::with_capacity(desc_writes.len());
    let mut descriptor_writes = Vec::with_capacity(desc_writes.len());

    for dw in desc_writes {
        match *dw {
            DescriptorWrite::Buffer {
                binding,
                descriptor_type,
                buffer,
                offset,
                size,
                access: _,
            } => {
                descriptors.push(DescriptorBufferOrImage {
                    buffer: vk::DescriptorBufferInfo {
                        buffer,
                        offset,
                        range: size,
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    // ignored for push descriptors
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_buffer_info: &descriptors.last().unwrap().buffer,
                    ..Default::default()
                });
            }
            DescriptorWrite::Image {
                binding,
                descriptor_type,
                image_view,
                access,
                format,
            } => {
                let image_layout = map_image_access_to_layout(access, format);
                descriptors.push(DescriptorBufferOrImage {
                    image: vk::DescriptorImageInfo {
                        sampler: Default::default(),
                        image_view,
                        image_layout,
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    // ignored for push descriptors
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_image_info: &descriptors.last().unwrap().image,
                    ..Default::default()
                });
            }
            DescriptorWrite::Sampler {
                sampler,
                binding,
                descriptor_type,
            } => {
                descriptors.push(DescriptorBufferOrImage {
                    image: vk::DescriptorImageInfo {
                        sampler,
                        image_view: Default::default(),
                        image_layout: Default::default(),
                    },
                });
                descriptor_writes.push(vk::WriteDescriptorSet {
                    dst_binding: binding,
                    dst_array_element: 0,
                    descriptor_count: 1,
                    descriptor_type,
                    p_image_info: &descriptors.last().unwrap().image,
                    ..Default::default()
                });
            }
        }
    }

    // TODO inline uniforms
    unsafe {
        device.khr_push_descriptor().cmd_push_descriptor_set(
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
    device: &Device,
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
        (device.deref().fp_v1_0().cmd_push_constants)(
            command_buffer,
            pipeline_layout,
            stages,
            0,
            size as u32,
            data as *const _ as *const c_void,
        );
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

///
pub struct CommandStream {
    pub(crate) device: Device,
    queue: Arc<QueueShared>,
    command_pool: CommandPool,
    /// Binary semaphores for which we've submitted a wait operation.
    semaphores: Vec<UnsignaledSemaphore>,
    /// Command buffers waiting to be submitted.
    command_buffers_to_submit: Vec<vk::CommandBuffer>,
    /// Current command buffer.
    command_buffer: Option<vk::CommandBuffer>,

    // TODO: hold a strong ref to the resources here, drop them only once the command stream is flushed.
    pub(crate) tracked_buffers: FxHashMap<BufferId, CommandBufferBufferState>,
    pub(crate) tracked_images: FxHashMap<ImageId, CommandBufferImageState>,
    pub(crate) tracked_image_views: FxHashMap<ImageViewId, ImageView>,

    pending_image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pending_buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
}

pub(crate) struct CommandBufferImageState {
    pub image: Image,
    pub handle: vk::Image,
    pub format: vk::Format,
    pub first_state: ImageAccess,
    pub last_state: ImageAccess,
}

pub(crate) struct CommandBufferBufferState {
    pub buffer: BufferUntyped,
    pub handle: vk::Buffer,
    pub first_state: BufferAccess,
    pub last_state: BufferAccess,
}

/// A wrapper around a signaled binary semaphore.
#[derive(Debug)]
pub struct SignaledSemaphore(pub(crate) vk::Semaphore);

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
        CommandStream {
            device,
            queue,
            command_pool,
            semaphores: vec![],
            command_buffers_to_submit: vec![],
            command_buffer: None,
            tracked_buffers: Default::default(),
            tracked_images: Default::default(),
            tracked_image_views: Default::default(),
            pending_image_barriers: vec![],
            pending_buffer_barriers: vec![],
        }
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

    pub(crate) fn use_image(&mut self, image: &Image, access: ImageAccess) {
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
                    first_state: access,
                    last_state: access,
                },
            );
        }
    }

    pub(crate) fn use_buffer(&mut self, buffer: &BufferUntyped, access: BufferAccess) {
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
    }

    pub(crate) fn use_image_view(&mut self, image_view: &ImageView, state: ImageAccess) {
        self.use_image(image_view.image(), state);
        self.tracked_image_views.insert(image_view.id(), image_view.clone());
    }

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

    /// Emits all pending pipeline barriers to the current command buffer,
    /// and clears the list of pending barriers.
    ///
    /// This is typically called before a draw or dispatch command.
    pub(crate) fn flush_barriers(&mut self) {
        if self.pending_image_barriers.is_empty() && self.pending_buffer_barriers.is_empty() {
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
    }

    /// Acquires the next image in a swapchain.
    pub unsafe fn acquire_next_swapchain_image(
        &mut self,
        swapchain: &Swapchain,
        timeout: Duration,
    ) -> Result<SwapchainImage, vk::Result> {
        let image_available = self.get_or_create_semaphore();
        let (image_index, _suboptimal) = match self.device.khr_swapchain().acquire_next_image(
            swapchain.handle,
            timeout.as_nanos() as u64,
            image_available.0,
            vk::Fence::null(),
        ) {
            Ok(result) => result,
            Err(err) => {
                // recycle the semaphore before returning
                self.semaphores.push(image_available);
                return Err(err);
            }
        };

        // Wait for the image to be available.
        // TODO this could be fused with the next submission
        self.flush(
            &[SemaphoreWait {
                kind: SemaphoreWaitKind::Binary {
                    semaphore: image_available.0,
                    transfer_ownership: true,
                },
                // This is overly pessimistic, but we don't know what the user will do with the image at this point.
                dst_stage: vk::PipelineStageFlags::ALL_COMMANDS,
            }],
            &[],
        )?;

        // FIXME we are doing nothing with the semaphore!
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
        self.use_image(&swapchain_image.image, ImageAccess::PRESENT);
        self.flush_barriers();

        // Signal a semaphore when rendering is finished
        let render_finished = self.get_or_create_semaphore().0;
        self.flush(
            &[],
            &[SemaphoreSignal::Binary {
                semaphore: render_finished,
            }],
        )?;

        /*// if necessary, insert an image layout transition command buffer
        let transition_cbuf;
        let mut tracker = self.device.inner.tracker.lock().unwrap();
        if let Some(access) = tracker.images.get_mut(swapchain_image.image.id()) {
            if *access != ImageAccess::PRESENT {
                let barrier = make_image_barrier(
                    swapchain_image.image.handle,
                    swapchain_image.image.format,
                    *access,
                    ImageAccess::PRESENT,
                );
                transition_cbuf = self.create_command_buffer_raw();
                self.device.cmd_pipeline_barrier2(
                    transition_cbuf,
                    &vk::DependencyInfo {
                        dependency_flags: Default::default(),
                        memory_barrier_count: 0,
                        p_memory_barriers: ptr::null(),
                        buffer_memory_barrier_count: 0,
                        p_buffer_memory_barriers: ptr::null(),
                        image_memory_barrier_count: 1,
                        p_image_memory_barriers: &barrier,
                        ..Default::default()
                    },
                );
                self.device.end_command_buffer(transition_cbuf).unwrap();
                submit_info.command_buffer_count = 1;
                submit_info.p_command_buffers = &transition_cbuf;
                *access = ImageAccess::PRESENT;
            }
        } else {
            panic!("image not found in tracker");
        }*/

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

        // Increment the global submission index and get the index of this submission
        tracker.last_submission_index += 1;
        let submission_index = tracker.last_submission_index;

        // Update the state of each resource used by the command buffer in the device tracker,
        // and insert pipeline barriers if necessary.
        //
        // At the same time, set the "last submission index" of each used resource to the index of this submission.
        let mut buffer_barriers = Vec::new();
        let mut image_barriers = Vec::new();

        for (_, state) in self.tracked_buffers.drain() {
            state.buffer.set_last_submission_index(submission_index);
            let last_access = tracker
                .buffers
                .insert(state.buffer.id(), state.last_state)
                .unwrap_or(BufferAccess::empty());

            if last_access != state.first_state || !state.first_state.all_ordered() {
                buffer_barriers.push(make_buffer_barrier(state.handle, last_access, state.first_state));
            }
        }

        for (_, state) in self.tracked_images.drain() {
            state.image.set_last_submission_index(submission_index);
            let last_access = tracker
                .images
                .insert(state.image.id(), state.last_state)
                .unwrap_or(ImageAccess::UNINITIALIZED);

            if last_access != state.first_state || !state.first_state.all_ordered() {
                image_barriers.push(make_image_barrier(
                    state.handle,
                    state.format,
                    last_access,
                    state.first_state,
                ));
            }
        }

        for (_, image_view) in self.tracked_image_views.drain() {
            // Just set the submission index, there's a separate entry in `tracked_images` for the underlying image.
            image_view.set_last_submission_index(submission_index);
        }

        // If we need a pipeline barrier before submitting the command buffers, we insert a "fixup" command buffer
        // containing the pipeline barrier, before the others.
        if !buffer_barriers.is_empty() || !image_barriers.is_empty() {
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
                        memory_barrier_count: 0,
                        p_memory_barriers: ptr::null(),
                        buffer_memory_barrier_count: buffer_barriers.len() as u32,
                        p_buffer_memory_barriers: buffer_barriers.as_ptr(),
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

        // Build submission
        let mut signal_semaphores = Vec::new();
        let mut signal_semaphore_values = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut wait_semaphore_values = Vec::new();
        let mut wait_semaphore_dst_stages = Vec::new();
        let mut d3d12_fence_submit = false;

        signal_semaphores.push(self.queue.timeline);
        signal_semaphore_values.push(submission_index);

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
            index: submission_index,
            queue: self.queue.index,
            command_pools: vec![retired_command_pool],
        });

        result
    }
}
