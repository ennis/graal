/*//! Resource state tracking and transition barriers
use crate::{
    aspects_for_format,
    map_buffer_access_to_barrier, map_image_access_to_barrier, map_image_access_to_layout, BufferId, BufferInner,
    BufferUntyped, Device, Format, Image, ImageId, ImageInner, ImageView, ImageViewId, ImageViewInner, RefCount,
};
use ash::{vk, vk::Handle};
use fxhash::FxHashMap;
use slotmap::SecondaryMap;
use std::{ptr, sync::Arc};

#[derive(Debug, thiserror::Error)]
pub(crate) enum UsageConflict {
    #[error("buffer {buffer:?} is used as {first_use:?} and {second_use:?} in the same scope")]
    IncompatibleBufferUses {
        buffer: BufferId,
        first_use: ResourceUse,
        second_use: ResourceUse,
    },
    #[error("image {image:?} is used as {first_use:?} and {second_use:?} in the same scope")]
    IncompatibleImageUses {
        image: ImageId,
        first_use: ResourceUse,
        second_use: ResourceUse,
    },
}

/*
#[derive(Copy, Clone)]
enum ImageOrBufferHandle {
    Image { image: vk::Image, format: vk::Format },
    Buffer(vk::Buffer),
}*/

pub(super) struct ImageBarrier {
    pub image: vk::Image,
    pub format: vk::Format,
    pub src: ResourceUse,
    pub dst: ResourceUse,
    // TODO subresources
}

pub(super) struct BufferBarrier {
    pub buffer: vk::Buffer,
    pub src: ResourceUse,
    pub dst: ResourceUse,
}

struct ImageTrackerEntry {
    image: Arc<ImageInner>,
    handle: vk::Image,
    format: vk::Format,
    first_state: ResourceUse,
    last_state: ResourceUse,
}

struct BufferTrackerEntry {
    buffer: Arc<BufferInner>,
    handle: vk::Buffer,
    first_state: ResourceUse,
    last_state: ResourceUse,
}

#[derive(Default)]
pub(super) struct PipelineBarrier {
    pub image_barriers: Vec<vk::ImageMemoryBarrier2>,
    pub buffer_barriers: Vec<vk::BufferMemoryBarrier2>,
}

impl PipelineBarrier {
    pub(super) fn is_empty(&self) -> bool {
        self.image_barriers.is_empty() && self.buffer_barriers.is_empty()
    }
}

pub(super) struct ArgumentBufferTracker {
    pub buffers: SecondaryMap<BufferId, (Arc<BufferInner>, ResourceUse)>,
    pub images: SecondaryMap<ImageId, (Arc<ImageInner>, ResourceUse)>,
    pub image_views: SecondaryMap<ImageViewId, Arc<ImageViewInner>>,
}

/// Global resource tracker.
pub(super) struct Tracker {
    pub buffers: SecondaryMap<BufferId, BufferTrackerEntry>,
    pub images: SecondaryMap<ImageId, ImageTrackerEntry>,
}

impl Tracker {
    pub(super) fn new() -> Self {
        Self {
            buffers: SecondaryMap::default(),
            images: SecondaryMap::default(),
            //image_views: SecondaryMap::default(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.buffers.clear();
        self.images.clear();
    }

    /// Iterates over all the buffers in the tracker
    pub(super) fn buffers(&self) -> impl Iterator<Item = &Arc<BufferInner>> {
        self.buffers.values().map(|entry| &entry.buffer)
    }

    pub(super) fn images(&self) -> impl Iterator<Item = &Arc<ImageInner>> {
        self.images.values().map(|entry| &entry.image)
    }

    /*pub(super) fn image_views(&self) -> impl Iterator<Item = &Arc<ImageViewInner>> {
        self.image_views.values()
    }*/

    #[inline]
    pub(super) fn use_buffer(
        &mut self,
        buffer: &BufferUntyped,
        state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<BufferBarrier>, UsageConflict> {
        self.use_buffer_inner(&buffer.inner, state, state, allow_barriers)
    }

    pub(super) fn use_buffer_inner(
        &mut self,
        buffer: &Arc<BufferInner>,
        state: ResourceUse,
        final_state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<BufferBarrier>, UsageConflict> {
        let mut barrier = None;
        // No barrier necessary if it's the first use
        if let Some(entry) = self.buffers.get_mut(buffer.id) {
            if entry.last_state != state || !state.all_ordered() {
                if allow_barriers {
                    barrier = Some(BufferBarrier {
                        buffer: buffer.handle,
                        src: entry.last_state,
                        dst: state,
                    })
                } else {
                    return Err(UsageConflict::IncompatibleBufferUses {
                        buffer: buffer.id,
                        first_use: entry.last_state,
                        second_use: state,
                    });
                }
            }
            entry.last_state = final_state;
        } else {
            self.buffers.insert(
                buffer.id,
                BufferTrackerEntry {
                    buffer: buffer.clone(),
                    handle,
                    first_state: state,
                    last_state: final_state,
                },
            );
        }

        Ok(barrier)
    }

    #[inline]
    pub(super) fn use_image(
        &mut self,
        image: &Image,
        state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<ImageBarrier>, UsageConflict> {
        self.use_image_inner(&image.inner, state, state, allow_barriers)
    }

    #[inline]
    pub(super) fn use_image_view(
        &mut self,
        image_view: &ImageView,
        state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<ImageBarrier>, UsageConflict> {
        self.use_image_inner(&image_view.inner.image, state, state, allow_barriers)
    }

    pub(super) fn use_image_inner(
        &mut self,
        image: &Arc<ImageInner>,
        state: ResourceUse,
        final_state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<ImageBarrier>, UsageConflict> {
        let mut barrier = None;
        // No barrier necessary if it's the first use
        if let Some(entry) = self.images.get_mut(image.id) {
            if entry.last_state != state || !state.all_ordered() {
                if allow_barriers {
                    //eprintln!("image barrier {:p}: from {:?} to {:?}", entry.handle,  entry.last_state, state);
                    barrier = Some(ImageBarrier {
                        image: image.handle,
                        format: entry.format,
                        src: entry.last_state,
                        dst: state,
                    })
                } else {
                    return Err(UsageConflict::IncompatibleImageUses {
                        image: image.id,
                        first_use: entry.last_state,
                        second_use: state,
                    });
                }
            }
            entry.last_state = final_state;
        } else {
            self.images.insert(
                image.id,
                ImageTrackerEntry {
                    image: image.clone(),
                    handle,
                    format,
                    first_state: state,
                    last_state: final_state,
                },
            );
        }

        Ok(barrier)
    }

    pub(super) fn dump(&self) {
        for (_id, entry) in self.images.iter() {
            eprintln!(
                "image {:p}: {:?} -> {:?}",
                entry.handle, entry.first_state, entry.last_state
            );
        }
    }

    pub(super) fn merge(&mut self, child_tracker: &Tracker) -> PipelineBarrier {
        let mut buffer_barriers = vec![];
        let mut image_barriers = vec![];

        for (id, entry) in child_tracker.buffers.iter() {
            let barrier = self
                .use_buffer_inner(&entry.buffer, entry.first_state, entry.last_state, true)
                .unwrap();
            if let Some(BufferBarrier { buffer, src, dst }) = barrier {
                let (ssrc, src_access_mask) = map_buffer_access_to_barrier(src);
                let (sdst, dst_access_mask) = map_buffer_access_to_barrier(dst);

                buffer_barriers.push(vk::BufferMemoryBarrier2 {
                    src_stage_mask: ssrc,
                    dst_stage_mask: sdst,
                    src_access_mask,
                    dst_access_mask,
                    src_queue_family_index: 0, // TODO?
                    dst_queue_family_index: 0,
                    buffer,
                    offset: 0,
                    size: vk::WHOLE_SIZE,
                    ..Default::default()
                })
            }
        }
        for (id, entry) in child_tracker.images.iter() {
            let barrier = self
                .use_image_inner(&entry.image, entry.first_state, entry.last_state, true)
                .unwrap();
            if let Some(ImageBarrier {
                image,
                format,
                src,
                dst,
            }) = barrier
            {
                let (ssrc, src_access_mask) = map_image_access_to_barrier(src);
                let (sdst, dst_access_mask) = map_image_access_to_barrier(dst);
                image_barriers.push(vk::ImageMemoryBarrier2 {
                    src_stage_mask: ssrc,
                    dst_stage_mask: sdst,
                    src_access_mask,
                    dst_access_mask,
                    old_layout: map_image_access_to_layout(src, format),
                    new_layout: map_image_access_to_layout(dst, format),
                    src_queue_family_index: 0, // TODO?
                    dst_queue_family_index: 0,
                    image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: aspects_for_format(format),
                        base_mip_level: 0,
                        level_count: vk::REMAINING_MIP_LEVELS,
                        base_array_layer: 0,
                        layer_count: vk::REMAINING_ARRAY_LAYERS,
                    },
                    ..Default::default()
                })
            }
        }

        PipelineBarrier {
            image_barriers,
            buffer_barriers,
        }
    }
}

pub(crate) unsafe fn emit_pipeline_barrier(device: &Device, command_buffer: vk::CommandBuffer, pb: &PipelineBarrier) {
    device.cmd_pipeline_barrier2(
        command_buffer,
        &vk::DependencyInfo {
            dependency_flags: Default::default(),
            memory_barrier_count: 0,
            p_memory_barriers: ptr::null(),
            buffer_memory_barrier_count: pb.buffer_barriers.len() as u32,
            p_buffer_memory_barriers: pb.buffer_barriers.as_ptr(),
            image_memory_barrier_count: pb.image_barriers.len() as u32,
            p_image_memory_barriers: pb.image_barriers.as_ptr(),
            ..Default::default()
        },
    );
}*/
