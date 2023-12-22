//! Resource state tracking and transition barriers
use crate::{
    aspects_for_format, map_buffer_use_to_barrier, map_texture_usage_to_barrier, map_texture_usage_to_layout, Buffer,
    BufferId, Device, Format, Image, ImageId, ImageView, ResourceId, ResourceUse,
};
use ash::vk;
use slotmap::SecondaryMap;
use std::ptr;

#[derive(Debug, thiserror::Error)]
pub(crate) enum UsageConflict {
    #[error("resource {resource:?} is used as {first_use:?} and {second_use:?} in the same scope")]
    IncompatibleUses {
        resource: ResourceId,
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
    id: ImageId,
    handle: vk::Image,
    format: vk::Format,
    first_state: ResourceUse,
    last_state: ResourceUse,
}

struct BufferTrackerEntry {
    id: BufferId,
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

pub(super) struct Tracker {
    // FIXME: having two maps here doesn't make sense because they share the same ID space (ResourceId)
    // They should be in different ID spaces
    buffers: SecondaryMap<ResourceId, BufferTrackerEntry>,
    images: SecondaryMap<ResourceId, ImageTrackerEntry>,
    // TODO this should also track references
}

impl Tracker {
    pub(super) fn new() -> Self {
        Self {
            buffers: SecondaryMap::default(),
            images: SecondaryMap::default(),
        }
    }

    pub(super) fn clear(&mut self) {
        self.buffers.clear();
        self.images.clear();
    }

    #[inline]
    pub(super) fn use_buffer(
        &mut self,
        buffer: &Buffer,
        state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<BufferBarrier>, UsageConflict> {
        self.use_buffer_raw(buffer.id().value, buffer.handle(), state, state, allow_barriers)
    }

    pub(super) fn use_buffer_raw(
        &mut self,
        id: BufferId,
        handle: vk::Buffer,
        state: ResourceUse,
        final_state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<BufferBarrier>, UsageConflict> {
        let mut barrier = None;
        // No barrier necessary if it's the first use
        if let Some(entry) = self.buffers.get_mut(id.resource_id()) {
            if entry.last_state != state || !state.all_ordered() {
                if allow_barriers {
                    barrier = Some(BufferBarrier {
                        buffer: handle,
                        src: entry.last_state,
                        dst: state,
                    })
                } else {
                    return Err(UsageConflict::IncompatibleUses {
                        resource: id.resource_id(),
                        first_use: entry.last_state,
                        second_use: state,
                    });
                }
                entry.last_state = final_state;
            }
        } else {
            self.buffers.insert(
                id.resource_id(),
                BufferTrackerEntry {
                    id,
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
        self.use_image_inner(
            image.id().value,
            image.handle(),
            image.format(),
            state,
            state,
            allow_barriers,
        )
    }

    #[inline]
    pub(super) fn use_image_view(
        &mut self,
        image: &ImageView,
        state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<ImageBarrier>, UsageConflict> {
        self.use_image_inner(
            image.parent_image.value,
            image.image_handle,
            image.format(),
            state,
            state,
            allow_barriers,
        )
    }

    pub(super) fn use_image_inner(
        &mut self,
        id: ImageId,
        handle: vk::Image,
        format: vk::Format,
        state: ResourceUse,
        final_state: ResourceUse,
        allow_barriers: bool,
    ) -> Result<Option<ImageBarrier>, UsageConflict> {
        let mut barrier = None;
        // No barrier necessary if it's the first use
        if let Some(entry) = self.images.get_mut(id.resource_id()) {
            if entry.last_state != state || !state.all_ordered() {
                if allow_barriers {
                    barrier = Some(ImageBarrier {
                        image: handle,
                        format: entry.format,
                        src: entry.last_state,
                        dst: state,
                    })
                } else {
                    return Err(UsageConflict::IncompatibleUses {
                        resource: id.resource_id(),
                        first_use: entry.last_state,
                        second_use: state,
                    });
                }
                entry.last_state = final_state;
            }
        } else {
            self.images.insert(
                id.resource_id(),
                ImageTrackerEntry {
                    id,
                    handle,
                    format,
                    first_state: state,
                    last_state: final_state,
                },
            );
        }

        Ok(barrier)
    }

    pub(super) fn merge(&mut self, child_tracker: &Tracker) -> PipelineBarrier {
        let mut buffer_barriers = vec![];
        let mut image_barriers = vec![];

        for (id, entry) in child_tracker.buffers.iter() {
            let barrier = self
                .use_buffer_raw(BufferId(id), entry.handle, entry.first_state, entry.last_state, true)
                .unwrap();
            if let Some(BufferBarrier { buffer, src, dst }) = barrier {
                let (ssrc, src_access_mask) = map_buffer_use_to_barrier(src);
                let (sdst, dst_access_mask) = map_buffer_use_to_barrier(dst);

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
                .use_image_inner(
                    ImageId(id),
                    entry.handle,
                    entry.format,
                    entry.first_state,
                    entry.last_state,
                    true,
                )
                .unwrap();
            if let Some(ImageBarrier {
                image,
                format,
                src,
                dst,
            }) = barrier
            {
                let (ssrc, src_access_mask) = map_texture_usage_to_barrier(src);
                let (sdst, dst_access_mask) = map_texture_usage_to_barrier(dst);
                image_barriers.push(vk::ImageMemoryBarrier2 {
                    src_stage_mask: ssrc,
                    dst_stage_mask: sdst,
                    src_access_mask,
                    dst_access_mask,
                    old_layout: map_texture_usage_to_layout(src, format),
                    new_layout: map_texture_usage_to_layout(dst, format),
                    src_queue_family_index: 0, // TODO?
                    dst_queue_family_index: 0,
                    image: image,
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

pub(crate) unsafe fn emit_pipeline_barrier(device: &Device, command_buffer: vk::CommandBuffer, pb: &PipelineBarrier)
{
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
}