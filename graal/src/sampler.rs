use std::rc::Weak;

use ordered_float::OrderedFloat;

use crate::{device::DeviceInner, vk, Device, WeakDevice};

#[derive(Clone, Debug, Hash, PartialEq, Eq, Ord, PartialOrd)]
pub struct SamplerCreateInfo {
    pub mag_filter: vk::Filter,
    pub min_filter: vk::Filter,
    pub mipmap_mode: vk::SamplerMipmapMode,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mip_lod_bias: OrderedFloat<f32>,
    pub anisotropy_enable: bool,
    pub max_anisotropy: OrderedFloat<f32>,
    pub compare_enable: bool,
    pub compare_op: vk::CompareOp,
    pub min_lod: OrderedFloat<f32>,
    pub max_lod: OrderedFloat<f32>,
    pub border_color: vk::BorderColor,
    pub unnormalized_coordinates: bool,
}

impl Default for SamplerCreateInfo {
    fn default() -> Self {
        SamplerCreateInfo {
            mag_filter: vk::Filter::LINEAR,
            min_filter: vk::Filter::LINEAR,
            mipmap_mode: vk::SamplerMipmapMode::LINEAR,
            address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
            mip_lod_bias: 0.0.into(),
            anisotropy_enable: false,
            max_anisotropy: 0.0.into(),
            compare_enable: false,
            compare_op: vk::CompareOp::ALWAYS,
            min_lod: 0.0.into(),
            max_lod: 0.0.into(),
            border_color: vk::BorderColor::INT_OPAQUE_BLACK,
            unnormalized_coordinates: false,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Sampler {
    // A weak ref is sufficient, the device already owns samplers in its cache
    device: WeakDevice,
    sampler: vk::Sampler,
}

impl Sampler {
    pub(super) fn new(device: &Device, sampler: vk::Sampler) -> Sampler {
        Sampler {
            device: device.weak(),
            sampler,
        }
    }

    pub fn handle(&self) -> vk::Sampler {
        // FIXME: check if the device is still alive, otherwise the sampler isn't valid anymore
        //assert!(self.device.strong_count() > 0);
        self.sampler
    }
}
