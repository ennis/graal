use crate::{device::Device, vk};
use ordered_float::OrderedFloat;
use std::rc::Weak;

/*
macro_rules! impl_static_sampler_type {
    ($v:vis $name:ident, $mag:ident, $min:ident, $mipmap_mode:ident, $addr_u:ident, $addr_v:ident, $addr_w:ident) => {
        #[derive(Copy, Clone, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
        $v struct $name;
        impl SamplerType for $name {
            fn unique_type_id(&self) -> Option<TypeId> {
                Some(::std::any::TypeId::of::<Self>())
            }

            fn to_sampler(&self, device: &graal::ash::Device) -> vk::Sampler {
                const SAMPLER_CREATE_INFO: vk::SamplerCreateInfo = vk::SamplerCreateInfo {
                    s_type: vk::StructureType::SAMPLER_CREATE_INFO,
                    p_next: std::ptr::null(),
                    flags: vk::SamplerCreateFlags::empty(),
                    mag_filter: vk::Filter::$mag,
                    min_filter: vk::Filter::$min,
                    mipmap_mode: vk::SamplerMipmapMode::$mipmap_mode,
                    address_mode_u: vk::SamplerAddressMode::$addr_u,
                    address_mode_v: vk::SamplerAddressMode::$addr_v,
                    address_mode_w: vk::SamplerAddressMode::$addr_w,
                    mip_lod_bias: 0.0,
                    anisotropy_enable: 0,
                    max_anisotropy: 0.0,
                    compare_enable: vk::FALSE,
                    compare_op: vk::CompareOp::ALWAYS,
                    min_lod: 0.0,
                    max_lod: 0.0,
                    border_color: vk::BorderColor::INT_OPAQUE_BLACK,
                    unnormalized_coordinates: 0,
                };
                unsafe {
                    // bail out if we can't create a simple sampler object with no particular extensions
                    device.create_sampler(&SAMPLER_CREATE_INFO, None).expect("failed to create static sampler")
                }
            }
        }
    };
}

impl_static_sampler_type!(pub Linear_ClampToEdge, LINEAR, LINEAR, LINEAR, CLAMP_TO_EDGE, CLAMP_TO_EDGE, CLAMP_TO_EDGE);
impl_static_sampler_type!(pub Nearest_ClampToEdge, NEAREST, NEAREST, NEAREST, CLAMP_TO_EDGE, CLAMP_TO_EDGE, CLAMP_TO_EDGE);
*/

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

#[derive(Clone)]
pub struct Sampler {
    // A weak ref is sufficient, the device already owns samplers in its cache
    device: Weak<Device>,
    sampler: vk::Sampler,
}

impl Sampler {
    pub fn raw(&self) -> vk::Sampler {
        assert!(self.device.strong_count() > 0);
        self.sampler
    }
}
