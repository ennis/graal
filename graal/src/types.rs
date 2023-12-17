use std::{borrow::Cow, path::Path};

// TODO: eventually all vk types should disappear from the public API
use ash::vk;
use bitflags::bitflags;
use gpu_allocator::MemoryLocation;
use ordered_float::OrderedFloat;

pub type Format = vk::Format;

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    /// Front face
    Front = 0,
    /// Back face
    Back = 1,
}

/// Describe a subresource range of an image.
///
/// Same as VkImageSubresourceRange, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceRange {
    pub aspect_mask: vk::ImageAspectFlags,
    pub base_mip_level: u32,
    pub level_count: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

#[derive(Copy, Clone, Debug)]
pub struct ImageDataLayout {
    pub offset: u64,
    /// In texels.
    // TODO make that bytes
    pub row_length: Option<u32>,
    /// In lines.
    pub image_height: Option<u32>,
}

#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageSubresourceLayers {
    pub aspect_mask: vk::ImageAspectFlags,
    pub mip_level: u32,
    pub base_array_layer: u32,
    pub layer_count: u32,
}

/// The parameters of an image view.
///
/// Same as VkImageViewCreateInfo, but implements Eq and PartialEq.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct ImageViewInfo {
    pub view_type: vk::ImageViewType,
    pub format: vk::Format,
    pub subresource_range: ImageSubresourceRange,
    pub component_mapping: [vk::ComponentSwizzle; 4],
}

#[derive(Copy, Clone, Debug)]
pub struct ResourceState {
    /// Stages that will access the resource.
    pub stages: vk::PipelineStageFlags2,
    /// Access flags for the resource.
    pub access: vk::AccessFlags2,
    /// Requested layout for the resource.
    pub layout: vk::ImageLayout,
}

impl ResourceState {
    pub const TRANSFER_SRC: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_READ,
        layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
    };
    pub const TRANSFER_DST: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::TRANSFER,
        access: vk::AccessFlags2::TRANSFER_WRITE,
        layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
    };
    pub const SHADER_READ: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::ALL_GRAPHICS,
        access: vk::AccessFlags2::SHADER_READ,
        layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
    };
    pub const COLOR_ATTACHMENT_OUTPUT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
        access: vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
    };
    pub const DEPTH_STENCIL_ATTACHMENT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::from_raw(
            vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS.as_raw()
                | vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS.as_raw(),
        ),
        access: vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
        layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
    };
    pub const VERTEX_BUFFER: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::VERTEX_INPUT,
        access: vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
        layout: vk::ImageLayout::UNDEFINED,
    };
    pub const INDEX_BUFFER: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::VERTEX_INPUT,
        access: vk::AccessFlags2::INDEX_READ,
        layout: vk::ImageLayout::UNDEFINED,
    };
    pub const PRESENT: ResourceState = ResourceState {
        stages: vk::PipelineStageFlags2::NONE,
        access: vk::AccessFlags2::NONE,
        layout: vk::ImageLayout::PRESENT_SRC_KHR,
    };
}

#[derive(Copy, Clone, Debug)]
pub enum ImageType {
    Image1D,
    Image2D,
    Image3D,
}

impl ImageType {
    pub const fn to_vk_image_type(self) -> vk::ImageType {
        match self {
            Self::Image1D => vk::ImageType::TYPE_1D,
            Self::Image2D => vk::ImageType::TYPE_2D,
            Self::Image3D => vk::ImageType::TYPE_3D,
        }
    }
}

impl From<ImageType> for vk::ImageType {
    fn from(ty: ImageType) -> Self {
        ty.to_vk_image_type()
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct ImageUsage: u32 {
        const TRANSFER_SRC = 0b1;
        const TRANSFER_DST = 0b10;
        const SAMPLED = 0b100;
        const STORAGE = 0b1000;
        const COLOR_ATTACHMENT = 0b1_0000;
        const DEPTH_STENCIL_ATTACHMENT = 0b10_0000;
        const TRANSIENT_ATTACHMENT = 0b100_0000;
        const INPUT_ATTACHMENT = 0b1000_0000;
    }
}

impl Default for ImageUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl ImageUsage {
    pub const fn to_vk_image_usage_flags(self) -> vk::ImageUsageFlags {
        vk::ImageUsageFlags::from_raw(self.bits())
    }
}

impl From<ImageUsage> for vk::ImageUsageFlags {
    fn from(usage: ImageUsage) -> Self {
        usage.to_vk_image_usage_flags()
    }
}

/// Information passed to `Context::create_image` to describe the image to be created.
#[derive(Copy, Clone, Debug)]
pub struct ImageCreateInfo {
    pub memory_location: MemoryLocation,
    /// Dimensionality of the image.
    pub type_: ImageType,
    /// Image usage flags. Must include all intended uses of the image.
    pub usage: ImageUsage,
    /// Format of the image.
    pub format: Format,
    /// Size of the image.
    pub width: u32,
    pub height: u32,
    pub depth: u32,
    /// Number of mipmap levels. Note that the mipmaps contents must still be generated manually. Default is 1. 0 is *not* a valid value.
    pub mip_levels: u32,
    /// Number of array layers. Default is `1`. `0` is *not* a valid value.
    pub array_layers: u32,
    /// Number of samples. Default is `1`. `0` is *not* a valid value.
    pub samples: u32,
}

impl Default for ImageCreateInfo {
    fn default() -> Self {
        ImageCreateInfo {
            memory_location: MemoryLocation::Unknown,
            type_: ImageType::Image2D,
            usage: Default::default(),
            format: Default::default(),
            width: 1,
            height: 1,
            depth: 1,
            mip_levels: 1,
            array_layers: 1,
            samples: 1,
        }
    }
}

bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct BufferUsage: u32 {
        const TRANSFER_SRC = 0b1;
        const TRANSFER_DST = 0b10;
        const UNIFORM_TEXEL_BUFFER = 0b100;
        const STORAGE_TEXEL_BUFFER = 0b1000;
        const UNIFORM_BUFFER = 0b1_0000;
        const STORAGE_BUFFER = 0b10_0000;
        const INDEX_BUFFER = 0b100_0000;
        const VERTEX_BUFFER = 0b1000_0000;
        const INDIRECT_BUFFER = 0b1_0000_0000;
    }
}

impl Default for BufferUsage {
    fn default() -> Self {
        Self::empty()
    }
}

impl BufferUsage {
    pub const fn to_vk_buffer_usage_flags(self) -> vk::BufferUsageFlags {
        vk::BufferUsageFlags::from_raw(self.bits())
    }
}

impl From<BufferUsage> for vk::BufferUsageFlags {
    fn from(usage: BufferUsage) -> Self {
        usage.to_vk_buffer_usage_flags()
    }
}

/// Information passed to `Context::create_buffer` to describe the buffer to be created.
#[derive(Copy, Clone, Debug)]
pub struct BufferCreateInfo {
    pub memory_location: MemoryLocation,
    /// Usage flags. Must include all intended uses of the buffer.
    pub usage: BufferUsage,
    /// Size of the buffer in bytes.
    pub byte_size: u64,
    /// Whether the memory for the resource should be mapped for host access immediately.
    /// If this flag is set, `create_buffer` will also return a pointer to the mapped buffer.
    /// This flag is ignored for resources that can't be mapped.
    pub map_on_create: bool,
}

impl Default for BufferCreateInfo {
    fn default() -> Self {
        BufferCreateInfo {
            memory_location: MemoryLocation::Unknown,
            usage: Default::default(),
            byte_size: 0,
            map_on_create: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum PipelineBindPoint {
    Graphics,
    Compute,
}

impl PipelineBindPoint {
    pub const fn to_vk_pipeline_bind_point(self) -> vk::PipelineBindPoint {
        match self {
            Self::Graphics => vk::PipelineBindPoint::GRAPHICS,
            Self::Compute => vk::PipelineBindPoint::COMPUTE,
        }
    }
}

impl From<PipelineBindPoint> for vk::PipelineBindPoint {
    fn from(bind_point: PipelineBindPoint) -> Self {
        bind_point.to_vk_pipeline_bind_point()
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ATTACHMENTS

/// Trait implemented for types that contain attachments.
pub trait StaticAttachments {
    /// Color attachment formats.
    const COLOR: &'static [vk::Format];
    /// Depth attachment format.
    const DEPTH: Option<vk::Format>;
    /// Stencil attachment format.
    const STENCIL: Option<vk::Format>;
}

#[derive(Clone, Copy, Debug)]
pub enum ClearColorValue {
    Float([f32; 4]),
    Int([i32; 4]),
    Uint([u32; 4]),
}

impl From<[f32; 4]> for ClearColorValue {
    fn from(v: [f32; 4]) -> Self {
        Self::Float(v)
    }
}

impl From<[i32; 4]> for ClearColorValue {
    fn from(v: [i32; 4]) -> Self {
        Self::Int(v)
    }
}

impl From<[u32; 4]> for ClearColorValue {
    fn from(v: [u32; 4]) -> Self {
        Self::Uint(v)
    }
}

impl From<ClearColorValue> for vk::ClearColorValue {
    fn from(v: ClearColorValue) -> Self {
        match v {
            ClearColorValue::Float(v) => vk::ClearColorValue { float32: v },
            ClearColorValue::Int(v) => vk::ClearColorValue { int32: v },
            ClearColorValue::Uint(v) => vk::ClearColorValue { uint32: v },
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// ARGUMENTS

#[derive(Debug, Clone)]
pub struct ArgumentsLayout<'a> {
    pub bindings: Cow<'a, [vk::DescriptorSetLayoutBinding]>,
}

#[derive(Copy, Clone, Debug)]
pub struct PipelineLayoutDescriptor<'a> {
    pub arguments: &'a [ArgumentsLayout<'a>],
    // None of the relevant drivers on desktop seem to care about precise push constant ranges,
    // so we just store the total size of push constants.
    pub push_constants_size: usize,
}

pub trait StaticPipelineLayout {
    fn push_constants_size(&self) -> usize;
    fn arguments(&self) -> Cow<[ArgumentsLayout]>;
}

/*
pub trait StaticPushConstants {
    #[doc(hidden)]
    fn __check_size()
    where
        Self: Sized,
    {
        assert_eq!(
            mem::size_of::<Self>() % 4,
            0,
            "push constant size must be a multiple of 4"
        );
    }

    // The push constant ranges of this argument.
    //const PUSH_CONSTANT_RANGES: &'static [vk::PushConstantRange];
}
*/

/*
pub trait PushConstants {
    /// Returns the push constant ranges.
    fn push_constant_ranges(&self) -> Cow<'static, [vk::PushConstantRange]>;
}

impl StaticPushConstants for () {
    const PUSH_CONSTANT_RANGES: &'static [vk::PushConstantRange] = &[];
}

impl<T> PushConstants for T
where
    T: StaticPushConstants,
{
    fn push_constant_ranges(&self) -> Cow<'static, [vk::PushConstantRange]> {
        Cow::Borrowed(Self::PUSH_CONSTANT_RANGES)
    }
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////
// SAMPLERS

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

////////////////////////////////////////////////////////////////////////////////////////////////////
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point2D {
    pub x: i32,
    pub y: i32,
}

impl Point2D {
    pub const ZERO: Self = Self { x: 0, y: 0 };
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size2D {
    pub width: u32,
    pub height: u32,
}

impl Size2D {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect2D {
    pub min: Point2D,
    pub max: Point2D,
}

impl Rect2D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }

    pub const fn from_origin_size(origin: Point2D, size: Size2D) -> Self {
        Self {
            min: origin,
            max: Point2D {
                x: origin.x + size.width as i32,
                y: origin.y + size.height as i32,
            },
        }
    }

    pub const fn from_xywh(x: i32, y: i32, width: u32, height: u32) -> Self {
        Self {
            min: Point2D { x, y },
            max: Point2D {
                x: x + width as i32,
                y: y + height as i32,
            },
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Point3D {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Rect3D {
    pub min: Point3D,
    pub max: Point3D,
}

impl Rect3D {
    pub const fn width(&self) -> u32 {
        (self.max.x - self.min.x) as u32
    }
    pub const fn height(&self) -> u32 {
        (self.max.y - self.min.y) as u32
    }
    pub const fn depth(&self) -> u32 {
        (self.max.z - self.min.z) as u32
    }

    pub const fn size(&self) -> Size3D {
        Size3D {
            width: self.width(),
            height: self.height(),
            depth: self.depth(),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Size3D {
    pub width: u32,
    pub height: u32,
    pub depth: u32,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// BLENDING

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendFactor {
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha,
    ConstantColor,
    OneMinusConstantColor,
    ConstantAlpha,
    OneMinusConstantAlpha,
    SrcAlphaSaturate,
}

impl BlendFactor {
    pub const fn to_vk_blend_factor(self) -> vk::BlendFactor {
        match self {
            BlendFactor::Zero => vk::BlendFactor::ZERO,
            BlendFactor::One => vk::BlendFactor::ONE,
            BlendFactor::SrcColor => vk::BlendFactor::SRC_COLOR,
            BlendFactor::OneMinusSrcColor => vk::BlendFactor::ONE_MINUS_SRC_COLOR,
            BlendFactor::DstColor => vk::BlendFactor::DST_COLOR,
            BlendFactor::OneMinusDstColor => vk::BlendFactor::ONE_MINUS_DST_COLOR,
            BlendFactor::SrcAlpha => vk::BlendFactor::SRC_ALPHA,
            BlendFactor::OneMinusSrcAlpha => vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
            BlendFactor::DstAlpha => vk::BlendFactor::DST_ALPHA,
            BlendFactor::OneMinusDstAlpha => vk::BlendFactor::ONE_MINUS_DST_ALPHA,
            BlendFactor::ConstantColor => vk::BlendFactor::CONSTANT_COLOR,
            BlendFactor::OneMinusConstantColor => vk::BlendFactor::ONE_MINUS_CONSTANT_COLOR,
            BlendFactor::ConstantAlpha => vk::BlendFactor::CONSTANT_ALPHA,
            BlendFactor::OneMinusConstantAlpha => vk::BlendFactor::ONE_MINUS_CONSTANT_ALPHA,
            BlendFactor::SrcAlphaSaturate => vk::BlendFactor::SRC_ALPHA_SATURATE,
        }
    }
}

impl From<BlendFactor> for vk::BlendFactor {
    fn from(factor: BlendFactor) -> Self {
        factor.to_vk_blend_factor()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum BlendOp {
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max,
}

impl BlendOp {
    pub const fn to_vk_blend_op(self) -> vk::BlendOp {
        match self {
            Self::Add => vk::BlendOp::ADD,
            Self::Subtract => vk::BlendOp::SUBTRACT,
            Self::ReverseSubtract => vk::BlendOp::REVERSE_SUBTRACT,
            Self::Min => vk::BlendOp::MIN,
            Self::Max => vk::BlendOp::MAX,
        }
    }
}

impl From<BlendOp> for vk::BlendOp {
    fn from(op: BlendOp) -> Self {
        op.to_vk_blend_op()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct ColorBlendEquation {
    pub src_color_blend_factor: BlendFactor,
    pub dst_color_blend_factor: BlendFactor,
    pub color_blend_op: BlendOp,
    pub src_alpha_blend_factor: BlendFactor,
    pub dst_alpha_blend_factor: BlendFactor,
    pub alpha_blend_op: BlendOp,
}

impl Default for ColorBlendEquation {
    fn default() -> Self {
        Self::REPLACE
    }
}

impl ColorBlendEquation {
    pub const REPLACE: Self = Self {
        src_color_blend_factor: BlendFactor::One,
        dst_color_blend_factor: BlendFactor::Zero,
        color_blend_op: BlendOp::Add,
        src_alpha_blend_factor: BlendFactor::One,
        dst_alpha_blend_factor: BlendFactor::Zero,
        alpha_blend_op: BlendOp::Add,
    };

    pub const ALPHA_BLENDING: Self = Self {
        src_color_blend_factor: BlendFactor::SrcAlpha,
        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
        color_blend_op: BlendOp::Add,
        src_alpha_blend_factor: BlendFactor::One,
        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
        alpha_blend_op: BlendOp::Add,
    };

    pub const PREMULTIPLIED_ALPHA_BLENDING: Self = Self {
        src_color_blend_factor: BlendFactor::SrcAlpha,
        dst_color_blend_factor: BlendFactor::OneMinusSrcAlpha,
        color_blend_op: BlendOp::Add,
        src_alpha_blend_factor: BlendFactor::One,
        dst_alpha_blend_factor: BlendFactor::OneMinusSrcAlpha,
        alpha_blend_op: BlendOp::Add,
    };
}

// From WGPU
bitflags::bitflags! {
    #[repr(transparent)]
    #[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
    pub struct ColorWriteMask: u32 {
        const RED = 1 << 0;
        const GREEN = 1 << 1;
        const BLUE = 1 << 2;
        const ALPHA = 1 << 3;
        const COLOR = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits();
        const ALL = Self::RED.bits() | Self::GREEN.bits() | Self::BLUE.bits() | Self::ALPHA.bits();
    }
}

impl ColorWriteMask {
    pub const fn to_vk_color_component_flags(self) -> vk::ColorComponentFlags {
        vk::ColorComponentFlags::from_raw(self.bits())
    }
}

impl From<ColorWriteMask> for vk::ColorComponentFlags {
    fn from(mask: ColorWriteMask) -> Self {
        vk::ColorComponentFlags::from_raw(mask.bits())
    }
}

impl Default for ColorWriteMask {
    fn default() -> Self {
        Self::ALL
    }
}

#[derive(Copy, Clone, Default)]
pub struct ColorBlendState {
    pub color: ColorBlendEquation,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct ColorTargetState {
    /// If `None`, blending is disabled.
    pub blend_equation: Option<ColorBlendEquation>,
    pub color_write_mask: ColorWriteMask,
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// RASTERIZATION

#[derive(Copy, Clone, Debug)]
pub enum PolygonMode {
    Fill,
    Line,
    Point,
}

impl Default for PolygonMode {
    fn default() -> Self {
        Self::Fill
    }
}

impl PolygonMode {
    pub const fn to_vk_polygon_mode(self) -> vk::PolygonMode {
        match self {
            Self::Fill => vk::PolygonMode::FILL,
            Self::Line => vk::PolygonMode::LINE,
            Self::Point => vk::PolygonMode::POINT,
        }
    }
}

impl From<PolygonMode> for vk::PolygonMode {
    fn from(mode: PolygonMode) -> Self {
        mode.to_vk_polygon_mode()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DepthBias {
    pub constant_factor: f32,
    pub clamp: f32,
    pub slope_factor: f32,
}

bitflags! {
    #[repr(transparent)]
    #[derive(Copy, Clone, Debug)]
    pub struct CullMode: u32 {
        const NONE = 0;
        const FRONT = 1 << 0;
        const BACK = 1 << 1;
        const ALL = Self::FRONT.bits() | Self::BACK.bits();
    }
}

impl Default for CullMode {
    fn default() -> Self {
        Self::NONE
    }
}

impl CullMode {
    pub const fn to_vk_cull_mode_flags(self) -> vk::CullModeFlags {
        vk::CullModeFlags::from_raw(self.bits())
    }
}

impl From<CullMode> for vk::CullModeFlags {
    fn from(mode: CullMode) -> Self {
        mode.to_vk_cull_mode_flags()
    }
}

#[derive(Copy, Clone, Debug)]
pub enum FrontFace {
    CounterClockwise = 0,
    Clockwise,
}

impl FrontFace {
    pub const fn to_vk_front_face(self) -> vk::FrontFace {
        match self {
            Self::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            Self::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

impl From<FrontFace> for vk::FrontFace {
    fn from(face: FrontFace) -> Self {
        face.to_vk_front_face()
    }
}

impl Default for FrontFace {
    fn default() -> Self {
        Self::CounterClockwise
    }
}

#[derive(Copy, Clone, Debug)]
pub enum LineRasterizationMode {
    Default = 0,
    Rectangular,
    Bresenham,
    RectangularSmooth,
}

impl LineRasterizationMode {
    pub const fn to_vk_line_rasterization_mode(self) -> vk::LineRasterizationModeEXT {
        match self {
            Self::Default => vk::LineRasterizationModeEXT::DEFAULT,
            Self::Rectangular => vk::LineRasterizationModeEXT::RECTANGULAR,
            Self::Bresenham => vk::LineRasterizationModeEXT::BRESENHAM,
            Self::RectangularSmooth => vk::LineRasterizationModeEXT::RECTANGULAR_SMOOTH,
        }
    }
}

impl From<LineRasterizationMode> for vk::LineRasterizationModeEXT {
    fn from(mode: LineRasterizationMode) -> Self {
        mode.to_vk_line_rasterization_mode()
    }
}

impl Default for LineRasterizationMode {
    fn default() -> Self {
        LineRasterizationMode::Default
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct LineRasterization {
    pub mode: LineRasterizationMode,
}

#[derive(Copy, Clone, Debug, Default)]
pub struct RasterizationState {
    pub polygon_mode: PolygonMode,
    pub cull_mode: CullMode,
    pub front_face: FrontFace,
    pub line_rasterization: LineRasterization,
    pub conservative_rasterization_mode: ConservativeRasterizationMode,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum CompareOp {
    Never = 0,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always,
}

impl Default for CompareOp {
    fn default() -> Self {
        Self::Always
    }
}

impl CompareOp {
    pub const fn needs_ref_value(self) -> bool {
        match self {
            Self::Never | Self::Always => false,
            _ => true,
        }
    }
}

impl CompareOp {
    pub const fn to_vk_compare_op(self) -> vk::CompareOp {
        match self {
            Self::Never => vk::CompareOp::NEVER,
            Self::Less => vk::CompareOp::LESS,
            Self::Equal => vk::CompareOp::EQUAL,
            Self::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            Self::Greater => vk::CompareOp::GREATER,
            Self::NotEqual => vk::CompareOp::NOT_EQUAL,
            Self::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            Self::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl From<CompareOp> for vk::CompareOp {
    fn from(op: CompareOp) -> Self {
        op.to_vk_compare_op()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StencilOp {
    Keep = 0,
    Zero = 1,
    Replace = 2,
    Invert = 3,
    IncrementClamp = 4,
    DecrementClamp = 5,
    IncrementWrap = 6,
    DecrementWrap = 7,
}

impl StencilOp {
    pub const fn to_vk_stencil_op(self) -> vk::StencilOp {
        match self {
            Self::Keep => vk::StencilOp::KEEP,
            Self::Zero => vk::StencilOp::ZERO,
            Self::Replace => vk::StencilOp::REPLACE,
            Self::Invert => vk::StencilOp::INVERT,
            Self::IncrementClamp => vk::StencilOp::INCREMENT_AND_CLAMP,
            Self::DecrementClamp => vk::StencilOp::DECREMENT_AND_CLAMP,
            Self::IncrementWrap => vk::StencilOp::INCREMENT_AND_WRAP,
            Self::DecrementWrap => vk::StencilOp::DECREMENT_AND_WRAP,
        }
    }
}

impl From<StencilOp> for vk::StencilOp {
    fn from(op: StencilOp) -> Self {
        op.to_vk_stencil_op()
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct StencilOpState {
    pub compare: CompareOp,
    pub fail_op: StencilOp,
    pub depth_fail_op: StencilOp,
    pub pass_op: StencilOp,
}

//  Adapted from WGPU
impl StencilOpState {
    pub const IGNORE: Self = StencilOpState {
        compare: CompareOp::Always,
        fail_op: StencilOp::Keep,
        depth_fail_op: StencilOp::Keep,
        pass_op: StencilOp::Keep,
    };

    /// Returns true if the face state uses the reference value for testing or operation.
    pub fn needs_ref_value(&self) -> bool {
        self.compare.needs_ref_value()
            || self.fail_op == StencilOp::Replace
            || self.depth_fail_op == StencilOp::Replace
            || self.pass_op == StencilOp::Replace
    }

    /// Returns true if the face state doesn't mutate the target values.
    pub fn is_read_only(&self) -> bool {
        self.pass_op == StencilOp::Keep && self.depth_fail_op == StencilOp::Keep && self.fail_op == StencilOp::Keep
    }
}

impl StencilOpState {
    pub const fn to_vk_stencil_op_state(&self) -> vk::StencilOpState {
        vk::StencilOpState {
            fail_op: self.fail_op.to_vk_stencil_op(),
            pass_op: self.pass_op.to_vk_stencil_op(),
            depth_fail_op: self.depth_fail_op.to_vk_stencil_op(),
            compare_op: self.compare.to_vk_compare_op(),
            compare_mask: !0,
            write_mask: !0,
            reference: 0,
        }
    }
}

impl From<StencilOpState> for vk::StencilOpState {
    fn from(state: StencilOpState) -> Self {
        state.to_vk_stencil_op_state()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct StencilState {
    pub front: StencilOpState,
    pub back: StencilOpState,
    pub read_mask: u32,
    pub write_mask: u32,
}

//  Adapted from WGPU
impl StencilState {
    /// Returns true if the stencil test is enabled.
    pub fn is_enabled(&self) -> bool {
        (self.front != StencilOpState::IGNORE || self.back != StencilOpState::IGNORE)
            && (self.read_mask != 0 || self.write_mask != 0)
    }
    /// Returns true if the state doesn't mutate the target values.
    pub fn is_read_only(&self, cull_mode: Option<Face>) -> bool {
        // The rules are defined in step 7 of the "Device timeline initialization steps"
        // subsection of the "Render Pipeline Creation" section of WebGPU
        // (link to the section: https://gpuweb.github.io/gpuweb/#render-pipeline-creation)

        if self.write_mask == 0 {
            return true;
        }

        let front_ro = cull_mode == Some(Face::Front) || self.front.is_read_only();
        let back_ro = cull_mode == Some(Face::Back) || self.back.is_read_only();

        front_ro && back_ro
    }
    /// Returns true if the stencil state uses the reference value for testing.
    pub fn needs_ref_value(&self) -> bool {
        self.front.needs_ref_value() || self.back.needs_ref_value()
    }
}

impl Default for StencilState {
    fn default() -> Self {
        Self {
            front: StencilOpState::IGNORE,
            back: StencilOpState::IGNORE,
            read_mask: 0,
            write_mask: 0,
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct DepthStencilState {
    pub depth_write_enable: bool,
    pub depth_compare_op: CompareOp,
    pub stencil_state: StencilState,
}

impl Default for DepthStencilState {
    fn default() -> Self {
        Self {
            depth_write_enable: false,
            depth_compare_op: CompareOp::Less,
            stencil_state: Default::default(),
        }
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VERTEX STATE

#[derive(Copy, Clone, Debug)]
pub enum IndexType {
    U16,
    U32,
}

impl IndexType {
    pub const fn to_vk_index_type(self) -> vk::IndexType {
        match self {
            Self::U16 => vk::IndexType::UINT16,
            Self::U32 => vk::IndexType::UINT32,
        }
    }
}

impl From<IndexType> for vk::IndexType {
    fn from(ty: IndexType) -> Self {
        ty.to_vk_index_type()
    }
}

/// Trait implemented by types that represent vertex data in a vertex buffer.
pub unsafe trait Vertex: Copy + 'static {
    const ATTRIBUTES: &'static [VertexAttributeDescription];
}

/// Trait implemented by types that can serve as indices.
pub unsafe trait VertexIndex: Copy + 'static {
    /// Index type.
    const FORMAT: VertexIndexFormat;
}

/// Describes the type of indices contained in an index buffer.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum VertexIndexFormat {
    /// 16-bit unsigned integer indices
    U16,
    /// 32-bit unsigned integer indices
    U32,
}

/// Description of a vertex attribute within a vertex layout.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct VertexAttributeDescription {
    pub format: vk::Format,
    pub offset: u32,
}

/// Trait implemented by types that can serve as a vertex attribute.
pub unsafe trait VertexAttribute {
    /// Returns the corresponding data format (the layout of the data in memory).
    const FORMAT: vk::Format;
}

/// Wrapper type for normalized integer attributes.
///
/// Helper for `normalized` in `derive(Vertex)`.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash, Default)]
#[repr(transparent)]
pub struct Norm<T>(T);

// Vertex attribute types
macro_rules! impl_vertex_attr {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexAttribute for $t {
            const FORMAT: vk::Format = vk::Format::$fmt;
        }
    };
}

// F32
impl_vertex_attr!(f32, R32_SFLOAT);
impl_vertex_attr!([f32; 2], R32G32_SFLOAT);
impl_vertex_attr!([f32; 3], R32G32B32_SFLOAT);
impl_vertex_attr!([f32; 4], R32G32B32A32_SFLOAT);

// U32
impl_vertex_attr!(u32, R32_UINT);
impl_vertex_attr!([u32; 2], R32G32_UINT);
impl_vertex_attr!([u32; 3], R32G32B32_UINT);
impl_vertex_attr!([u32; 4], R32G32B32A32_UINT);

impl_vertex_attr!(i32, R32_SINT);
impl_vertex_attr!([i32; 2], R32G32_SINT);
impl_vertex_attr!([i32; 3], R32G32B32_SINT);
impl_vertex_attr!([i32; 4], R32G32B32A32_SINT);

// U16
impl_vertex_attr!(u16, R16_UINT);
impl_vertex_attr!([u16; 2], R16G16_UINT);
impl_vertex_attr!([u16; 3], R16G16B16_UINT);
impl_vertex_attr!([u16; 4], R16G16B16A16_UINT);

impl_vertex_attr!(i16, R16_SINT);
impl_vertex_attr!([i16; 2], R16G16_SINT);
impl_vertex_attr!([i16; 3], R16G16B16_SINT);
impl_vertex_attr!([i16; 4], R16G16B16A16_SINT);

// UNORM16
impl_vertex_attr!(Norm<u16>, R16_UNORM);
impl_vertex_attr!(Norm<[u16; 2]>, R16G16_UNORM);
impl_vertex_attr!(Norm<[u16; 3]>, R16G16B16_UNORM);
impl_vertex_attr!(Norm<[u16; 4]>, R16G16B16A16_UNORM);

// SNORM16
impl_vertex_attr!(Norm<i16>, R16_SNORM);
impl_vertex_attr!(Norm<[i16; 2]>, R16G16_SNORM);
impl_vertex_attr!(Norm<[i16; 3]>, R16G16B16_SNORM);
impl_vertex_attr!(Norm<[i16; 4]>, R16G16B16A16_SNORM);

// U8
impl_vertex_attr!(u8, R8_UINT);
impl_vertex_attr!([u8; 2], R8G8_UINT);
impl_vertex_attr!([u8; 3], R8G8B8_UINT);
impl_vertex_attr!([u8; 4], R8G8B8A8_UINT);

impl_vertex_attr!(Norm<u8>, R8_UNORM);
impl_vertex_attr!(Norm<[u8; 2]>, R8G8_UNORM);
impl_vertex_attr!(Norm<[u8; 3]>, R8G8B8_UNORM);
impl_vertex_attr!(Norm<[u8; 4]>, R8G8B8A8_UNORM);

impl_vertex_attr!(i8, R8_SINT);
impl_vertex_attr!([i8; 2], R8G8_SINT);
impl_vertex_attr!([i8; 3], R8G8B8_SINT);
impl_vertex_attr!([i8; 4], R8G8B8A8_SINT);

// Vertex types from glam --------------------------------------------------------------------------

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec2,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 2
    },
    R32G32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec3,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 3
    },
    R32G32B32_SFLOAT
);

#[cfg(feature = "graal-glam")]
impl_vertex_attr!(
    glam::Vec4,
    TypeDesc::Vector {
        elem_ty: PrimitiveType::Float,
        len: 4
    },
    R32G32B32A32_SFLOAT
);

// Index data types --------------------------------------------------------------------------------
macro_rules! impl_index_data {
    ($t:ty, $fmt:ident) => {
        unsafe impl VertexIndex for $t {
            const FORMAT: VertexIndexFormat = VertexIndexFormat::$fmt;
        }
    };
}

impl_index_data!(u16, U16);
impl_index_data!(u32, U32);

#[derive(Copy, Clone, Debug)]
pub enum VertexInputRate {
    Vertex,
    Instance,
}

impl VertexInputRate {
    pub const fn to_vk_vertex_input_rate(self) -> vk::VertexInputRate {
        match self {
            Self::Vertex => vk::VertexInputRate::VERTEX,
            Self::Instance => vk::VertexInputRate::INSTANCE,
        }
    }
}

impl From<VertexInputRate> for vk::VertexInputRate {
    fn from(rate: VertexInputRate) -> Self {
        rate.to_vk_vertex_input_rate()
    }
}

#[derive(Copy, Clone, Debug)]
pub struct VertexBufferLayoutDescription {
    pub binding: u32,
    pub stride: u32,
    pub input_rate: VertexInputRate,
}

#[derive(Copy, Clone, Debug)]
pub struct VertexInputAttributeDescription {
    pub location: u32,
    pub binding: u32,
    pub format: Format,
    pub offset: u32,
}

#[derive(Copy, Clone, Debug)]
pub enum PrimitiveTopology {
    PointList,
    LineList,
    LineStrip,
    TriangleList,
    TriangleStrip,
    TriangleFan,
    LineListWithAdjacency,
    LineStripWithAdjacency,
    TriangleListWithAdjacency,
    TriangleStripWithAdjacency,
    PatchList,
}

impl PrimitiveTopology {
    pub const fn to_vk_primitive_topology(self) -> vk::PrimitiveTopology {
        match self {
            Self::PointList => vk::PrimitiveTopology::POINT_LIST,
            Self::LineList => vk::PrimitiveTopology::LINE_LIST,
            Self::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Self::TriangleStrip => vk::PrimitiveTopology::TRIANGLE_STRIP,
            Self::TriangleList => vk::PrimitiveTopology::TRIANGLE_LIST,
            Self::TriangleFan => vk::PrimitiveTopology::TRIANGLE_FAN,
            Self::LineListWithAdjacency => vk::PrimitiveTopology::LINE_LIST_WITH_ADJACENCY,
            Self::LineStripWithAdjacency => vk::PrimitiveTopology::LINE_STRIP_WITH_ADJACENCY,
            Self::TriangleListWithAdjacency => vk::PrimitiveTopology::TRIANGLE_LIST_WITH_ADJACENCY,
            Self::TriangleStripWithAdjacency => vk::PrimitiveTopology::TRIANGLE_STRIP_WITH_ADJACENCY,
            Self::PatchList => vk::PrimitiveTopology::PATCH_LIST,
        }
    }
}

impl From<PrimitiveTopology> for vk::PrimitiveTopology {
    fn from(topology: PrimitiveTopology) -> Self {
        topology.to_vk_primitive_topology()
    }
}

impl Default for PrimitiveTopology {
    fn default() -> Self {
        Self::TriangleList
    }
}

#[derive(Copy, Clone, Debug)]
pub enum ConservativeRasterizationMode {
    Disabled,
    Overestimate,
    Underestimate,
}

impl ConservativeRasterizationMode {
    pub const fn to_vk_conservative_rasterization_mode(self) -> vk::ConservativeRasterizationModeEXT {
        match self {
            Self::Disabled => vk::ConservativeRasterizationModeEXT::DISABLED,
            Self::Overestimate => vk::ConservativeRasterizationModeEXT::OVERESTIMATE,
            Self::Underestimate => vk::ConservativeRasterizationModeEXT::UNDERESTIMATE,
        }
    }
}

impl From<ConservativeRasterizationMode> for vk::ConservativeRasterizationModeEXT {
    fn from(mode: ConservativeRasterizationMode) -> Self {
        mode.to_vk_conservative_rasterization_mode()
    }
}

impl Default for ConservativeRasterizationMode {
    fn default() -> Self {
        Self::Disabled
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct VertexInputState<'a> {
    pub topology: PrimitiveTopology,
    pub buffers: &'a [VertexBufferLayoutDescription],
    pub attributes: &'a [VertexInputAttributeDescription],
}

pub trait StaticVertexInput {
    /// Vertex buffers
    const BUFFER_LAYOUT: &'static [VertexBufferLayoutDescription];

    /// Vertex attributes.
    const ATTRIBUTES: &'static [VertexInputAttributeDescription];
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// MULTISAMPLE STATE

// From WGPU
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct MultisampleState {
    pub count: u32,
    pub mask: u64,
    pub alpha_to_coverage_enabled: bool,
}

impl Default for MultisampleState {
    fn default() -> Self {
        Self {
            count: 1,
            mask: !0,
            alpha_to_coverage_enabled: false,
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
pub struct FragmentOutputInterfaceDescriptor<'a> {
    pub color_attachment_formats: &'a [vk::Format],
    pub depth_attachment_format: Option<vk::Format>,
    pub stencil_attachment_format: Option<vk::Format>,
    pub multisample: MultisampleState,
    pub color_targets: &'a [ColorTargetState],
    pub blend_constants: [f32; 4],
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// SHADERS

#[derive(Debug, Clone, Copy)]
pub enum ShaderSource<'a> {
    Content(&'a str),
    File(&'a Path),
}

#[derive(Debug, Clone, Copy)]
pub enum ShaderKind {
    Vertex,
    Fragment,
    Geometry,
    Compute,
    TessControl,
    TessEvaluation,
    Mesh,
    Task,
}

impl ShaderKind {
    pub fn to_vk_shader_stage(&self) -> vk::ShaderStageFlags {
        match self {
            ShaderKind::Vertex => vk::ShaderStageFlags::VERTEX,
            ShaderKind::Fragment => vk::ShaderStageFlags::FRAGMENT,
            ShaderKind::Compute => vk::ShaderStageFlags::COMPUTE,
            ShaderKind::Geometry => vk::ShaderStageFlags::GEOMETRY,
            ShaderKind::TessControl => vk::ShaderStageFlags::TESSELLATION_CONTROL,
            ShaderKind::TessEvaluation => vk::ShaderStageFlags::TESSELLATION_EVALUATION,
            ShaderKind::Mesh => vk::ShaderStageFlags::MESH_NV,
            ShaderKind::Task => vk::ShaderStageFlags::TASK_NV,
        }
    }
}
