//! Abstractions over a vulkan device & queues.
use std::{
    borrow::Cow,
    cell::Cell,
    collections::{HashMap, VecDeque},
    ffi::{c_char, c_void, CString},
    fmt,
    ops::Deref,
    ptr,
    rc::{Rc, Weak},
    sync::{Arc, Mutex},
};

use ash::vk;
use slotmap::{Key, SecondaryMap, SlotMap};
use tracing::error;

use crate::{
    compile_shader,
    device::resource::ResourceGroup,
    instance::{vk_ext_debug_utils, vk_khr_surface},
    is_write_access, platform_impl, ArgumentBuffer, ArgumentBufferDesc, Arguments, ArgumentsLayout, BufferAccess,
    BufferInner, BufferRange, BufferRangeUntyped, BufferUsage, CommandPool, CommandStream, CompareOp, ComputePipeline,
    ComputePipelineCreateInfo, Error, GraphicsPipeline, GraphicsPipelineCreateInfo, ImageAccess, ImageInner, ImageView,
    ImageViewInfo, ImageViewInner, PreRasterizationShaders, Sampler, SamplerCreateInfo, ShaderCode, ShaderStage,
    Size3D, Swapchain,
};

mod init;
mod resource;

////////////////////////////////////////////////////////////////////////////////////////////////////

/// Wrapper around a vulkan device, associated queues and tracked resources.
#[derive(Clone)]
pub struct Device {
    pub(crate) inner: Rc<DeviceInner>,
}

impl fmt::Debug for Device {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Device").finish_non_exhaustive()
    }
}

/// Weak reference to a device
#[derive(Clone)]
pub struct WeakDevice {
    pub(crate) inner: Weak<DeviceInner>,
}

impl WeakDevice {
    pub(crate) fn upgrade(&self) -> Option<Device> {
        self.inner.upgrade().map(|inner| Device { inner })
    }
}

impl fmt::Debug for WeakDevice {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("WeakDevice").finish_non_exhaustive()
    }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////
pub(crate) struct ActiveSubmission {
    pub index: u64,
    pub queue: u32,
    pub command_pools: Vec<CommandPool>,
}

impl ActiveSubmission {
    pub(crate) fn new(index: u64, queue: u32, command_pools: Vec<CommandPool>) -> ActiveSubmission {
        ActiveSubmission {
            index,
            queue,
            command_pools,
        }
    }
}

pub(crate) struct DeviceTracker {
    pub active_submissions: VecDeque<ActiveSubmission>,
    pub last_submission_index: u64,
    pub buffers: SecondaryMap<BufferId, BufferAccess>,
    pub images: SecondaryMap<ImageId, ImageAccess>,
}

impl DeviceTracker {
    fn new() -> DeviceTracker {
        DeviceTracker {
            active_submissions: VecDeque::new(),
            last_submission_index: 0,
            buffers: SecondaryMap::new(),
            images: SecondaryMap::new(),
            //retired_command_pools: Vec::new(),
            //free_command_pools: Vec::new(),
        }
    }
}

trait DeleteLater {}
impl<T> DeleteLater for T {}

// FIXME: Mutexes are useless here since this is wrapped in Rc and can't be sent across threads.
// Just use RefCells
pub(crate) struct DeviceInner {
    /// Underlying vulkan device
    device: ash::Device,

    /// Platform-specific extension functions
    platform_extensions: platform_impl::PlatformExtensions,
    physical_device: vk::PhysicalDevice,
    queues: Vec<Arc<QueueShared>>,
    allocator: Mutex<gpu_allocator::vulkan::Allocator>,
    vk_khr_swapchain: ash::extensions::khr::Swapchain,
    vk_ext_shader_object: ash::extensions::ext::ShaderObject,
    vk_khr_push_descriptor: ash::extensions::khr::PushDescriptor,
    vk_ext_mesh_shader: ash::extensions::ext::MeshShader,
    vk_ext_extended_dynamic_state3: ash::extensions::ext::ExtendedDynamicState3,
    vk_ext_descriptor_buffer: ash::extensions::ext::DescriptorBuffer,

    // physical device properties
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device_descriptor_buffer_properties: vk::PhysicalDeviceDescriptorBufferPropertiesEXT,
    physical_device_properties: vk::PhysicalDeviceProperties2,

    // We don't need to hold strong refs here, we just need an ID for them.
    pub(crate) image_ids: Mutex<SlotMap<ImageId, ()>>,
    pub(crate) buffer_ids: Mutex<SlotMap<BufferId, ()>>,
    pub(crate) image_view_ids: Mutex<SlotMap<ImageViewId, ()>>,
    groups: Mutex<ResourceGroupMap>,
    // Command pools per queue and thread.
    free_command_pools: Mutex<Vec<CommandPool>>,

    /// Resources that have a zero user reference count and that should be ready for deletion soon,
    /// but we're waiting for the GPU to finish using them.
    dropped_resources: Mutex<Vec<(u64, Box<dyn DeleteLater>)>>,
    pub(crate) tracker: Mutex<DeviceTracker>,

    deletion_lists: Mutex<Vec<DeferredDeletionList>>,
    sampler_cache: Mutex<HashMap<SamplerCreateInfo, Sampler>>,
    compiler: shaderc::Compiler,

    /// VkDescriptorSetLayouts for universal argument buffers.
    pub(crate) argbuf_layouts: ArgumentBufferLayouts,
}

pub const STATIC_UNIFORM_BUFFERS_COUNT: u32 = 16;
pub const STATIC_STORAGE_BUFFERS_COUNT: u32 = 32;
pub const STATIC_SAMPLED_IMAGES_COUNT: u32 = 32;
pub const STATIC_STORAGE_IMAGES_COUNT: u32 = 32;
pub const STATIC_SAMPLERS_COUNT: u32 = 32;
pub const MAX_INDEXED_STORAGE_BUFFERS_COUNT: u32 = 4096;
pub const MAX_INDEXED_SAMPLED_IMAGES_COUNT: u32 = 4096;
pub const MAX_INDEXED_STORAGE_IMAGES_COUNT: u32 = 4096;

pub const UNIFORM_BUFFERS_SET: u32 = 0;
pub const STORAGE_BUFFERS_SET: u32 = 1;
pub const SAMPLED_IMAGES_SET: u32 = 2;
pub const STORAGE_IMAGES_SET: u32 = 3;
pub const SAMPLERS_SET: u32 = 4;

pub(crate) struct ArgumentBufferLayouts {
    pub(crate) ubo: vk::DescriptorSetLayout,
    pub(crate) ssbo: vk::DescriptorSetLayout,
    pub(crate) texture: vk::DescriptorSetLayout,
    pub(crate) image: vk::DescriptorSetLayout,
    pub(crate) sampler: vk::DescriptorSetLayout,
}

unsafe fn create_argbuf_layouts(device: &ash::Device) -> ArgumentBufferLayouts {
    let sets = [
        (vk::DescriptorType::UNIFORM_BUFFER, STATIC_UNIFORM_BUFFERS_COUNT, None),
        (
            vk::DescriptorType::STORAGE_BUFFER,
            STATIC_STORAGE_BUFFERS_COUNT,
            Some(MAX_INDEXED_STORAGE_BUFFERS_COUNT),
        ),
        (
            vk::DescriptorType::SAMPLED_IMAGE,
            STATIC_SAMPLED_IMAGES_COUNT,
            Some(MAX_INDEXED_SAMPLED_IMAGES_COUNT),
        ),
        (
            vk::DescriptorType::STORAGE_IMAGE,
            STATIC_STORAGE_IMAGES_COUNT,
            Some(MAX_INDEXED_STORAGE_IMAGES_COUNT),
        ),
        (vk::DescriptorType::SAMPLER, STATIC_SAMPLERS_COUNT, None),
    ];

    let mut layouts = vec![];

    for (descriptor_type, binding_count, max_indexed_count) in sets {
        let mut bindings = vec![];
        let mut flags = vec![];

        bindings.extend((0..binding_count).map(|i| vk::DescriptorSetLayoutBinding {
            binding: i,
            descriptor_type,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::ALL,
            p_immutable_samplers: ptr::null(),
        }));
        flags.extend((0..binding_count).map(|_| vk::DescriptorBindingFlags::empty()));

        if let Some(max_indexed_count) = max_indexed_count {
            bindings.push(vk::DescriptorSetLayoutBinding {
                binding: bindings.len() as u32,
                descriptor_type,
                descriptor_count: max_indexed_count,
                stage_flags: vk::ShaderStageFlags::ALL,
                p_immutable_samplers: ptr::null(),
            });
            flags.push(vk::DescriptorBindingFlags::VARIABLE_DESCRIPTOR_COUNT);
        }

        let dslbfci = vk::DescriptorSetLayoutBindingFlagsCreateInfo {
            binding_count: flags.len() as u32,
            p_binding_flags: flags.as_ptr(),
            ..Default::default()
        };

        let mut dslci = vk::DescriptorSetLayoutCreateInfo {
            p_next: &dslbfci as *const _ as *const c_void,
            flags: Default::default(),
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };

        let layout = device
            .create_descriptor_set_layout(&dslci, None)
            .expect("failed to create descriptor set layout");
        layouts.push(layout);
    }

    ArgumentBufferLayouts {
        ubo: layouts[0],
        ssbo: layouts[1],
        texture: layouts[2],
        image: layouts[3],
        sampler: layouts[4],
    }
}

/// Errors during device creation.
#[derive(thiserror::Error, Debug)]
pub enum DeviceCreateError {
    #[error(transparent)]
    Vulkan(#[from] vk::Result),
}

#[derive(Copy, Clone, Debug)]
pub struct QueueFamilyConfig {
    pub family_index: u32,
    pub count: u32,
}

impl Deref for Device {
    type Target = ash::Device;

    fn deref(&self) -> &Self::Target {
        &self.inner.device
    }
}

slotmap::new_key_type! {
    /// Identifies a GPU resource.
    pub struct ImageId;

    /// Identifies a GPU resource.
    pub struct BufferId;

    /// Identifies a GPU resource.
    pub struct ImageViewId;

    /// Identifies a resource group.
    pub struct GroupId;
}

// TODO: move this to a separate module?
#[derive(Debug)]
pub struct ResourceRegistrationInfo<'a> {
    pub name: &'a str,
}

#[derive(Debug)]
pub struct ImageRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
    pub handle: vk::Image,
    pub format: vk::Format,
}

#[derive(Debug)]
pub struct BufferRegistrationInfo<'a> {
    pub resource: ResourceRegistrationInfo<'a>,
}

/// Describes how a resource got its memory.
#[derive(Debug)]
pub enum ResourceAllocation {
    /// We allocated a block of memory exclusively for this resource.
    Allocation {
        allocation: gpu_allocator::vulkan::Allocation,
    },
    /// The memory for this resource was imported or exported from/to an external handle.
    DeviceMemory { device_memory: vk::DeviceMemory },

    /// We don't own the memory for this resource.
    External,
}

impl ResourceAllocation {
    fn is_external(&self) -> bool {
        matches!(self, ResourceAllocation::External)
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

#[derive(Debug)]
enum DeferredDeletionObject {
    Sampler(vk::Sampler),
    DescriptorSetLayout(vk::DescriptorSetLayout),
    PipelineLayout(vk::PipelineLayout),
    Pipeline(vk::Pipeline),
    Semaphore(vk::Semaphore),
    Shader(vk::ShaderEXT),
}

impl From<vk::Sampler> for DeferredDeletionObject {
    fn from(sampler: vk::Sampler) -> Self {
        DeferredDeletionObject::Sampler(sampler)
    }
}
impl From<vk::PipelineLayout> for DeferredDeletionObject {
    fn from(layout: vk::PipelineLayout) -> Self {
        DeferredDeletionObject::PipelineLayout(layout)
    }
}
impl From<vk::Pipeline> for DeferredDeletionObject {
    fn from(pipeline: vk::Pipeline) -> Self {
        DeferredDeletionObject::Pipeline(pipeline)
    }
}
impl From<vk::Semaphore> for DeferredDeletionObject {
    fn from(semaphore: vk::Semaphore) -> Self {
        DeferredDeletionObject::Semaphore(semaphore)
    }
}
impl From<vk::DescriptorSetLayout> for DeferredDeletionObject {
    fn from(layout: vk::DescriptorSetLayout) -> Self {
        DeferredDeletionObject::DescriptorSetLayout(layout)
    }
}
impl From<vk::ShaderEXT> for DeferredDeletionObject {
    fn from(shader: vk::ShaderEXT) -> Self {
        DeferredDeletionObject::Shader(shader)
    }
}

/// Chooses a swapchain surface format among a list of supported formats.
///
/// TODO there's only one supported format right now...
fn get_preferred_swapchain_surface_format(surface_formats: &[vk::SurfaceFormatKHR]) -> vk::SurfaceFormatKHR {
    surface_formats
        .iter()
        .find_map(|&fmt| {
            if fmt.format == vk::Format::B8G8R8A8_SRGB && fmt.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR {
                Some(fmt)
            } else {
                None
            }
        })
        .expect("no suitable surface format available")
}

pub unsafe fn create_device_and_command_stream(
    present_surface: Option<vk::SurfaceKHR>,
) -> Result<(Device, CommandStream), DeviceCreateError> {
    let device = Device::new(present_surface)?;
    let command_stream = device.create_command_stream(0);
    Ok((device, command_stream))
}

////////////////////////////////////////////////////////////////////////////////////////////////////
struct DeferredDeletionList {
    // Timestamp that we should wait for on each queue before deleting the objects.
    timestamps: Vec<u64>,
    // Objects to delete.
    objects: Vec<DeferredDeletionObject>,
}

pub(crate) struct QueueShared {
    /// Family index.
    pub family: u32,
    /// Index within queues of the same family (see vkGetDeviceQueue).
    pub index_in_family: u32,
    pub index: u32,
    pub queue: vk::Queue,
    pub timeline: vk::Semaphore,
    pub next_submission_timestamp: Cell<u64>,
    pub free_command_pools: Mutex<Vec<CommandPool>>,
}

type ResourceGroupMap = SlotMap<GroupId, ResourceGroup>;

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    pub fn weak(&self) -> WeakDevice {
        WeakDevice {
            inner: Rc::downgrade(&self.inner),
        }
    }

    pub fn raw(&self) -> &ash::Device {
        &self.inner.device
    }

    /// Function pointers for VK_KHR_swapchain.
    pub fn khr_swapchain(&self) -> &ash::extensions::khr::Swapchain {
        &self.inner.vk_khr_swapchain
    }

    /// Function pointers for VK_KHR_push_descriptor.
    pub fn khr_push_descriptor(&self) -> &ash::extensions::khr::PushDescriptor {
        &self.inner.vk_khr_push_descriptor
    }

    pub fn ext_extended_dynamic_state3(&self) -> &ash::extensions::ext::ExtendedDynamicState3 {
        &self.inner.vk_ext_extended_dynamic_state3
    }

    pub fn ext_mesh_shader(&self) -> &ash::extensions::ext::MeshShader {
        &self.inner.vk_ext_mesh_shader
    }

    pub fn ext_descriptor_buffer(&self) -> &ash::extensions::ext::DescriptorBuffer {
        &self.inner.vk_ext_descriptor_buffer
    }

    /// Helper function to associate a debug name to a vulkan handle.
    pub(crate) fn set_debug_object_name(
        &self,
        object_type: vk::ObjectType,
        object_handle: u64,
        name: &str,
        serial: Option<u64>,
    ) {
        unsafe {
            let name = if let Some(serial) = serial {
                format!("{}@{}", name, serial)
            } else {
                name.to_string()
            };
            let object_name = CString::new(name.as_str()).unwrap();

            vk_ext_debug_utils()
                .set_debug_utils_object_name(
                    self.inner.device.handle(),
                    &vk::DebugUtilsObjectNameInfoEXT {
                        object_type,
                        object_handle,
                        p_object_name: object_name.as_ptr(),
                        ..Default::default()
                    },
                )
                .unwrap();
        }
    }

    /// Creates a swapchain object.
    pub unsafe fn create_swapchain(
        &self,
        surface: vk::SurfaceKHR,
        format: vk::SurfaceFormatKHR,
        width: u32,
        height: u32,
    ) -> Swapchain {
        let mut swapchain = Swapchain {
            handle: Default::default(),
            surface,
            images: vec![],
            format,
            width,
            height,
        };
        self.resize_swapchain(&mut swapchain, width, height);
        swapchain
    }

    /// Returns the list of supported swapchain formats for the given surface.
    pub unsafe fn get_surface_formats(&self, surface: vk::SurfaceKHR) -> Vec<vk::SurfaceFormatKHR> {
        vk_khr_surface()
            .get_physical_device_surface_formats(self.inner.physical_device, surface)
            .unwrap()
    }

    /// Returns one supported surface format. Use if you don't care about the format of your swapchain.
    pub unsafe fn get_preferred_surface_format(&self, surface: vk::SurfaceKHR) -> vk::SurfaceFormatKHR {
        let surface_formats = self.get_surface_formats(surface);
        get_preferred_swapchain_surface_format(&surface_formats)
    }

    /// Resizes a swapchain.
    pub unsafe fn resize_swapchain(&self, swapchain: &mut Swapchain, width: u32, height: u32) {
        let phy = self.inner.physical_device;
        let capabilities = vk_khr_surface()
            .get_physical_device_surface_capabilities(phy, swapchain.surface)
            .unwrap();
        /*let formats = self
        .vk_khr_surface
        .get_physical_device_surface_formats(phy, swapchain.surface)
        .unwrap();*/
        let present_modes = vk_khr_surface()
            .get_physical_device_surface_present_modes(phy, swapchain.surface)
            .unwrap();

        let present_mode = init::get_preferred_present_mode(&present_modes);
        let image_extent = init::get_preferred_swap_extent((width, height), &capabilities);
        let image_count =
            if capabilities.max_image_count > 0 && capabilities.min_image_count + 1 > capabilities.max_image_count {
                capabilities.max_image_count
            } else {
                capabilities.min_image_count + 1
            };

        let create_info = vk::SwapchainCreateInfoKHR {
            flags: Default::default(),
            surface: swapchain.surface,
            min_image_count: image_count,
            image_format: swapchain.format.format,
            image_color_space: swapchain.format.color_space,
            image_extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 0,
            p_queue_family_indices: ptr::null(),
            pre_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: swapchain.handle,
            ..Default::default()
        };

        let new_handle = self
            .inner
            .vk_khr_swapchain
            .create_swapchain(&create_info, None)
            .expect("failed to create swapchain");
        if swapchain.handle != vk::SwapchainKHR::null() {
            // FIXME what if the images are in use?
            self.inner.vk_khr_swapchain.destroy_swapchain(swapchain.handle, None);
        }

        swapchain.handle = new_handle;
        swapchain.width = width;
        swapchain.height = height;
        swapchain.images = self
            .inner
            .vk_khr_swapchain
            .get_swapchain_images(swapchain.handle)
            .unwrap();
    }

    /// Creates a universal argument buffer.
    pub fn create_argument_buffer(
        &self,
        uniform_buffer_bindings: &[(u32, BufferRangeUntyped)],
        storage_buffer_bindings: &[BufferRangeUntyped],
        sampled_image_bindings: &[BufferRangeUntyped],
    ) -> ArgumentBuffer {
        let mut bindings = [vk::Des];

        let dsci = vk::DescriptorSetLayoutCreateInfo {
            flags: Default::default(),
            binding_count: desc.bindings.len() as u32,
            p_bindings: desc.bindings.as_ptr(),
            ..Default::default()
        };

        unsafe { self.inner.device.create_descriptor_set_layout() }
    }

    //pub fn create_

    pub fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.inner.sampler_cache.lock().unwrap().get(info) {
            return sampler.clone();
        }

        let create_info = vk::SamplerCreateInfo {
            flags: Default::default(),
            mag_filter: info.mag_filter,
            min_filter: info.min_filter,
            mipmap_mode: info.mipmap_mode,
            address_mode_u: info.address_mode_u,
            address_mode_v: info.address_mode_v,
            address_mode_w: info.address_mode_w,
            mip_lod_bias: info.mip_lod_bias.0,
            anisotropy_enable: info.anisotropy_enable.into(),
            max_anisotropy: info.max_anisotropy.0,
            compare_enable: info.compare_enable.into(),
            compare_op: info.compare_op.into(),
            min_lod: info.min_lod.0,
            max_lod: info.max_lod.0,
            border_color: info.border_color,
            ..Default::default()
        };

        let sampler = unsafe {
            self.inner
                .device
                .create_sampler(&create_info, None)
                .expect("failed to create sampler")
        };
        let sampler = Sampler::new(self, sampler);
        self.inner
            .sampler_cache
            .lock()
            .unwrap()
            .insert(info.clone(), sampler.clone());
        sampler
    }

    pub(crate) fn get_or_create_command_pool(&self, queue_family: u32) -> CommandPool {
        let free_command_pools = &mut self.inner.free_command_pools.lock().unwrap();
        let index = free_command_pools
            .iter()
            .position(|pool| pool.queue_family == queue_family);
        if let Some(index) = index {
            return free_command_pools.swap_remove(index);
        } else {
            unsafe { CommandPool::new(&self.inner.device, queue_family) }
        }
    }

    /// Creates a shader module.
    fn create_shader_module(
        &self,
        kind: ShaderStage,
        code: &ShaderCode,
        entry_point: &str,
    ) -> Result<vk::ShaderModule, Error> {
        let code = match code {
            ShaderCode::Source(source) => Cow::Owned(compile_shader(
                kind,
                *source,
                entry_point,
                "",
                shaderc::CompileOptions::new().unwrap(),
            )?),
            ShaderCode::Spirv(spirv) => Cow::Borrowed(*spirv),
        };

        let create_info = vk::ShaderModuleCreateInfo {
            flags: Default::default(),
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        let module = unsafe { self.inner.device.create_shader_module(&create_info, None)? };
        Ok(module)
    }

    fn create_descriptor_set_layout(&self, arguments: &ArgumentsLayout) -> vk::DescriptorSetLayout {
        let create_info = vk::DescriptorSetLayoutCreateInfo {
            flags: vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR,
            binding_count: arguments.bindings.len() as u32,
            p_bindings: arguments.bindings.as_ptr(),
            ..Default::default()
        };
        let sl = unsafe {
            self.inner
                .device
                .create_descriptor_set_layout(&create_info, None)
                .expect("failed to create descriptor set layout")
        };
        sl
    }

    /// Creates a pipeline layout object.
    fn create_pipeline_layout(
        &self,
        bind_point: vk::PipelineBindPoint,
        arg_layouts: &[ArgumentsLayout],
        push_constants_size: usize,
    ) -> (Vec<vk::DescriptorSetLayout>, vk::PipelineLayout) {
        let mut set_layouts = Vec::with_capacity(arg_layouts.len());
        for layout in arg_layouts.iter() {
            set_layouts.push(self.create_descriptor_set_layout(layout));
        }

        let pc_range = if push_constants_size != 0 {
            Some(match bind_point {
                vk::PipelineBindPoint::GRAPHICS => vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::ALL_GRAPHICS
                        | vk::ShaderStageFlags::MESH_EXT
                        | vk::ShaderStageFlags::TASK_EXT,
                    offset: 0,
                    size: push_constants_size as u32,
                },
                vk::PipelineBindPoint::COMPUTE => vk::PushConstantRange {
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    offset: 0,
                    size: push_constants_size as u32,
                },
                _ => unimplemented!(),
            })
        } else {
            None
        };
        let pc_range = pc_range.as_slice();

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: pc_range.len() as u32,
            p_push_constant_ranges: pc_range.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.inner
                .device
                .create_pipeline_layout(&create_info, None)
                .expect("failed to create pipeline layout")
        };

        (set_layouts, pipeline_layout)
    }

    pub fn create_compute_pipeline(&self, create_info: ComputePipelineCreateInfo) -> Result<ComputePipeline, Error> {
        // ------ create pipeline layout from statically known information ------
        let (descriptor_set_layouts, pipeline_layout) = self.create_pipeline_layout(
            vk::PipelineBindPoint::COMPUTE,
            create_info.layout.arguments.as_ref(),
            create_info.layout.push_constants_size,
        );

        let compute_shader = self.create_shader_module(
            ShaderStage::Compute,
            &create_info.compute_shader.code,
            create_info.compute_shader.entry_point,
        )?;

        let create_info = vk::ComputePipelineCreateInfo {
            flags: vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT,
            stage: vk::PipelineShaderStageCreateInfo {
                flags: Default::default(),
                stage: vk::ShaderStageFlags::COMPUTE,
                module: compute_shader,
                p_name: b"main\0".as_ptr() as *const c_char, // TODO
                p_specialization_info: ptr::null(),
                ..Default::default()
            },
            layout: pipeline_layout,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .inner
                .device
                .create_compute_pipelines(vk::PipelineCache::null(), &[create_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(ComputePipeline::new(self.clone(), pipeline, pipeline_layout))
    }

    /// Creates a graphics pipeline.
    pub fn create_graphics_pipeline(&self, create_info: GraphicsPipelineCreateInfo) -> Result<GraphicsPipeline, Error> {
        // ------ create pipeline layout from statically known information ------
        let (descriptor_set_layouts, pipeline_layout) = self.create_pipeline_layout(
            vk::PipelineBindPoint::GRAPHICS,
            create_info.layout.arguments.as_ref(),
            create_info.layout.push_constants_size,
        );

        // FIXME: delete descriptor_set_layouts

        // ------ Dynamic states ------

        // TODO: this could be a static property of the pipeline interface
        let mut dynamic_states = vec![vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];

        if matches!(
            create_info.pre_rasterization_shaders,
            PreRasterizationShaders::PrimitiveShading { .. }
        ) {
            dynamic_states.push(vk::DynamicState::PRIMITIVE_TOPOLOGY);
        }

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        // ------ Vertex state ------

        let vertex_input = create_info.vertex_input;
        let vertex_attribute_count = vertex_input.attributes.len();
        let vertex_buffer_count = vertex_input.buffers.len();

        let mut vertex_attribute_descriptions = Vec::with_capacity(vertex_attribute_count);
        let mut vertex_binding_descriptions = Vec::with_capacity(vertex_buffer_count);

        for attribute in vertex_input.attributes.iter() {
            vertex_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: attribute.format,
                offset: attribute.offset,
            });
        }

        for desc in vertex_input.buffers.iter() {
            vertex_binding_descriptions.push(vk::VertexInputBindingDescription {
                binding: desc.binding,
                stride: desc.stride,
                input_rate: desc.input_rate.into(),
            });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_buffer_count as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_count as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo {
            topology: vertex_input.topology.into(),
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };

        // ------ Shader stages ------
        let mut stages = Vec::new();
        match create_info.pre_rasterization_shaders {
            PreRasterizationShaders::PrimitiveShading {
                vertex,
                tess_control,
                tess_evaluation,
                geometry,
            } => {
                let vertex = self.create_shader_module(ShaderStage::Vertex, &vertex.code, vertex.entry_point)?;
                let tess_control = tess_control
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderStage::TessControl, &t.code, t.entry_point))
                    .transpose()?;
                let tess_evaluation = tess_evaluation
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderStage::TessEvaluation, &t.code, t.entry_point))
                    .transpose()?;
                let geometry = geometry
                    .as_ref()
                    .map(|t| self.create_shader_module(ShaderStage::Geometry, &t.code, t.entry_point))
                    .transpose()?;

                stages.push(vk::PipelineShaderStageCreateInfo {
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vertex,
                    p_name: b"main\0".as_ptr() as *const c_char, // TODO
                    p_specialization_info: ptr::null(),
                    ..Default::default()
                });
                if let Some(tess_control) = tess_control {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TESSELLATION_CONTROL,
                        module: tess_control,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
                if let Some(tess_evaluation) = tess_evaluation {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TESSELLATION_EVALUATION,
                        module: tess_evaluation,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
                if let Some(geometry) = geometry {
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::GEOMETRY,
                        module: geometry,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }
            }
            PreRasterizationShaders::MeshShading { mesh, task } => {
                if let Some(task) = task {
                    let task = self.create_shader_module(ShaderStage::Task, &task.code, task.entry_point)?;
                    stages.push(vk::PipelineShaderStageCreateInfo {
                        flags: Default::default(),
                        stage: vk::ShaderStageFlags::TASK_EXT,
                        module: task,
                        p_name: b"main\0".as_ptr() as *const c_char, // TODO
                        p_specialization_info: ptr::null(),
                        ..Default::default()
                    });
                }

                let mesh = self.create_shader_module(ShaderStage::Mesh, &mesh.code, mesh.entry_point)?;
                stages.push(vk::PipelineShaderStageCreateInfo {
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::MESH_EXT,
                    module: mesh,
                    p_name: b"main\0".as_ptr() as *const c_char, // TODO
                    p_specialization_info: ptr::null(),
                    ..Default::default()
                });
            }
        };

        let fragment = self.create_shader_module(
            ShaderStage::Fragment,
            &create_info.fragment_shader.code,
            create_info.fragment_shader.entry_point,
        )?;
        stages.push(vk::PipelineShaderStageCreateInfo {
            flags: Default::default(),
            stage: vk::ShaderStageFlags::FRAGMENT,
            module: fragment,
            p_name: b"main\0".as_ptr() as *const c_char, // TODO
            p_specialization_info: ptr::null(),
            ..Default::default()
        });

        let attachment_states: Vec<_> = create_info
            .fragment_output
            .color_targets
            .iter()
            .map(|target| match target.blend_equation {
                None => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::FALSE,
                    color_write_mask: target.color_write_mask.into(),
                    ..Default::default()
                },
                Some(blend_equation) => vk::PipelineColorBlendAttachmentState {
                    blend_enable: vk::TRUE,
                    src_color_blend_factor: blend_equation.src_color_blend_factor.into(),
                    dst_color_blend_factor: blend_equation.dst_color_blend_factor.into(),
                    color_blend_op: blend_equation.color_blend_op.into(),
                    src_alpha_blend_factor: blend_equation.src_alpha_blend_factor.into(),
                    dst_alpha_blend_factor: blend_equation.dst_alpha_blend_factor.into(),
                    alpha_blend_op: blend_equation.alpha_blend_op.into(),
                    color_write_mask: target.color_write_mask.into(),
                },
            })
            .collect();

        // Rasterization state
        let line_rasterization_state = vk::PipelineRasterizationLineStateCreateInfoEXT {
            line_rasterization_mode: create_info.rasterization.line_rasterization.mode.into(),
            stippled_line_enable: vk::FALSE,
            line_stipple_factor: 0,
            line_stipple_pattern: 0,
            ..Default::default()
        };

        let conservative_rasterization_state = vk::PipelineRasterizationConservativeStateCreateInfoEXT {
            p_next: &line_rasterization_state as *const _ as *const _,
            conservative_rasterization_mode: create_info.rasterization.conservative_rasterization_mode.into(),
            //extra_primitive_overestimation_size: 0.1,
            ..Default::default()
        };

        let rasterization_state = vk::PipelineRasterizationStateCreateInfo {
            p_next: &conservative_rasterization_state as *const _ as *const _,
            depth_clamp_enable: 0,
            rasterizer_discard_enable: 0,
            polygon_mode: create_info.rasterization.polygon_mode.into(),
            cull_mode: create_info.rasterization.cull_mode.into(),
            front_face: create_info.rasterization.front_face.into(),
            depth_bias_enable: vk::FALSE,
            depth_bias_constant_factor: 0.0,
            depth_bias_clamp: 0.0,
            depth_bias_slope_factor: 0.0,
            line_width: 1.0,
            ..Default::default()
        };

        let multisample_state = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: vk::SampleCountFlags::TYPE_1,
            sample_shading_enable: vk::FALSE,
            min_sample_shading: 0.0,
            p_sample_mask: ptr::null(),
            alpha_to_coverage_enable: create_info.fragment_output.multisample.alpha_to_coverage_enabled.into(),
            alpha_to_one_enable: vk::FALSE,
            ..Default::default()
        };

        let color_blend_state = vk::PipelineColorBlendStateCreateInfo {
            flags: Default::default(),
            logic_op_enable: vk::FALSE,
            logic_op: Default::default(),
            attachment_count: attachment_states.len() as u32,
            p_attachments: attachment_states.as_ptr(),
            blend_constants: create_info.fragment_output.blend_constants,
            ..Default::default()
        };

        let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo {
            flags: Default::default(),
            depth_test_enable: (create_info.depth_stencil.depth_compare_op != CompareOp::Always).into(),
            depth_write_enable: create_info.depth_stencil.depth_write_enable.into(),
            depth_compare_op: create_info.depth_stencil.depth_compare_op.into(),
            stencil_test_enable: create_info.depth_stencil.stencil_state.is_enabled().into(),
            front: create_info.depth_stencil.stencil_state.front.into(),
            back: create_info.depth_stencil.stencil_state.back.into(),
            depth_bounds_test_enable: vk::FALSE,
            min_depth_bounds: 0.0,
            max_depth_bounds: 0.0,
            ..Default::default()
        };

        let rendering_info = vk::PipelineRenderingCreateInfo {
            view_mask: 0,
            color_attachment_count: create_info.fragment_output.color_attachment_formats.len() as u32,
            p_color_attachment_formats: create_info.fragment_output.color_attachment_formats.as_ptr(),
            depth_attachment_format: create_info.fragment_output.depth_attachment_format.unwrap_or_default(),
            stencil_attachment_format: create_info
                .fragment_output
                .stencil_attachment_format
                .unwrap_or_default(),
            ..Default::default()
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            p_next: &rendering_info as *const _ as *const _,
            flags: vk::PipelineCreateFlags::DESCRIPTOR_BUFFER_EXT,
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: &input_assembly_state,
            p_tessellation_state: &Default::default(),
            p_viewport_state: &vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            },
            p_rasterization_state: &rasterization_state,
            p_multisample_state: &multisample_state,
            p_depth_stencil_state: &depth_stencil_state,
            p_color_blend_state: &color_blend_state,
            p_dynamic_state: &dynamic_state_create_info,
            layout: pipeline_layout,
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .inner
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(GraphicsPipeline::new(self.clone(), pipeline, pipeline_layout))
    }
}
