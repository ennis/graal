//! Wrapper around device and queues.
use crate::{
    argument::{Arguments, ArgumentsLayout, StaticArguments},
    attachments::{Attachments, StaticAttachments},
    encoder::RenderEncoder,
    error::Error,
    pipeline_layout::PipelineLayout,
    sampler::{Sampler, SamplerCreateInfo},
    shader::{GraphicsPipeline, GraphicsShaders, ShaderCode, ShaderKind, ShaderSource},
    vertex::{
        StaticVertexInput, VertexAttributeDescription, VertexBufferLayoutDescription, VertexInput,
        VertexInputAttributeDescription,
    },
    vk,
};
use graal::{
    ash::prelude::VkResult,
    device::{BufferHandle, ImageHandle},
    queue::{ResourceState, Submission},
};
use std::{
    borrow::Cow,
    cell::RefCell,
    collections::HashMap,
    ffi::{c_char, c_void},
    ptr,
};

pub struct Device {
    device: graal::device::Device,
    queue: graal::queue::Queue,
    sampler_cache: RefCell<HashMap<SamplerCreateInfo, Sampler>>,
    compiler: shaderc::Compiler,
}

pub struct RenderPass<'a> {
    device: &'a mut Device,
    submission: &'a mut Submission,
    cb: vk::CommandBuffer,
}

impl<'a> RenderPass<'a> {
    pub fn device(&self) -> &Device {
        self.device
    }
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        self.cb
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

impl Device {
    pub fn new() -> Result<Device, Error> {
        // SAFETY: no surface is passed, so there are no requirements on the call.
        let (device, queue) = unsafe { graal::device::create_device_and_queue(None)? };

        Ok(Device {
            device,
            queue,
            sampler_cache: RefCell::new(HashMap::new()),
            compiler: shaderc::Compiler::new().expect("failed to create the shader compiler"),
        })
    }

    pub fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.sampler_cache.borrow().get(info) {
            return sampler.clone();
        }
        todo!()
        /*let sampler = Sampler::new(self.device.clone(), info);
        self.sampler_cache.borrow_mut().insert(info.clone(), sampler.clone());
        sampler*/
    }

    /// Returns the underlying graal device.
    pub fn device(&self) -> &graal::device::Device {
        &self.device
    }

    /// Compiles a shader
    fn compile_shader(&self, kind: ShaderKind, source: ShaderSource, entry_point: &str) -> Result<Vec<u32>, Error> {
        let input_file_name = match source {
            ShaderSource::Content(_) => "<embedded shader>",
            ShaderSource::File(path) => path.file_name().unwrap().to_str().unwrap(),
        };

        let source_from_path;
        let source_content = match source {
            ShaderSource::Content(str) => str,
            ShaderSource::File(path) => {
                source_from_path = std::fs::read_to_string(path)?;
                source_from_path.as_str()
            }
        };
        let kind = match kind {
            ShaderKind::Vertex => shaderc::ShaderKind::Vertex,
            ShaderKind::Fragment => shaderc::ShaderKind::Fragment,
            ShaderKind::Compute => shaderc::ShaderKind::Compute,
            ShaderKind::TessControl => shaderc::ShaderKind::TessControl,
            ShaderKind::TessEvaluation => shaderc::ShaderKind::TessEvaluation,
            ShaderKind::Mesh => shaderc::ShaderKind::Mesh,
            ShaderKind::Task => shaderc::ShaderKind::Task,
        };

        let mut compile_options = shaderc::CompileOptions::new().unwrap();
        let mut base_include_path = std::env::current_dir().expect("failed to get current directory");

        match source {
            ShaderSource::File(path) => {
                if let Some(parent) = path.parent() {
                    base_include_path = parent.to_path_buf();
                }
            }
            _ => {}
        }

        compile_options.set_include_callback(move |requested_source, type_, requesting_source, include_depth| {
            let mut path = base_include_path.clone();
            path.push(requested_source);
            let content = match std::fs::read_to_string(&path) {
                Ok(content) => content,
                Err(e) => return Err(e.to_string()),
            };
            Ok(shaderc::ResolvedInclude {
                resolved_name: path.display().to_string(),
                content,
            })
        });

        let compilation_artifact = self.compiler.compile_into_spirv(
            source_content,
            kind,
            input_file_name,
            entry_point,
            Some(&compile_options),
        )?;

        Ok(compilation_artifact.as_binary().into())
    }

    /// Creates a shader module.
    fn create_shader_module(
        &self,
        kind: ShaderKind,
        code: &ShaderCode,
        entry_point: &str,
    ) -> Result<vk::ShaderModule, Error> {
        let code = match self {
            ShaderCode::Source(source) => Cow::Owned(self.compile_shader(kind, *source, entry_point)?),
            ShaderCode::Spirv(spirv) => Cow::Borrowed(*spirv),
        };

        let create_info = vk::ShaderModuleCreateInfo {
            flags: Default::default(),
            code_size: code.len() * 4,
            p_code: code.as_ptr(),
            ..Default::default()
        };
        let module = unsafe { self.device.create_shader_module(&create_info, None)? };
        Ok(module)
    }

    /// Creates a pipeline layout object.
    fn create_pipeline_layout(
        &self,
        arg_layouts: &[ArgumentsLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> (Vec<vk::DescriptorSetLayout>, vk::PipelineLayout) {
        let mut set_layouts = Vec::with_capacity(arg_layouts.len());
        for layout in arg_layouts.iter() {
            let create_info = vk::DescriptorSetLayoutCreateInfo {
                flags: vk::DescriptorSetLayoutCreateFlags::empty(),
                binding_count: layout.bindings.len() as u32,
                p_bindings: layout.bindings.as_ptr(),
                ..Default::default()
            };
            let sl = unsafe {
                self.device
                    .create_descriptor_set_layout(&create_info, None)
                    .expect("failed to create descriptor set layout")
            };
            set_layouts.push(sl);
        }

        let create_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: set_layouts.len() as u32,
            p_set_layouts: set_layouts.as_ptr(),
            push_constant_range_count: push_constant_ranges.len() as u32,
            p_push_constant_ranges: push_constant_ranges.as_ptr(),
            ..Default::default()
        };

        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&create_info, None)
                .expect("failed to create pipeline layout")
        };

        (set_layouts, pipeline_layout)
    }

    /// Creates a graphics pipeline.
    pub fn create_graphics_pipeline(
        &self,
        interface: &PipelineInterfaceDescriptor,
        shaders: &GraphicsShaders,
    ) -> Result<GraphicsPipeline, Error> {
        // ------ create pipeline layout from statically known information ------
        let (descriptor_set_layouts, pipeline_layout) =
            self.create_pipeline_layout(interface.arguments.as_ref(), interface.push_constants.as_ref());

        // ------ Dynamic states ------

        // Make most of the things dynamic
        // TODO: this could be a static property of the pipeline interface
        let dynamic_states = [
            vk::DynamicState::VIEWPORT,
            vk::DynamicState::SCISSOR,
            vk::DynamicState::STENCIL_COMPARE_MASK,
            vk::DynamicState::STENCIL_WRITE_MASK,
            vk::DynamicState::STENCIL_REFERENCE,
            // pInputAssemblyState
            vk::DynamicState::PRIMITIVE_RESTART_ENABLE,
            vk::DynamicState::PRIMITIVE_TOPOLOGY,
            // pTessellationState
            vk::DynamicState::PATCH_CONTROL_POINTS_EXT,
            // pRasterizationState
            vk::DynamicState::DEPTH_CLAMP_ENABLE_EXT,
            vk::DynamicState::RASTERIZER_DISCARD_ENABLE,
            vk::DynamicState::POLYGON_MODE_EXT,
            vk::DynamicState::CULL_MODE,
            vk::DynamicState::FRONT_FACE,
            vk::DynamicState::DEPTH_BIAS_ENABLE,
            vk::DynamicState::DEPTH_BIAS,
            vk::DynamicState::LINE_WIDTH,
            // pMultisampleState
            vk::DynamicState::RASTERIZATION_SAMPLES_EXT,
            vk::DynamicState::SAMPLE_MASK_EXT,
            vk::DynamicState::ALPHA_TO_COVERAGE_ENABLE_EXT,
            vk::DynamicState::ALPHA_TO_ONE_ENABLE_EXT,
            // pDepthStencilState
            vk::DynamicState::DEPTH_TEST_ENABLE,
            vk::DynamicState::DEPTH_WRITE_ENABLE,
            vk::DynamicState::DEPTH_COMPARE_OP,
            vk::DynamicState::DEPTH_BOUNDS_TEST_ENABLE,
            vk::DynamicState::STENCIL_TEST_ENABLE,
            vk::DynamicState::STENCIL_OP,
            vk::DynamicState::DEPTH_BOUNDS,
            // pColorBlendState
            vk::DynamicState::LOGIC_OP_ENABLE_EXT,
            vk::DynamicState::LOGIC_OP_EXT,
            vk::DynamicState::COLOR_BLEND_ENABLE_EXT,
            vk::DynamicState::COLOR_BLEND_EQUATION_EXT,
            vk::DynamicState::COLOR_WRITE_MASK_EXT,
            vk::DynamicState::BLEND_CONSTANTS,
        ];

        let dynamic_state_create_info = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };

        // ------ Vertex state ------
        let vertex_attribute_count = interface.vertex_attributes.len();
        let vertex_buffer_count = interface.vertex_buffers.len();

        let mut vertex_attribute_descriptions = Vec::with_capacity(vertex_attribute_count);
        let mut vertex_binding_descriptions = Vec::with_capacity(vertex_buffer_count);

        for attribute in interface.vertex_attributes.iter() {
            vertex_attribute_descriptions.push(vk::VertexInputAttributeDescription {
                location: attribute.location,
                binding: attribute.binding,
                format: attribute.format,
                offset: attribute.offset,
            });
        }

        for desc in interface.vertex_buffers.iter() {
            vertex_binding_descriptions.push(vk::VertexInputBindingDescription {
                binding: desc.binding,
                stride: desc.stride,
                input_rate: desc.input_rate,
            });
        }

        let vertex_input_state = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: vertex_buffer_count as u32,
            p_vertex_binding_descriptions: vertex_binding_descriptions.as_ptr(),
            vertex_attribute_description_count: vertex_attribute_count as u32,
            p_vertex_attribute_descriptions: vertex_attribute_descriptions.as_ptr(),
            ..Default::default()
        };

        // ------ Shader stages ------
        let mut stages = Vec::new();
        match shaders {
            GraphicsShaders::PrimitiveShading {
                vertex,
                tess_control,
                tess_evaluation,
                geometry,
                fragment,
            } => {
                let vertex = self.create_shader_module(ShaderKind::Vertex, &vertex.code, vertex.entry_point)?;
                let tess_control = tess_control
                    .map(|t| self.create_shader_module(ShaderKind::TessControl, &t.code, t.entry_point))
                    .transpose()?;
                let tess_evaluation = tess_evaluation
                    .map(|t| self.create_shader_module(ShaderKind::TessEvaluation, &t.code, t.entry_point))
                    .transpose()?;
                let geometry = geometry
                    .map(|t| self.create_shader_module(ShaderKind::Geometry, &t.code, t.entry_point))
                    .transpose()?;
                let fragment = self.create_shader_module(ShaderKind::Fragment, &fragment.code, fragment.entry_point)?;

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
                stages.push(vk::PipelineShaderStageCreateInfo {
                    flags: Default::default(),
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: fragment,
                    p_name: b"main\0".as_ptr() as *const c_char, // TODO
                    p_specialization_info: ptr::null(),
                    ..Default::default()
                });
            }
            GraphicsShaders::MeshShading { .. } => {
                todo!("mesh shading")
            }
        };

        let pipeline_create_info = vk::GraphicsPipelineCreateInfo {
            flags: Default::default(),
            stage_count: stages.len() as u32,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input_state,
            p_input_assembly_state: ptr::null(),
            p_tessellation_state: ptr::null(),
            p_viewport_state: ptr::null(),
            p_rasterization_state: ptr::null(),
            p_multisample_state: ptr::null(),
            p_depth_stencil_state: ptr::null(),
            p_color_blend_state: ptr::null(),
            p_dynamic_state: &dynamic_state_create_info,
            layout: pipeline_layout.pipeline_layout(),
            render_pass: Default::default(),
            subpass: 0,
            base_pipeline_handle: Default::default(),
            base_pipeline_index: 0,
            ..Default::default()
        };

        let pipeline = unsafe {
            match self
                .device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
            {
                Ok(pipelines) => pipelines[0],
                Err(e) => {
                    return Err(Error::Vulkan(e.1));
                }
            }
        };

        Ok(GraphicsPipeline::new(
            self.device.clone(),
            pipeline,
            pipeline_layout.pipeline_layout(),
        ))
    }

    /// Start a rendering pass
    ///
    /// # Arguments
    ///
    /// * `attachments` - The attachments to use for the render pass
    /// * `render_area` - The area to render to. If `None`, the entire area of the attached images is rendered to.
    pub fn render<A: Attachments>(
        &mut self,
        attachments: &A,
        render_area: Option<vk::Rect2D>,
        pass_fn: impl FnOnce(&mut RenderPass),
    ) -> VkResult<()> {
        // collect attachments
        let mut color_attachments: Vec<_> = attachments.color_attachments().collect();
        let mut depth_attachment = attachments.depth_attachment();
        let mut stencil_attachment = attachments.stencil_attachment();

        // determine render area
        let render_area = if let Some(render_area) = render_area {
            render_area
        } else {
            // FIXME validate that all attachments have the same size
            // FIXME validate that all images are 2D
            let extent = color_attachments
                .first()
                .or(depth_attachment.as_ref())
                .or(stencil_attachment.as_ref())
                .expect("render_area must be specified if no attachments are specified")
                .image
                .extent();
            vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent: vk::Extent2D {
                    width: extent.width,
                    height: extent.height,
                },
            }
        };

        let mut submission = Submission::new();

        // register resource uses
        // TODO layout should be configurable
        for color in color_attachments.iter() {
            submission.use_image(color.image.handle().id, ResourceState::COLOR_ATTACHMENT_OUTPUT);
        }
        if let Some(ref depth) = depth_attachment {
            submission.use_image(depth.image.handle().id, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }
        if let Some(ref stencil) = stencil_attachment {
            // Could be the same image as depth, but resource tracking should be OK with that
            submission.use_image(stencil.image.handle().id, ResourceState::DEPTH_STENCIL_ATTACHMENT);
        }

        // Setup VkRenderingAttachmentInfos
        let mut color_attachment_infos: Vec<_> = color_attachments
            .iter()
            .map(|a| {
                // NOTE: the image view could be created at the same time as the Attachment struct
                // but actually no, because if both a depth and stencil attachment are specified, they
                // must be the same image view.

                let image_view = a.image.create_view(&a.view_info);
                vk::RenderingAttachmentInfo {
                    image_view,
                    image_layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                    resolve_mode: vk::ResolveModeFlags::NONE,
                    load_op: a.load_op,
                    store_op: a.store_op,
                    clear_value: a.clear_value.unwrap_or_default(),
                    // TODO multisampling resolve
                    ..Default::default()
                }
            })
            .collect();
        let depth_attachment_info = if let Some(ref depth) = depth_attachment {
            let image_view = depth.image.create_view(&depth.view_info);
            Some(vk::RenderingAttachmentInfo {
                image_view,
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: depth.load_op,
                store_op: depth.store_op,
                clear_value: depth.clear_value.unwrap_or_default(),
                // TODO multisampling resolve
                ..Default::default()
            })
        } else {
            None
        };
        let stencil_attachment_info = if let Some(ref stencil) = stencil_attachment {
            let image_view = stencil.image.create_view(&stencil.view_info);
            Some(vk::RenderingAttachmentInfo {
                image_view,
                // TODO different layouts
                image_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                resolve_mode: vk::ResolveModeFlags::NONE,
                load_op: stencil.load_op,
                store_op: stencil.store_op,
                clear_value: stencil.clear_value.unwrap_or_default(),
                // TODO multisampling resolve
                ..Default::default()
            })
        } else {
            None
        };

        let rendering_info = vk::RenderingInfo {
            flags: Default::default(),
            render_area,
            layer_count: 1, // TODO?
            view_mask: 0,
            color_attachment_count: color_attachment_infos.len() as u32,
            p_color_attachments: color_attachment_infos.as_ptr(),
            p_depth_attachment: depth_attachment_info
                .as_ref()
                .map(|a| a as *const _)
                .unwrap_or(ptr::null()),
            p_stencil_attachment: stencil_attachment_info
                .as_ref()
                .map(|a| a as *const _)
                .unwrap_or(ptr::null()),
            ..Default::default()
        };

        let cb = self.queue.allocate_command_buffer();
        unsafe {
            self.device
                .begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo {
                        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                        ..Default::default()
                    },
                )
                .expect("begin_command_buffer failed");
            self.device.cmd_begin_rendering(cb, &rendering_info);
            let mut render_pass = RenderPass {
                device: self,
                cb,
                submission: &mut submission,
            };
            pass_fn(&mut render_pass);
            self.device.cmd_end_rendering(cb);
            self.device.end_command_buffer(cb).expect("end_command_buffer failed");
            self.queue.submit(submission)
        }
    }
}

/// Describes the interface of a pipeline.
#[derive(Clone, Debug)]
pub struct PipelineInterfaceDescriptor<'a> {
    pub arguments: Cow<'a, [ArgumentsLayout<'a>]>,
    pub push_constants: Cow<'a, [vk::PushConstantRange]>,
    pub vertex_buffers: Cow<'a, [VertexBufferLayoutDescription]>,
    pub vertex_attributes: Cow<'a, [VertexInputAttributeDescription]>,
    pub color_attachment_formats: Cow<'a, [vk::Format]>,
    pub depth_attachment_format: Option<vk::Format>,
    pub stencil_attachment_format: Option<vk::Format>,
}

pub trait StaticPipelineInterface {
    const ARGUMENTS: &'static [&'static ArgumentsLayout<'static>];
    const PUSH_CONSTANTS: &'static [vk::PushConstantRange];

    type VertexInput: StaticVertexInput;
    type Attachments: StaticAttachments;
}

/// Specifies the type of a graphics pipeline with the given vertex input, arguments and push constants.
#[macro_export]
macro_rules! graphics_pipeline_interface {
    [
        $(#[$meta:meta])*
        $v:vis struct $name:ident {
            arguments {
                $(
                    $(#[$arg_meta:meta])*
                    $arg_binding:literal => $arg_method:ident($arg_ty:ty)
                ),*
            }

            $(push_constants {
                $(#[$push_cst_meta:meta])*
                $push_constants_method:ident($push_constants_ty:ty)
            })?

            $(output_attachments($attachments_ty:ty))?
        }
    ] => {
        $(#[$meta])*
        $v struct $name<'a> {
            p: crate::encoder::RenderEncoder<'a>,
        }

        impl<'a> $name<'a> {
            $(
                $(#[$arg_meta])*
                pub fn $arg_method(&mut self, arg: &$arg_ty) {
                    unsafe {
                        self.p.bind_arguments($arg_binding, &arg)
                    }
                }
            )*

            $(
                $(#[$push_cst_meta])*
                pub fn $push_constants_method(&mut self, push_constants: &$push_constants_ty) {
                    unsafe {
                        self.p.bind_push_constants(push_constants)
                    }
                }
            )?
        }

        impl<'a> crate::device::StaticPipelineInterface for $name<'a> {
            const ARGUMENTS: &'static [&'static crate::argument::ArgumentsLayout<'static>] = &[
                $(
                    &$arg_ty::LAYOUT
                ),*
            ];

            const PUSH_CONSTANTS: &'static [crate::vk::PushConstantRange] = graphics_pipeline_interface!(@push_constants $($push_constants_ty)?);

            type Attachments = graphics_pipeline_interface!(@attachments $($attachments_ty)?);
        }
    };

    //------------------------------------------------

    // No push constants -> empty constant ranges
    (@push_constants ) => { &[] };
    (@push_constants $t:ty) => { <$t as $crate::argument::StaticPushConstants>::PUSH_CONSTANT_RANGES };

    // Default attachment type if left unspecified
    (@attachments ) => { () };
    (@attachments $t:ty) => { $t };
}

/*
graphics_pipeline_interface! {

    /// Testing the pipeline interface macro.
    pub struct MyPipelineInterface {
        // Arguments (descriptor sets & push constants)
        arguments {
            /// Specifies scene data parameters.
            0 => set_scene_data(SceneData),
            /// Specifies material parameters.
            1 => set_material_data(MaterialData)
        }

        push_constants {
            /// Sets per-object data.
            set_object_data(ObjectData)
        }

        // Output attachments
        output_attachments(Attachments)
    }
}*/

////////////////////////////////////////////////////////////////////////////////////////////////////

impl<'a> RenderPass<'a> {
    pub unsafe fn bind_graphics_pipeline(&mut self, pipeline: &GraphicsPipeline) {
        unsafe {
            self.device
                .device
                .cmd_bind_pipeline(self.cb, vk::PipelineBindPoint::GRAPHICS, pipeline.pipeline());
        }
    }
}
