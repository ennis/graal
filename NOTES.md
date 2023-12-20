# Module organization

Right now the `context` module is a mess that contains all the following in a single file:
* frame pacing 
* autosync
* frame creation 
* submission

Queue submission can probably be moved to another module.
The "Pass" type, given that it's used by both frame creation, autosync and submission, should probably be moved into the root module.
Same for frame?

Right now, autosync locks the device objects so that resource tracking info can be updated.
Would it be better to store temporary tracking info somewhere else and then update the trackers afterward?
=> why? it just makes it less efficient


# Consider removing deferred autosync

* Build command buffers on-the-fly
  * Pass creation and submission done at the same time
  * No "retroactive" synchronization
* Make autosync simpler
  * Don't try to be clever
  * if not efficient enough, provide escape hatches so that the application can handle sync manually
* Keep frame numbers?

# Remove deferred pass submission

Currently, work to be submitted to the GPU is split in "passes". Each pass declares its own set of read and/or write dependencies
(memory dependencies). The sequence of passes thus defines a kind of "frame graph" (with the edges representing the resource dependencies between passes).

Passes are not submitted to the GPU immediately: instead they are collected in a frame object.
This was done, originally, so that we have a "complete view" of the frame in order to do optimizations:
1. optimize the placement of pipeline barriers and semaphore operations 
2. optimize resource layout transitions
3. perform automatic resource memory aliasing of transient resources

However, this comes at a cost in terms of complexity, both in implementation and for the user of the API.

It has become clear that (1) and (2) can be done on-the-fly if we tolerate potentially suboptimal and/or redundant barriers (whether this affects performance is unknown).
(3) has always been difficult. The current iteration of the transient allocation algorithm is largely untested and 
almost certainly has blind spots. It also forces us to delay the creation of VkBuffers/VkImages until the moment the frame is submitted,
which is inconvenient for the user who might want a handle to a fully created resource immediately.

For these reasons, we propose to remove deferred pass submission from the API.

Instead, expose "virtual queues" to the user. The user pushes work to those queues and the library figures out when to 
- insert pipeline and/or memory barriers,
- signal semaphores, 
- or finish the current command buffer and start a new one.
Between work items, the user should push "resource barriers", which indicate in which states the resources should be for the next work item.
This is the equivalent of the image and buffer dependencies (`add_image_dependency`, etc.) of passes.

The library tracks the current state of all resources and figures out which barriers to insert.

# Graal V2 API

Issue: Except for other than images & buffers, we don't track on which queues objects are used. In order to reclaim them, 
we need to signal the timeline on every queue that might use it, even if the queue is otherwise unused.

"Solution": move deferred object destruction to queues

# Module org, again

From the user POV: no module hierarchy.

Internally:
- `device` contains most of the things resource IDs, and RAII wrappers, `Buffer` and `Image`, plus typed wrappers.
- `device` contains `Device`, `create_device_and_queue`


# Can we avoid waiting on multiple frames?

If a resource is used on multiple queues, require the user to synchronize with all queues before deleting the resource


# Middle-layer

- RAII resource wrappers
- blit image
- mipmap generation
- descriptor shit
- typed buffers
- vertex formats
- `uniforms!` macro
- rendering
- clear color / depth

- draw layers: 
  - loop 1: bind pipeline
    - loop 2: bind uniforms
      - loop 3: draw calls 


Use VK_KHR_dynamic_rendering and VK_EXT_descriptor_buffer.
Possibly VK_KHR_push_descriptor so that we don't have to manage the memory for descriptors at all.


Use a ref to `Queue` to submit work.
Issue: RAII deletion of resources needs backref to owner `Queue`, but `Queue` is not shareable.

Solution:
- set a flag in the resource tracker and have queues scan all resources on end_frame to recycle all discarded resources.

Alternative:
- all resources are `Rc<Resource>`, tracking info stored in a `RefCell` inside it. 
  -> even so, still need an rc backref to the queue, pay (negligible) cost of Rc alloc

```rust

// Use when creating pipelines and queue.render
#[derive(FragmentOutput)]
struct DefaultOutputOutput {
  #[color_attachment]
  color: ImageView<>,
}

#[derive(VertexInput)]
struct VertexInput {
  
}

/// The type describes a DescriptorSetLayout,
/// instances of the type can be used to fill a descriptor buffer
#[derive(Descriptors)]
struct ShaderArguments {
  
}

fn test() {
  let render_output = DefaultRenderPassOutput { .. };

  queue.render(render_output, |rctx| {
    // rctx: RenderCtxt<DefaultRenderPassOutput>
    
    // set_pipeline(P) where P: Pipeline<FragmentOutput=DefaultRenderPassOutput> 
    rctx.set_pipeline(pipeline, |shctx| {
        // shctx: ShaderPipelineCtx
        
        // with_vertex_input(P::VertexInput) where 
        shctx.with_vertex_input(vertex_input, |drawctx| {
            // drawctx: DrawCtx
            // issue argument buffer binds & draw calls
            draw_ctx.bind_arguments(0, scene_args);
            draw_ctx.bind_arguments(1, scene_args);
            draw_ctx.push_constants(...);
            
        });
    });
    
  })
}

```


# Issue: tracking non-resource objects across queues
Deleting an object: must assume that all queues can need it.


# Hot-reloading shaders:
- Essential
- Maybe don't bundle the compiler? Have a daemon that watches GLSL changes and recompiles?

# Descriptor set layouts
Where to stash them?

1. hash-cache in device: meh
2. cache by argument type: need a lazy static associated to the type or unique typeid for the argument struct
3. owned by the shader object: descriptor set layout duplicated between all shaders with the same interface
4. have the application store them somewhere => go fuck yourself
5. create a VkDescriptorSetLayout on-the-fly when needed and delete just after => it needs to be alive when using a descriptor set of the specified layout

2.1 lazy_static associated to the type: can't work with generics

Issue: associating a DescriptorSetLayout to a type statically requires the type to be `Any` (not great, but OK) and that we use a type map.

Otherwise, requires the type to have no generics: ouch.
Associating a DescriptorSetLayout instance to a type that can be generic is near-intractable (look up "rust generic statics").

=> store layouts in the Shaders

Issue: pipeline layouts need descriptor set layouts, but pipeline layouts are associated to multiple shaders

# Bindless?

Would eliminate the need for separate descriptor set layouts: just pass everything in push constants.
Each resource has one index (GPU handle) into a global descriptor table.
This table is stored in a descriptor set. There's one per-queue, same across frames. 

Alternatively:
- linear allocation of descriptors in a buffer with VK_EXT_descriptor_buffer
- no changes in the shader


Philosophy: command buffer oriented.
* Set arguments procedurally.
* Stateful: draw commands take the last arguments
* Set arguments once, then draw multiple times
* More closely matches the actual commands emitted in the command buffer

Declarative alternative:
* Specify arguments all at once in a big struct
* Issue: if the arguments haven't changed, it will potentially upload a lot of redundant data
* It's "cleaner" in the sense that it's harder to misuse and more things are checked at compile time
  (e.g. it's impossible to forget to set an argument, this is checked statically).
* It hides the underlying command buffer / binding model
* However, it needs some kind of caching to avoid changing states when not necessary => complex, more impl work, possibly inefficient

=> Don't hide the underlying GPU binding model.
We could abstract it away if we were building a compiler, because then we'd be able to analyze the code
and determine how to bind things optimally. But we're not building a compiler (at least not yet...).

    device.render(attachments, |render_pass| {
        render_pass.bind_pipeline(&pipeline, |pctx| {
            pctx.set_scene_data(...);
            for object in scene.object() {
                pctx.set_vertex_buffers(...);   // typed method
                pctx.set_object_data(...);      // typed method
                pctx.draw(...);                 // ???
            }
    
            pipeline.set_arguments_0(???);
            pipeline.set_arguments_1(???);
        });
    });

"shallow" alternative:
    
    device.render(attachments, |encoder| {
        encoder.bind_pipeline(...);
        // arguments need a pipeline layout
        encoder.set_arguments(0, ...);      // untyped
        encoder.set_arguments(1, ...);      // untyped
        encoder.set_vertex_buffers(...);    // untyped
        encoder.draw(...);
        encoder.bind_pipeline(...);   // bind another pipeline, keep bound arguments
    });


Problem 1: set_arguments need a VkPipelineLayout when using push descriptors => require binding a pipeline before setting the arguments (not unreasonable)
Problem 2: no validation of arguments

----------------------------

untyped alternative:

    device.render(attachments, |render_pass| {
        render_pass.bind_pipeline(&pipeline, |cmd| {
            cmd.set_arguments(0, ...);
            for object in scene.object() {
                cmd.set_vertex_buffers(...);
                cmd.set_arguments(1, ...);
                cmd.draw(...);                 
            }
        });
    });

issue: no way to ensure that set_arguments is called with the correct type. We can't use typeids because the Argument type may not be 'static.



Unsolved questions:
Q: Are set_vertex_buffers and such always generated by the macro?
A: It can be a trait method (trait VertexInputInterface)

Q: Documentation of the generated methods? Since they are generated, there's no documentation in the library docs.
A: The contents of the macro would need to be self-documenting.

Q: Is it possible to have two VkPipelines with the same interface?
A: Yes: the source code is specified dynamically, it is decoupled from the interface type. There isn't necessarily a 1:1 mapping between interface types and VkPipelines.

Q: are pipeline interfaces traits or concrete structs?
With traits: it's possible to implement the pipeline interface for a custom pipeline type, and make
some functions generic over the actual pipeline instance.

Q: what about draw calls? should those be provided by the pipeline interface?
A: draw() methods included in the base PipelineInterface trait. Must be in scope, though.

Issue: most implementations will just call the methods in PipelineInterfaceBase, it's useless busy work.

# Guidelines for module organization

- If something needs access to private parts of an object, put them in the same module instead of using `pub(crate)`
- For `device.create_xxx` functions (creator functions), put the creation logic in `device.rs`, have a `pub(super)` constructor for the object in its own file.

# Merge MLR and graal

I won't use graal alone anyway.


# WGPU?

Get vulkan/metal/dx12 for free.
Made primarily for portability

# ImageViews

We want to keep ImageViews. However, they should keep the parent image alive. This means that either `ImageAny` is an `Rc<ImageInner>`
or that resources have an internal refcount.

Ref count overview:
- add a `ref_count: usize` to `Resource`
- add `Device::resource_add_ref(&self, resource_id)`
- `Device::destroy_resource` only decreases the refcount.

Device API:

    // ID-based
    create_resource() -> ResourceId
    destroy_resource(id)

    // Arc-based
    create_resource() -> Arc<Resource>


# Uniform buffers and upload buffers

So far they haven't been a problem: all uniform data so far fits in push constants, the rest are just descriptors.
It's possible that we might never need them.

# Issue with pipeline barriers:

- blit encoder, layout is TRANSFER_DST
- same command buffer, draw operation, write to storage image in frag shader: layout should be transferred to GENERAL, but it must be done before entering the render pass
  - at the moment we enter the render pass, we don't know yet which images are going to be used, and it might not be the first use of the image in the command buffer 


We **can't** insert barriers in the middle of a rendering operation and we **can't** retroactively insert barriers before the beginning
of the rendering operation.


Solution?
- use GENERAL layout if the image can be used in shaders?
    - not enough: layout transitions are not the only thing that matters, there's also memory visibility
1. require the user to specify used resources up front, at the beginning of the render pass 
2. defer recording of command buffers
  - that's what wgpu does :( 
3. use different command buffers for render / compute / blit, insert fixup command buffers

Solution 1 seems the most in-line with the rest of the API.
But that's a shame, because the mechanism to track memory accesses in argument buffers works rather well.
At least we still track resource uses.

We could validate that they're in the correct layout, at least. Or do nothing at all.
Barriers are flushed on begin_blit, begin_compute, begin_rendering



Real example:

    // Clear the tile curve count image and the tile buffer
    {
        cmdbuf.use(&self.bin_rast_tile_curve_count_image, TransferDst);
        cmdbuf.use(&self.bin_rast_tile_buffer, TransferDst);

        let mut encoder = cmdbuf.begin_blit();
        encoder.clear_image(&self.bin_rast_tile_curve_count_image, ClearColorValue::Uint([0, 0, 0, 0]));
        encoder.fill_buffer(&self.bin_rast_tile_buffer.slice(..).any(), 0);
    }

    // Render the curves
    {
        cmdbuf.use(&self.color_target_view, ColorAttachment);
        cmdbuf.use(&self.depth_buffer_view, DepthAttachment);
        cmdbuf.use(&animation.position_buffer, StorageReadOnly);
        cmdbuf.use(&animation.curve_buffer, StorageReadOnly);
        cmdbuf.use(&self.bin_rast_tile_curve_count_image, StorageReadWrite);
        cmdbuf.use(&self.bin_rast_tile_buffer, StorageReadWrite);

        let mut encoder = cmdbuf.begin_rendering(&RenderAttachments {
            color: &color_target_view,
            depth: &self.depth_buffer_view,
        });

        let bin_rast_tile_curve_count_image_view = self.bin_rast_tile_curve_count_image.create_top_level_view();
        encoder.bind_graphics_pipeline(bin_rast_pipeline);
        encoder.set_viewport(0.0, 0.0, tile_count_x as f32, tile_count_y as f32, 0.0, 1.0);
        encoder.set_scissor(0, 0, tile_count_x, tile_count_y);
        encoder.bind_arguments(
            0,
            &BinRastArguments {
                position_buffer: animation.position_buffer.as_read_only_storage_buffer(),
                curve_buffer: animation.curve_buffer.as_read_only_storage_buffer(),
                tiles_curve_count_image: bin_rast_tile_curve_count_image_view.as_read_write_storage(),
                tiles_buffer: self.bin_rast_tile_buffer.as_read_write_storage_buffer(),
            },
        );

        encoder.bind_push_constants(
            PipelineBindPoint::Graphics,
            &BinRastPushConstants {
                view_proj: self.camera_control.camera().view_projection(),
                base_curve: animation.frames[self.bin_rast_current_frame].curves.start,
                stroke_width: self.bin_rast_stroke_width,
                tile_count_x,
                tile_count_y,
            },
        );

        encoder.draw_mesh_tasks(frame.curves.count, 1, 1);
    }

    {
        cmdbuf.use_image(&self.bin_rast_tile_curve_count_image, ResourceState::TRANSFER_SRC);
        cmdbuf.use_image_view(&self.color_target_view, ResourceState::COLOR_ATTACHMENT);

        let mut encoder = cmdbuf.begin_blit();
        encoder.blit_image(
            &self.bin_rast_tile_curve_count_image,
            ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            Rect3D {
                min: Point3D { x: 0, y: 0, z: 0 },
                max: Point3D {
                    x: tile_count_x as i32,
                    y: tile_count_y as i32,
                    z: 1,
                },
            },
            &color_target,
            ImageSubresourceLayers {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            Rect3D {
                min: Point3D { x: 0, y: 0, z: 0 },
                max: Point3D {
                    x: width as i32,
                    y: height as i32,
                    z: 1,
                },
            },
            vk::Filter::NEAREST,
        );
    }


# Refactor

Two arrays:
- state: holds current resource states + vulkan handles (all data necessary to build barriers)


Modules:
- device.rs: device, queues 
  - init.rs: device initialization
- command.rs: command buffers 
  - render.rs: render commands & RenderEncoder
  - blit.rs: blit commands & BlitEncoder
- track.rs: use trackers
- pipeline.rs: pipeline creation (Device::create_graphics_pipeline & GraphicsPipeline & shader compilation)