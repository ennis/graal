# Module organization

Right now the `context` module is a mess that contains all the following in a single file:

* frame pacing
* autosync
* frame creation
* submission

Queue submission can probably be moved to another module.
The "Pass" type, given that it's used by both frame creation, autosync and submission, should probably be moved into the
root module.
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

Currently, work to be submitted to the GPU is split in "passes". Each pass declares its own set of read and/or write
dependencies
(memory dependencies). The sequence of passes thus defines a kind of "frame graph" (with the edges representing the
resource dependencies between passes).

Passes are not submitted to the GPU immediately: instead they are collected in a frame object.
This was done, originally, so that we have a "complete view" of the frame in order to do optimizations:

1. optimize the placement of pipeline barriers and semaphore operations
2. optimize resource layout transitions
3. perform automatic resource memory aliasing of transient resources

However, this comes at a cost in terms of complexity, both in implementation and for the user of the API.

It has become clear that (1) and (2) can be done on-the-fly if we tolerate potentially suboptimal and/or redundant
barriers (whether this affects performance is unknown).
(3) has always been difficult. The current iteration of the transient allocation algorithm is largely untested and
almost certainly has blind spots. It also forces us to delay the creation of VkBuffers/VkImages until the moment the
frame is submitted,
which is inconvenient for the user who might want a handle to a fully created resource immediately.

For these reasons, we propose to remove deferred pass submission from the API.

Instead, expose "virtual queues" to the user. The user pushes work to those queues and the library figures out when to

- insert pipeline and/or memory barriers,
- signal semaphores,
- or finish the current command buffer and start a new one.
  Between work items, the user should push "resource barriers", which indicate in which states the resources should be
  for the next work item.
  This is the equivalent of the image and buffer dependencies (`add_image_dependency`, etc.) of passes.

The library tracks the current state of all resources and figures out which barriers to insert.

# Graal V2 API

Issue: Except for other than images & buffers, we don't track on which queues objects are used. In order to reclaim
them,
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
struct VertexInput {}

/// The type describes a DescriptorSetLayout,
/// instances of the type can be used to fill a descriptor buffer
#[derive(Descriptors)]
struct ShaderArguments {}

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
5. create a VkDescriptorSetLayout on-the-fly when needed and delete just after => it needs to be alive when using a
   descriptor set of the specified layout

2.1 lazy_static associated to the type: can't work with generics

Issue: associating a DescriptorSetLayout to a type statically requires the type to be `Any` (not great, but OK) and that
we use a type map.

Otherwise, requires the type to have no generics: ouch.
Associating a DescriptorSetLayout instance to a type that can be generic is near-intractable (look up "rust generic
statics").

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
* However, it needs some kind of caching to avoid changing states when not necessary => complex, more impl work,
  possibly inefficient

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

Problem 1: set_arguments need a VkPipelineLayout when using push descriptors => require binding a pipeline before
setting the arguments (not unreasonable)
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

issue: no way to ensure that set_arguments is called with the correct type. We can't use typeids because the Argument
type may not be 'static.

Unsolved questions:
Q: Are set_vertex_buffers and such always generated by the macro?
A: It can be a trait method (trait VertexInputInterface)

Q: Documentation of the generated methods? Since they are generated, there's no documentation in the library docs.
A: The contents of the macro would need to be self-documenting.

Q: Is it possible to have two VkPipelines with the same interface?
A: Yes: the source code is specified dynamically, it is decoupled from the interface type. There isn't necessarily a 1:1
mapping between interface types and VkPipelines.

Q: are pipeline interfaces traits or concrete structs?
With traits: it's possible to implement the pipeline interface for a custom pipeline type, and make
some functions generic over the actual pipeline instance.

Q: what about draw calls? should those be provided by the pipeline interface?
A: draw() methods included in the base PipelineInterface trait. Must be in scope, though.

Issue: most implementations will just call the methods in PipelineInterfaceBase, it's useless busy work.

# Guidelines for module organization

- If something needs access to private parts of an object, put them in the same module instead of using `pub(crate)`
- For `device.create_xxx` functions (creator functions), put the creation logic in `device.rs`, have a `pub(super)`
  constructor for the object in its own file.

# Merge MLR and graal

I won't use graal alone anyway.

# WGPU?

Get vulkan/metal/dx12 for free.
Made primarily for portability

# ImageViews

We want to keep ImageViews. However, they should keep the parent image alive. This means that either `ImageAny` is
an `Rc<ImageInner>`
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
- same command buffer, draw operation, write to storage image in frag shader: layout should be transferred to GENERAL,
  but it must be done before entering the render pass
    - at the moment we enter the render pass, we don't know yet which images are going to be used, and it might not be
      the first use of the image in the command buffer

We **can't** insert barriers in the middle of a rendering operation and we **can't** retroactively insert barriers
before the beginning
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

# Next steps

Basic functionality is done! The rest:

- Move vulkan-specific stuff in its own submodule. Note that the only other backend we might need is metal on mac.
- Maybe ray tracing
- Fix TODOs
- Remove inline uniforms
- Resurrect external image extensions
- Remove queues, replace with a special device API that submits on another queue

# Command buffers?

They exist so that GPU commands can be recorded in parallel. But from how many draw calls does this start to be
beneficial? 1000? 10k? 100k?
"When the CPU becomes the bottleneck": do we have numbers for recent desktop CPUs?

Multithreaded recording is also less relevant with GPU-driven pipelines, since they may not issue draw calls per object
on the CPU.

# Arguments next

- Remove special descriptor types (Read{Only,Write}Storage{Image,Buffer}), replace by attributes on argument fields.
    - Reason: less boilerplate this way
- Support descriptor arrays
- Introduce `ArgumentBlock`s that hold descriptors
    - also they hold refs to the resources inside
    - they will need to hold onto ImageView objects
- Keep track of image views, etc. the same way as resources: record the last use
- Resurrect resource groups

# Resources next

Keep the current design. In addition:

- Resources now hold an `Arc<ResourceInner>`, with `ResourceInner` used for tracking: the Arc refcount is used to check
  if the user
  (or any other object) still holds references, and `ResourceInner::timestamp` holds the last submission that uses the
  resource.
  Checking this `Arc` is enough to see if a resource should be deleted or not.

OR:

- Resources hold `Arc<ImageInner>`, `Arc<BufferInner>`. Trackers hold references to those.
- It's easier to hold onto them in argument buffers. Argbuffers need to merge the resources that it references into the
  parent tracker (the encoder tracker).
  The tracker needs: the ID of the resource, a strong reference to it, and the raw handle (to set up pipeline
  barriers). `Arc<{Buffer,Image}Inner>` has all three in the same location.

Note: going through the whole list of resources on every frame is wasteful.
Alternatives:

- when the last user reference to the resource is dropped:
    - remove the resource in the tracker
    - if it is in use (check last submission index): add it to the list of resources awaiting deletion
    - otherwise: destroy the resource immediately
- periodically:
    - go through the list of resources awaiting deletion

// ArgumentBuffers and encoders:
// - Buffer references and uses: (Arc<BufferInner>, ResourceUse)
// - Image references and uses: (Arc<ImageInner>, ResourceUse)
// - stateless references
//
// Command buffers:
// - Sparse two-sided buffer references and uses: (Arc<BufferInner>, start_state, end_state)
// - Sparse two-sided image references and uses: (Arc<ImageInner>, start_state, end_state)
// - stateless references
//
// Device tracker:
// - Dense one-sided buffer references and uses: (Arc<BufferInner>, end_state)
// - Dense one-sided image references and uses: (Arc<ImageInner>, end_state)
//
// Active submissions:
// - Buffer references
// - Image references
// - stateless references
//
// Two-sided buffer tracker => BufferTracker
// Two-sided image tracker => ImageTracker
// stateless references => individual SecondaryMaps or FxHashMaps

# Command pool allocation

Right now we allocate one command pool per submission, and since we submit often, and usually only one command buffer,
this means almost one command pool per command buffer.
Check whether this matters in relevant drivers.

FIXME: this is broken: nothing guarantees that there aren't unsubmitted command buffers that will be submitted in a
later call to `submit_raw`.
We retire the command pool after the submission, but nothing guarantees that the caller passed all created command
buffers back to submit.
It's possible to create cbs, submit some of them in one submission, and submit the rest later. We can't retire the
command pool unless we know that all CBs are submitted.

=> Impossible to reliably track command pools if we're exposing command buffers to the application as long-lived objects

Implementations:

- radv, nvk: command pools are just boilerplate; command buffers are allocated separately. Pool flags ignored, reset
  pool just loops over all command buffers.

Conclusion: don't bother with tracking pools. Just use one pool per-thread and per-queue, and recycle individual command
buffers in them.

## Alternative

Do not expose Queues directly, instead expose "command streams".
Command streams reference a queue and are responsible for allocating command buffers.
Call `CommandStream::flush` to finish recording and submit all pending command buffers to the GPU.
`CommandStreams` are tied to a particular thread, but they can be cloned and sent to other threads for multithreaded
command recording (`command_stream.fork()`).
Cloned command streams are "joined" to a parent command stream when dropped: command buffers (and pools) are merged into
the main command stream.

Q: Removing compute/transfer encoders?

Issue: binding arguments and setting push constants need a pipeline bind point (graphics or compute). Previously this
was
not necessary because we knew whether we were in a compute or graphics encoder. Without encoders, we don't.


------------------------------------------------------

# Idea 2: a more declarative approach

By simply wrapping the underlying graphics API, there's a lot that is still exposed to users, and a lot of boilerplate
necessary (queues, command buffers, encoders, resource usages).
Plus automatic synchronization is tricky because we don't know the whole frame in advance (e.g. we must defer the actual
recording of draw commands because we can't put pipeline barriers in the middle of render passes).

Thus, render graphs. But there are a few things to consider:

## Build every pass VS build once

If we build the graph on every pass, it's probably not more efficient than the "automatic tracking" approach (the amount
of work is equivalent and the API is less flexible).
So build the graph only when something changes.

## Conditional rendering

I.e. running a pass conditionally. This may change barriers, so at first this will need a pipeline recompilation.

## Inputs and outputs

We can't **only** have resources internal to the graph. We need to be able to import / export resources between graphs.
Those resources need to be allocated and their usage cannot be inferred from usage, since the application can
dynamically
change where they are used.
Compiler analogy: calls to external libraries.
This means that tracking of memory resources is still necessary.

## Parameters

The builder functions for render graphs will need to take a special "Variable" type for parameters that may either be
constant
or dynamic.
E.g. `clear_image(image, impl Variable<ClearColor>)`.
Parameters include:

- image clear color
- buffer fill value
- buffer slice start/size
- viewport x,y,width,height
- scissor
- imported image formats (this can affect )

Before execution:

```rust
fn execute() {
    let mut graph_params = GraphParameters::new();
    graph_params.set(CLEAR_COLOR, ...);   // With CLEAR_COLOR being some kind of typed identifier
    graph_params.set(INPUT_IMAGE, ...);   // external image
    device.execute(graph, graph_params);
}
```

## Arguments

I'd be nice if we could keep the same structure, and use callbacks to set the uniform buffers.
Unfortunately, this means that the callback would be in control of defining which buffers/images are used in the pass,
and that will require dynamic tracking.

### Argument arrays

These are built outside of graphs, and represent groups of external resources.

## Graphs

- Submitted to one queue only
- Make them as big as possible
- They can own long-lived resources, like pipelines
    - In fact, pipelines can be tied to graphs

## Conclusion

Challenging:

- passing arguments
    - especially things containing lifetimes
- imported/exported resources

Evolve current API:

- instead of submitting command buffers, submit render graphs. No tracking necessary inside the render graphs since
  their structure is known ahead of time.

# Argument buffers (groups of resources)

Argument buffers will hold arrays of descriptors to resources (images & buffers).
The resulting descriptor set is always bound, and resources are accessed by indexing into the descriptor array in the
shader (bindless rendering).

Note that adding a resource to an argument buffer "locks" it into a particular state. The individual resources can't be
used as a group anymore.

Use cases:

- accessing textures by index in a shader: bindless rendering
- accessing images by index in shaders: both as textures AND as images (requires a layout transition)

# Graal-lite

=> explicit resource transitions.
Instead of tracking resources separately within a render encoder, require the user to specify the resources beforehand.
Also, remove `BlitEncoder` and `ComputeEncoder`.

=> argument buffers
We want to store descriptors to reuse them later.
Right now we only have push descriptors, and we need a bound pipeline in order to push them. We can't create them in
advance,
we also can't _bind them_ before running passes.

Solutions:

Option A: Create a dummy "ArgumentBuffer" type. For now only contains refs to resources, via a list
of `ArgumentDescription`
entries, and the corresponding `VkPipelineLayout`. The layout must be specified on buffer creation, or it can be
inferred
from the descriptor that are passed to the constructor (it will create a VkPipelineLayout internally)
Issue: push descriptors need the whole VkPipelineLayout.

Option B: Use `VK_EXT_descriptor_buffer`.

Non-option: use VkDescriptorSets. This would mean managing VkDescriptorPools, and I don't want to deal with this shit.

Q: Why do we need DescriptorSetLayouts?
Metal doesn't need them: the layout is given from the shader, or standardized.

If we only allocate one descriptor set for all resources, it's easier to create the descriptor set layout and the set
itself in the same constructor (`DescriptorSet::new(bindings: &[BindingDesc])`).
However, we also need the layout when creating the VkPipeline.

# Keeping in-use resources alive?

Strategies:

- Store a strong ref to used resources in ActiveSubmissions
    - suboptimal because need to store a potentially large number of strong refs
- Keep a strong ref to the resources in an array/map in Device, then periodically prune the map for expired resources
  that
  have only one reference remaining
    - scanning the map is costly
- When the last strong ref to the resource is being dropped, move it to the "deferred deletion" list in device
    - issue: difficult to reliably detect deletion of the last strong ref, because it's possible to form weak pointers

# Ensuring the resources are available

- Metal: useResource *inside* the RenderPass, only valid inside the RenderPass
- Vulkan: vkCmdPipelineBarrier *outside* the RenderPass
- D3D12: ???

If we do it at the CommandStream level (before RenderPass), then it's not compatible with metal, if we do it inside a
renderpass, then it's incompatible with vulkan (must retroactively insert vkCmdPipelineBarrier)

# Remove encoders?

Dynamically enter/exit render passes.
If not inside a render pass, then vkBeginRendering with last set attachments?

No: complicates implementation for no good reason.

# Simplest impl for vulkan

- Encoders:
    - RenderEncoder
    - ComputeEncoder (optional)
    - BlitEncoder (optional)
- use_resource() only outside encoders

Problem: metal needs use_resource when using argument buffers, not only for synchronization but also to
make sure that the resources used in an argument buffer is resident in GPU memory; and this must be done in the encoder,
and only valid for the scope of the encoder; AAAAAAAARGH

Two concerns:

- ensuring that the command buffer holds a strong ref to referenced objects (images, buffers, image views, samplers)
- ensuring that the access to resources are properly synchronized according to their intended usage in the pass (images,
  buffers)

Responsibilities:

- argument buffers: holds strong refs, but they don't specify usages
- individual arguments: holds strong refs + usages

No need to specify explicit resource usage for blit operations.

For render passes we'd like to know which resources are going to be used in advance because when `bind_arguments` is
called we have already begun recording the render pass, and it's too late to emit a barrier.

When specifying usages, we might also keep a strong ref here. But we'd need to specify stuff like samplers & image views
which have no need for synchronization.

Tracking usages manually is a big pain, but it's not for end-users.
But the API requires specifying usages even if
we know that the data is visible already.

I.e.

- allocate a vector
- multiple for loops to add all kinds of possibly used resources
- all of this for nothing on metal if argument buffers are not used, for nothing on vulkan if the resources are already
  properly synchronized

# Compromise

Option 1: since argument buffers shouldn't automatically sync on all the resource they contain, add an API to explicitly
declare that they will be used

- Metal: useResource, useHeap
- Vulkan: "antedated" pipeline barrier

Option 2: to avoid antedated pipeline barriers, require syncs before entering a render pass

- Vulkan: CmdPipelineBarrier
- Metal: impossible, useResource is scoped to the current encoder

Option 3: specify all syncs (tracked arguments & argument buffers) at the beginning of passes

- Compatible with vulkan & metal
- However, need to allocate a list of all used resources before encoding it, which is error-prone, and mostly useless on
  Metal when not using argument buffers

Issue with antedated pipeline barriers: before using resources in draw calls,
we sometimes need to add a CmdPipelineBarrier, which must happen before CmdBeginRendering,
but with the current API we don't know which resources are used until set_vertex_buffer, set_arguments, etc.
which happens after the pass is opened.

This leaves us with two options:

- defer the actual recording of the render pass until we know all used resources; for this we need to create our own
  temporary command list,
  which is the kind of pointless busywork we'd like to avoid.
- start a separate command buffer for the render pass; we can record the pass immediately, and append the
  CmdPipelineBarrier to the
  previous command buffer. Still this creates an extra command buffer that is unnecessary in theory.

So, the compromise:

- for render passes, specify used resources up-front

```
RenderPassDescriptor {
  color: ...,
  depth: ...,
  resources: ...,
}


pass_builder.use_resource(...);
pass_builder.bind_arguments(...);
  

```

- for compute passes, don't bother with passes; put the dispatch alongside the rest of the parameters:

```
cmd.dispatch(ComputeDispatchDescriptor {
  resources: ...,
  argument_buffer: ...,
  arguments: ...,
  size: ...,
});
```

Alternative:

For render passes, specify:

- used resources, common argument buffers at pass setup
- individual arguments, push constants in encoder

For compute passes, specify everything in one go.

Other issues:

- recording the uses of host-uploaded buffers (e.g. host-uploaded vertex & index buffers) is useless since host writes
  are guaranteed to be visible.
    - except, of course, on metal when using argument buffers?


- resources that must be made resident
- resources in arguments
- objects in arguments

What really fucks everything is the need to specify resource uses up-front, due to the way CmdPipelineBarrier must work
on vulkan.

Concepts:

- Device
- CommandStream
- ArgumentSet (or ArgumentBuffer)

## Main annoyances

- having to collect resource uses before recording a render pass (on vulkan, to avoid splitting command buffers)
- collecting resource uses for a pass if the backend doesn't need it (e.g. metal)
- collecting resource uses when there is no synchronization necessary (e.g. on vulkan all host writes are automatically
  visible)

Basically, APIs need more data up-front, which makes it less flexible, with more work,
and the additional data is wasted anyway when porting to other APIs.

## Not the right level of abstraction?

Store pre-built passes containing:

- pipeline
- referenced resources
- bindings (non-dynamic)

This way we don't have to collect referenced resources on each frame if the pipeline doesn't need it.

## Idea 2: low-level render graph

- CommandStream: where to submit commands
- RenderPass: retained render pass object; holds all parameters
- ComputePass: retained compute pass
- BlitPass: ditto
- RenderPlan: holds list of passes & resources

What's different:

- it doesn't infer resource usages
- resources are allocated immediately
- it isn't concerned with binding details

Main issues:

- dynamic buffers like CPU generated vertex data, or uniforms that are not push constants
    - main use case: GUI: dynamic buffers & texture bindings that can change
- one-off draws & dispatches
- anything that depends on the current swapchain image

```
RenderPlanBuilder
  .add_image(...);                // add to argument buffer 
  .add_buffer(...);  
  .push_render_pass(...);         // with recording callback  
  .push_mesh_render_pass(...);
  .push_compute_pass(...);   
  .build() -> RenderPlan;    // infers necessary synchronization between passes

RenderPlan
  .import_buffer(...);
```

=> meh, the current command buffer API is here for a reason

## Idea 3: write different implementations of the engine for vulkan & metal

Probably the sanest idea.
Consider graal as an implementation detail of the vulkan backend. So remove stuff that's not required by vulkan:

- BlitEncoders: move to CommandStream
- enums should typedef to vk types

# Enhancements

# Dev notes

- it's **very easy** to forget calling `reference_resource` on transient resources; not obvious at first glance, no
  warnings from validation layers with buffer device address.
    - TODO: methods to create "pre-referenced" transient resources so that we don't have to call reference_resource
