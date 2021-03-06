

# Accessors
They describe how a resource (image or buffer) is accessed within a task. 
They contain the intended usage of the resource and its _access mode_ (read-only/read-write/write-only)

For images, the possible usages are:
- sampled image
- storage image
- transfer source/destination
- color attachment
- depth attachment
- input attachment

For buffers, the possible usages are:
- vertex buffer
- uniform buffer
- storage buffer
- transform feedback buffer
- index buffer
- transfer source
- transfer destination
- uniform texel buffer 
- storage texel buffer
- indirect buffer (commands)

## Accessor syntax(es)

Questions:
- what to optimize?
	- verbosity
	- complexity of implementation
- type-erased accessors?
- how are accessors used afterwards?


### Option A: single template
The same `accessor` template is used for both images and buffers. 
A large set of constructor overloads and deduction guides are provided for ease of use and type safety.

```c++
void task(handler& h) {
accessor color_target { image, color_attachment, ..., h };
accessor rw_storage { image, storage_image, read_write, h };
accessor ubo { buffer, uniform_buffer, h  };
accessor ssbo { buffer, storage_buffer, read_write, h };
accessor vbo { buffer, vertex_buffer, h };
}
```

`accessor` primary template:
```
template <
	typename DataT, 
	image_type Type,
	usage Usage,
	access_mode AccessMode>
class accessor; 
```

A problem with this is that some of the template params only make sense for one particular type of resource.
For instance, `DataT` only makes sense for buffers, while `Type` only makes sense for images. 
In addition, `Usage` has values that are only valid for images, and others for buffers.
For some usages, not all values of `AccessMode` are valid (for instance, when `Usage == sampled_image` then `AccessMode` must be `read`).


### Option B: many types
One accessor template per usage:

```c++
void task(handler& h) {
color_attachment_accessor color_target { image, ..., h };
storage_image_accessor storage {image, read_write, h};
uniform_buffer_accessor ubo { buffer, h  };
storage_buffer_accessor ssbo { buffer, read_write, h };
vertex_buffer_accessor vbo { buffer, h };
}
```

Solves the issue of unused template parameters.

### Option C: buffer and image accessors

```
image_accessor color_target { image, color_attachment, ..., h };
image_accessor rw_storage { image, storage_image, read_write, h };
buffer_accessor ubo { buffer, uniform_buffer, h  };
buffer_accessor ssbo { buffer, storage_buffer, read_write, h };
buffer_accessor vbo { buffer, vertex_buffer, h };
```

More verbose than A, but solves the issue of unused template parameters.

### Option D: methods on resources:
```
auto color_target = image->access_as_color_attachment(image, ..., h );
auto rw_storage = image->access_as_storage_image(image, read_write, h);
auto ubo = buffer->access_as_uniform_buffer(buffer, h);
auto ssbo = buffer->access_as_storage_buffer(buffer, read_write, h);
auto vbo = buffer->access_as_vertex_buffer_buffer(buffer, h);
```

## Conclusion
Option A vs option B: the only difference is that option A has a single "accessor" type in the declaration, 
making it like a "keyword". 
However, it can be more difficult to read, and more difficult to write down the full type. Using it in a function signature is harder (but it's possible to add alias templates).

Q: do we want functions that are fully generic over the type of accessors?
I.e. a (template) function that takes an accessor and then, regarless of resource type and usage, does something with it.
The "something" cannot be:
	- storing it in a list (unless it uses type erasure somehow)

Option A is one primary template with many specializations.
Option B is many different templates.

Q: what do we want to do with accessors?
A: only use them in draw/compute/transfer commands in during submit-time;

Q: can we access the same resource in two different ways within one task? 
A: possibly; what about accessing an image as a `sampled_image` and as a `storage_image` at the same time => YES

accessors without external memory dependencies: "scratch buffers", only used within the task.


## Using accessors

For binding vertex buffers:
```c++

template <typename VertexType>
void bind_vertex_buffer(
	size_t slot, 
	const accessor<VertexType, image_type::image_1d, usage::vertex_buffer, access_mode::read_only>& access) 
{
	...
}
```

## Comparison with SYCL
In SYCL, accessors serve a double role:
- signalling dependencies
- accessing resource data in the device function (provides operator[])

With Vulkan, the resources are not accessed directly: they are bound to the pipeline instead, and referred to in command buffers.
But there's no need for data access operators in c++ (data in accessed in shaders).

## Are accessors necessary?

Definitions:
- DAG build-time: when calling schedule, and executing the closures with handler => building the DAG
- Submit-time: during calling `queue::submit_pending` => creating concrete resources; creating command buffers and filling them.

What about signalling usages when using the resource in a draw command?
```c++
void task(handler& h) 
{
	// problem: image is not a concrete resource during DAG-build, so 
	// we need to create a callback function 
	// (which is what we would do anyway)
	h.clear_image(image, color);

	// problem: image is accessed another time, need to check that the access is compatible
	h.clear_image(image, color);

	// assume this:

	h.draw(image, ...);		// repeated 1000s of times

	// all draws are emitted during DAG-build; can't parallelize
	// => by putting all commands into the DAG-build, it removes submit-time callbacks (for better or worse).
	// => it should be possible to do custom work during submit-time callbacks (e.g. parallel command buffer generation)
	// => split DAG-time (accessors => dependency declaration) and submit-time (command buffer generation)
}
```


# Pipeline creation

In contrast with the "queue" abstraction, I don't think that there's anything to gain by abstracting stuff here.
There are a *lot* of things to abstract here:
- render states [RS]
- render passes [RP]
	- attachments
	- attachment dependencies
- descriptor set layouts [DESC]
- shader modules [SH]
- vertex input layout [VTX]


Among those, I suppose render passes could be "deduced from usage"; however, pipeline creation would need to be deferred to 
submit-time.
Q: deduce render passes from usage?
A: tricky to implement, many questions to solve (should subpasses correspond to different tasks or not?), pipeline creation deferred to submit-time, need caching (and hashing)

Some of those could be deduced from the shader interface; for instance:
- descriptor sets layouts could be inferred from the shader interface
- vertex input layout: not really, but it can be inferred from the passed data and what the shader expects
- render passes: just use a single pass
	- render pass attachments: infer from shader and queue usages

=> let's be clear: the API will be for **experimentation**; not necessarily fit for high-performance rendering (as in games)

## Descriptor set layouts
In order to change the interface of a shader:
1. edit the uniforms as defined in the shader (GLSL)
2. edit the descriptor set layout (C++)
Ideally, we should factor out common parts of interfaces into a common descriptor set layout.

Note that the two steps convey almost the same information.

### Proposal:
- Generate descriptor set layouts automatically by parsing GLSL source code
- Detect identical descriptor set layouts across shaders and re-use them
	- in addition, use "interface include files" that define a descriptor set layout:
```c
// defines 
#pragma graal_shader_interface (set=0)

layout(set=0,binding=0) uniform FrameUniforms {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;
```
	- these files can be parsed independently of other shaders, and can be used to create a descriptor set layout
	- eventually, they could also be parsed at build-time to generate C++ headers that facilitate the creation of matching descriptor sets
	- shaders include those files to define a part of their interface

In the application, when loading a pipeline, the user can specify zero or more automatically-generated descriptor set layouts to conform to.
This can even be encoded in the pipeline type.
	
### Rebuttal
The GLSL uniform declaration does not contain enough information to specify a descriptor set layout. For instance:
- there's no way to specify that a uniform buffer binding is dynamic **
- there's no way to specify immutable samplers
- no way to specify which shader stages are going to use the uniforms
We need custom "annotations" for that, which we need to parse manually.

Additionally, in order to get proper reflection from an interface file, it needs to be compiled to SPIR-V first, which means that it needs to be a valid shader.
A solution is to append `void main() {}` to the source in order to make it a valid shader, but that's a bit hacky.

### Proposal 2: interface description files
Interface description files contain JSON data that contains information about a shader interface, from the point-of-view of both the shader
and the application. Those files will be written by hand, and then used to create both a GLSL include and a C++ include.

```json
{
	"structs": [
		{
			"name": "FrameUniforms",
			"members": [ 
				{"name": "model", "type": "mat4"},
				{"name": "view", "type": "mat4"},
				{"name": "projection", "type": "mat4"}
			],
		}
	],
	"bindings": {
		"0": {
			"descriptor_type": "uniform_buffer_dynamic",
			"type": "FrameUniforms",
			"count": 1,
			"stage_flags": "all_graphics"
		}
	}
}
```


### Proposal 3: set conventions
In order to determine whether a uniform buffer should be dynamic, look at the set number. By convention:
- set #0 is for data that changes per-frame => uniform buffer should be state
- set #1 is for data that changes per-object => uniform buffer should be dynamic
- push constants will be for data that changes per-draw 

Q: should render pass information be put in the shader?
A: No. It does not belong in a shader interface.


## Case study: an extensible compositing application 
Extensible with (fragment) shaders.
Nodes define a number of input images, output images, parameters, and allowed formats.

The node provides:
- a fragment shader SPIR-V binary
- something that identifies the well-known uniform interfaces (descriptor sets) that the shader supports
- for each input, a list of supported input formats
- for each output, given input formats, a list of supported output formats


Ideal situation:
- in the application, "invoke" a shader by filling a struct corresponding to the descriptor set layout; 
	it automatically creates the descriptor set;
	- the struct type (==interface type) is a parameter of the pipeline so it can be statically verified
	- there can be more than one interface type
- the "static interface" is in the template parameters, the "dynamic interface" can be queried 
	- it returns information about descriptor set layouts, and layout of buffers (types, members)

- "user" in control:
	- shader source
	- number of inputs
	- number of outputs
- "application" in control:
	- 

- hot-reloading: just re-build the pipeline
	- the static interface cannot change

- need to create pipeline variants for 
	- different render target formats
	- different vertex buffer inputs

Idea: a "pipeline" class that is like a query for a concrete pipeline
- template parameters: descriptor set interface types, vertex bindings
- for maximum flexibility: temporarily erase some template params
	- e.g. may want a function that operates only on "pipelines with specific vertex bindings"
		or "pipelines with that specific set interface"
	- 
- pipeline_base base class without any template parameters
	- query

- e.g. full-screen quad shader
```cpp
pipeline_base quad_pipeline();

void test() 
{
	auto pp = quad_pipeline();
	pp.set_fragment_shader(spv);
	pp.set_rasterization_state(...);
	auto tpp = pp.enforce<global_params>();	// copy, not reference
	// tpp : pipeline<unknown, global_params>

	auto variant_1 = pp; // copy semantics, not a reference
	pp.set_rasterization_state(...);
	auto variant_2 = pp;

	// 
	variant_1.resolve();	// creates a concrete VkPipeline
	variant_2.resolve();	// creates a concrete VkPipeline

	// 
	auto vpp = pp_base.with_vertex_input<ty>();	// derived pipeline with specific vertex bindings
	auto vpp2 = pp_base.with_vertex_input<ty2>();	// same but with different vertex bindings
	// 

	// "pipeline_base" is a query object, like QFont.
	// pipeline_base caches its last resolved vkpipeline, which is invalidated when calling set_xxx
	// 
}
```

### Shader interface types
A type, associated to a descriptor set layout, used to build a descriptor set.
Contains references to buffers.
The type has a static "set" index associated to it.
Defines dynamic offsets (indices).
Each instance can be turned into a descriptor set: `descriptor_set<T,dynamic_indices>`
The size_t parameters are for the dynamic indices.
Bind to the pipeline with cmd.bind_descriptor_set(set, index_1, index_2, ...)

```cpp

struct frame_parameters {
	buffer_view<T> ubo;
	image_view<...> 
};

```


# Exception safety

Consider this:
```cpp
int main() {
	queue q{...};
	image img{...};

	try {
		q.submit([](handler& h) {
			accessor img{..., h};
			do_something_that_throws();		// may throw
		});

	} catch (std::exception e) {
		// the accessor will have modified the current batch and the last_write_sequence_number of the image
		// leaving the queue in an invalid state
		// => the handler should only modify the current task
		// => force the handler callback to be noexcept?
	}
}
```

This would be safe:
```cpp
int main() {
	queue q{...};
	image img{...};

	q.submit([](handler& h) {
		try {
			accessor img{..., h};
			do_something_that_throws();		// may throw
		} catch (...) {
			// recover somehow, or terminate
		}
	});
}
```

I think it's reasonable to force the handler callback to be noexcept. 
Being able to throw exception across the handler boundary would be useful only for this:
```cpp
int main() {
	queue q{...};
	image img{...};

	try {
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		//...
		q.submit([](handler& h) { ... });
	} catch(std::exception e) {
		// ...
	}
}
```
but even that would leave the queue in an indeterminate state (which tasks were submitted?).

`queue::submit` could fail as well (why?). So same problem here.
If `queue::submit` fails, then there's a guarantee that the handler callback was not called and no resources were touched.
=> queue submit cannot fail once the resources have been modified: no rollback => add_task is noexcept

Question: explicit batch?
Could be useful to recover from an exception during submission (just drop the batch). However it needs some refactoring work. 
There's also the problem that currently it's unclear when a batch starts and ends. This is somewhat implicit.

Issues with exception safety: we can't mutate resources for tracking, because the tracking info may become invalid if batch submission fails because of an exception.
- tmp_index
- last_write_sequence_number

Replace with a function in batch:
```
batch::get_resource_tmp_index(resource&) -> size_t;
queue::last_write_sequence_number(resource&) -> uint64_t;
```

Need to store a `map<resource*,size_t>`. Must be efficient.
TODO Assign a generational index to resources. 
There needs to be a big map resource->size_t in the queue to track sequence numbers across batches.
=> don't do that. instead, once a batch is submitted, "commit" the last_write_sequence_number to all resources (there's nothing cancellable above batches anyway)

Questions:
- can an individual submission fail? YES
	- and leave the queue in an indeterminate state? NO
- can a batch fail? 
	- if yes, then the queue should be in a determinate state if a batch fails mid-building
	- otherwise, don't care

This boils down to: where should we put the "point-of-no-return" for exceptions?
	- task submit?
	- batch submit?

One could argue that the point of no return should be the submission to the backend API (i.e. vulkan). In this case,
that's _batch submit_.

## Explicit batches
```cpp
int main() {
	queue q{...};

	batch b{q};
	image img{...};

	try {
		b.submit([](handler& h) { ... });
		b.submit([](handler& h) { ... });
		b.submit([](handler& h) { ... });
		//...
		b.submit([](handler& h) { ... });

	} catch(std::exception e) {
		// queue is unaffected here
		// return or rethrow.
		return -1;
	}

	q.submit(std::move(b));	// could also submit in the destructor of the batch, but that's sketchy.
}
```
Issues:
- one more concept exposed to the user(batches)
- b.submit() could return awaitable events. However waiting on them will deadlock q.submit() has been called. That's one more way to screw up. 

## Implicit batches
```cpp
int main() {
	queue q{...};

	image img{...};

	try {
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		q.submit([](handler& h) { ... });
		//...
		q.submit([](handler& h) { ... });

	} catch(std::exception e) {
		// state of the queue may be affected here
		// option 1: reset the current batch (but it's unclear which commands will be cancelled)
		//q.reset();
		// option 2: do nothing; what's submitted can't be taken back. **
	}

	q.finish_batch();	// this must be called
}
```
Issues:
- could forget to finish the batch
- finish_batch() is sometimes implicit
	- waiting on a submit-returned event can force a finish_batch
	- destruction of the queue forces a finish_batch and waitDeviceIdle
- submits can't be taken back
- can't create multiple batches in parallel
	- is that useful?
	- do games do that?
	- command buffer creation would be the thing that takes the most time anyway
		- might be able to pipeline it across batches 
		- create command buffers asynchronously, but allow the next batch to start


#### Conclusion:
Let's go with implicit batches for now. 

#### When to generate the command buffer?

Determined during batch analysis:
- memory allocations
According to the vulkan spec (https://www.khronos.org/registry/vulkan/specs/1.2-extensions/html/vkspec.html#resources-association)

	Non-sparse resources must be bound completely and contiguously to a single VkDeviceMemory object 
	before the resource is passed as a parameter to any of the following operations:
    	* creating image or buffer views
    	* updating descriptor sets
    	* recording commands in a command buffer

So command buffer generation must happen after batch analysis

## Shared vs unique semantics for user-facing objects?
Right now buffer<>, image<> are copyable with shared reference semantics (inspired by SYCL).
Switch to unique semantics instead?

Problem: the underlying resource must be shared_ptr (references are stored in batches), so
we must add weird unique semantics on top of shared_ptr. Plus, if shared semantics are needed,
then must do `shared_ptr<image<>>`, which is a double pointer indirection (`shared_ptr<shared_ptr<image_impl>>`) 
So, no.


## Thread safety for external reference counting?

## Automatic pipeline barriers / layout transitions
Pipeline barriers are inserted _between_ tasks.

Each task has :
- a "consumer" pipeline stage mask that describes the stages that consume data from read dependencies.
	- corresponds to the "destination" stage mask of CmdPipelineBarrier
- a "producer" PSM that describes the stages that write data to write dependencies.
	- corresponds to the "source" stage mask of CmdPipelineBarrier

## Exception safety for resource usage flags
Buffers and images have usage flags (VkImageUsageFlags/VkBufferUsageFlags), which need to be set on construction 
and that determine how the resource will be used.
There are three options to keep track of required usages:

1. a "flags" member on the resource, updated whenever an accessor is constructed that references the resource
	- not exception safe if a task fails
2. a "flags" member on the resource, updated whenever a task referencing the resource is successfully submitted
	- current solution
3. no member, collect during batch submission just before allocating the resources
	- cleanest

## Consider making task handler callbacks noexcept
Handler callbacks that can throw forces us to delay registration of temporaries and accesses until 
the callback is finished (otherwise, may leave the DAG in an invalid state if an exception is thrown).

A noexcept callback would mean that we can incrementally build task dependencies without worrying that 
the task would suddenly become invalid because of an exception. This means that we can allocate the task in place
instead of moving things around.

## Pipeline barriers
Access flags and barriers are irrelevant for tasks that are not command buffer submissions (they are synchronized with semaphores).
For instance, a present operation is not a command, so there's no need to sync it with the pipeline and an access mode.
The only common information that we should store it is:
- whether we are reading or writing to the resource
- for images, the layout in which the image should be
The rest (pipeline barriers) should be specific to `submit_tasks`.

Two options:
- store only resource+layout in base `task`, pipeline barriers in `submit_task`
- store resource+layout+pipeline barriers in `task`, ignore barriers for `present_task`.

Third option:
- remove present tasks from the DAG.
Present operations will then be something that operates outside the DAG, forcing the resolution of the current batch.
This may be preferable.

Questions:
1. do we want to "schedule" present operations within a DAG? 
2. what other things can be scheduled on queues and do they correspond to a task?

For 2., the operations we can do on a queue are:
- submit a batch of command buffers
- present
- bind device memory to a sparse resource object
- insert debug markers 
- wait idle
- do perf queries (vkQueueSetPerformanceConfigurationINTEL)

Out of all of these, only sparse memory bindings is relevant to the DAG. 
The question becomes:
- what is sparse binding? what are the typical use cases?
- do we want to schedule sparse binds within a batch?
	- i.e. can sparse binds be done in the middle of a batch?
	- probably not? (do it between batches) 

Problem with that: they might not need to be scheduled, but they still need to be assigned a serial number.
So, if there are no tasks assigned to bind sparse operations, 
then suddenly the serial numbers do not map one-to-one with tasks anymore.

For 1., not scheduling present operations means that 

## Problem with separating present operations from the graph
When presenting, we must submit a command buffer to transition the image layout to PRESENT_SRC if it's not
already in this layout.
This causes a lot of complexity because this operation needs a command buffer and semaphore that we need to manage 
outside the batch mechanism.
- 
Ideally we'd like to put the transition in the previous batch, but it's not possible if 
it has already been submitted.


## Problem with Renderpasses
It is recommended to use renderpass external dependencies instead of pipeline barriers where possible. However, 
we don't know which external dependencies are required before the graph is submitted, which means that
renderpasses, and thus all pipelines, cannot be created before submission, which is *annoying*.

Solution: embrace caching. Use [Fossilize](https://github.com/ValveSoftware/Fossilize). 

Q: where to create renderpasses?
A: if merging renderpasses, then it should be the queue

Q: should we automatically try to merge renderpasses in the queue?

## Renderpass API

Given:
```cpp
image A;
image B;
image C;
```

### Option A: accessors
```cpp
q.render_pass([&](handler& h) {
	color_attachment a {A, h, clear{ ... }};
	color_attachment b {B, h, load};
	depth_attachment c {C, h};

	// ... other resources: uniforms, storage, etc ...

	h.commands([&](const command_context& ctx) {
		// render pass and framebuffer already bound here
	});
});
```
Issues: attachment index depends on the order of declaration.

### Option B: framebuffer accessor
```cpp
q.render_pass([&](handler& h) {
	framebuffer fbo {h, 
		/* color attachments */ {
			color_attachment{A, clear{}},
			color_attachment{B, load{}}, 
		},
		/* depth attachment */ depth_attachment{C}
	}; 

	// ... other resources: uniforms, storage, etc ...

	h.commands([&](const command_context& ctx) {
		// render pass and framebuffer already bound here
	});
});
``` 
The order is clear. FBO is accessible within a command context via the accessor.
Issue: 
* creating multiple framebuffers is invalid.
* attachments not bound to a variable, must reference by index 

### Option C: arguments
```cpp
q.render_pass( 
	/* color attachments */ { 
		color_attachment{A, clear{}},
		color_attachment{B, load{}}, 
	},
	/* depth attachment */ 
	depth_attachment {C},
	[&](handler& h) {

		// ... other resources: uniforms, storage, etc ...

		h.commands([&](const command_context& ctx) {
			// render pass and framebuffer already bound here
		});
	} 
);
```
Advantages: unambiguous
Issues: 
* harder to parse visually? (use a render_pass_desc struct if necessary)
* attachments not bound to a variable, must reference by index 

Option C seems best. Referencing by index is not that big of a deal (only used in vkCmdClearAttachments?)

Syntax proposal:
```cpp
render_pass_desc rpd {
	.color_attachments = { color_attachment{A, clear{...} } },
	.depth_attachment = depth_attachment{ },
	.input_attachments = { ... }
};

q.render_pass(rpd, [&](handler& h) {

});


color_attachment{A, attachment_load_op::clear, attachment_store_op::store };
depth_attachment{B, attachment_load_op::, attachment_store_op::dont_care };

```

## Accessing resources
Issue: it's annoying to access resources and properties of resources (need casts and whatever).
There are three types:

- buffer_impl: for (virtual) buffers
- image_impl: for (virtual) images
- swapchain_image_impl: for swapchain images

All derive from resource. buffer_impl and image_impl derive from virtual_resource.
Problem: to get the format of an image, must cast. Annoying because there's no common base class for swapchain images and regular images.

Proposal: put more things in the base
```
struct resource {
	resource_type type;
	union {
		VkImage image;
		VkBuffer buffer;
	} object;

}
```

Other proposal: a base class for image.
Issue: multiple-inheritance necessary with virtual_resource.

### Resource usage flags
They must be specified on creation (like width, height and format).

```
buffer<T> buf { buffer_usage_flag::uniform_buffer | buffer_usage_flag::vertex_buffer }
```

# Staging resources
Buffers that can be mapped in CPU memory just after they are created, and that have a short life (only live in the batch).
Currently, there's no way to tell that a buffer will only be used in the current batch. 
It can be deduced during submission, but that's too late: the buffer must already be allocated before submission in order to 
map it and upload data. And it's good to know that before allocation because then it can be allocated in a ring buffer.
Possible designs:
- `staging_buffer<T>`: not a resource, managed by the queue, mapped. Throws an exception when used outside of the current batch.
	- meh
- `queue.with_staging_buffer(<lambda>)`: same but the staging buffer cannot escape the lambda
	- all uses of the staging buffer must happen in the lambda

## Sub-problem: it's not clear what the "current batch" means
- "until finish_batch is called"


# Losing track of the goal for the task graph
Focus on a simple goal, don't try to stuff more things into the task graph.
- Don't lock into a design (e.g. accessors). Provide a minimal API that can do automatic scheduling and synchronization, 
  but enable use of raw vulkan for the rest.
   - this means that it should be possible to record command buffers manually, and access the vulkan objects directly
   - set aside accessors for now
- possibly simplify the image class? require full specialization
- allow manual synchronization (e.g. for presentation)
- don't put more things in the queue
	- staging buffers? could be implemented out-of-queue, and then "recycled" at the end of the batch
		- `staging_pool`
		- problem: no tracking of the staging buffer 
	- problem: functions that load image data now need both a ref to the queue and the staging pool. meh.

Prioritize:
- 

# Remove partial specification of objects
Adds complexity for no good reason. Create VkImage in constructor.
Alternate syntaxes:
```
int main() {
	graal::image { <usage>, <format>, {128,128,128}, image_properties { 
		.mip_levels = 1,
		.aliasable = true
	}};
}
```

`HostVisible` doesn't make much sense (it's always visible to the host one way or another, via copies to readback buffers).
The template parameter is supposed to enable/disable some methods: which ones?
Readback should always be possible.
Mapping the memory is different though. 

### Do we even care about the `image_type Type` template parameter?
It only serves to enable `height` or `depth` if it's 2D or 3D. It doesn't provide any accessor methods that depend on the dimension, unlike the SYCL counterpart.
It might be useful when binding the image to a descriptor, to statically check that the image has the correct dimensions, but it's not there yet. 
And there will probably be other checks that need to be made at runtime anyway.

# Management of vulkan handles by resources
The handle to objects are stored in the base class `image_resource` or `buffer_resource` or `resource` (access semaphores), 
but it's the responsibility of the derived class `image_impl` to do the cleanup?
-> The main issue is that there's a base `resource` class, but is doesn't represent ownership of the resource (only the derived class does).
Idea: replace inheritance with composition:
- image_impl owns a `shared_ptr<resource>`
- swapchain_impl owns one resource per image

## Getting rid of swapchain resources?
Swapchain images are annoying because we don't really own them once they are passed to presentation.
Also, they need binary semaphores for synchronization. 
Some notes:
- they don't need to be reclaimed since they are owned by the swapchain
- they can be used like regular images in the pipeline, so it must be tracked like a `resource`
- don't expose swapchain images, only provide accessors to swapchains
TODO:
- augment image_impl with a way to create references to external images
	-> image_resource may or may not own the resource depending on a flag 



# Issue: granularity of access tracking
Until now, we assumed that we could track the resource states at the granularity of the resource, that is, track the state of the whole image,
or the whole buffer, etc.
But in practice, resources are not always accessed as a whole. For example, only a range of bytes inside a buffer needs to be accessed, or one mip level of an image, etc.
This is a problem because since we're tracking resource states at a high granularity, we need to ensure that the whole resource is in a known state
between passes instead of just the part that is accessed. This means performing a layout transition on all array layers and mip levels for images, 
emitting a memory barrier for the whole buffer instead of just the accessed region, etc.

Unfortunately, keeping track of accesses at a smaller granularity seems more complex. 
For images, we need to maintain a different state for each subresource.
For buffers, we need to maintain a dynamic list of ranges for each state.



## Access tracking in existing render graph systems
[Granite](https://github.com/Themaister/Granite/blob/master/renderer/render_graph.hpp) doesn't track accesses at subresource granularity.
[Diligent](https://diligentgraphics.com/2018/12/09/resource-state-management/) allows transitions on subresource ranges, 
but does not seem to track states per subresource (https://github.com/DiligentGraphics/DiligentCore/blob/36f11d692acd2272c4b15b34a3753f565310a074/Graphics/GraphicsEngineVulkan/src/DeviceContextVkImpl.cpp#L2245), so I'm not sure automatic transitions even work correctly.
EDIT: the article says this:

	The state is tracked for the whole resource only. Individual mip levels and/or texture array slices cannot be transitioned.

[V-EZ](https://github.com/GPUOpen-LibrariesAndSDKs/V-EZ/blob/master/Source/Core/PipelineBarriers.cpp) implements fine-grained resource tracking, which is impressive:

	IMPLEMENTATION NOTES:    
        This class handles tracking resource usages within the same command buffer for automated pipeline barrier insertion.

        Buffer accesses are tracked per region with read-combining and write-combining done on adjacent 1D ranges.
        Buffer accesses are stored in an STL map keyed by the buffer's memory address, offset and range.
        Image accesses are tracked per array layer and mip level ranges.  Read and write combining are performed between 2D ranges where
        the array layer is treated as the x-coordinate and mip level as the y-coordinate. If two accesses' rectangles intersect, then either
        their regions are merged into a larger rectangle or a pipeline barrier is inserted if the accesses require it.
        Images are stored in an STL unordered_map keyed by the image's memory address and value being a linked list of all accesses.
    	
        This implementation likely needs to be optimized and improved to handle the cases of random scattered accesses across images and buffers as
        the process of merging and pipeline barrier insertion could become quite expensive.  However in the ideal case where accesses and linear and semi-coallesced,
        the performance should not be an issue.



## Aside: diminishing returns
Fine-grained access tracking is probably too complex to implement and overkill. We need to be careful about diminishing returns
when implementing such a thing, and think back on the original goal, which is to be able to easily write passes with vulkan:
- be able to write self-contained passes that don't need to know about other passes
	- without having to care about synchronization *with other passes*
- abstract memory allocation and aliasing

An important principle is that we should focus on making the *composition* of passes easy, but give low-level access *inside* a pass: basically,
we control the *interface* between passes, but not the inside. This means that a pass can do whatever it want inside their command buffers, 
as long as it honors the interface that it declared.

Tracking accesses at the resource granularity is a compromise.
However, this means that a task cannot request access to two layers of the same image with different layouts 
(it must perform a transition inside the task).
- Or rather, it can, and the queue will honor the request, but it will internally make sure that between passes all subresources
  are in the same layout.

Note: in this light, accessors seems like a complication: they are here to simultaneously register an access to a resource within a task
*and* "safely" represent the actual use of the resource in the command buffer. But we don't provide such "safe" API for building command buffers yet, 
and it is a massive undertaking.

Example of "safe" API for command buffers:
```cpp
q.compute_pass([](handler& h) {
	accessor a{img, ...};
	accessor b{img2, ...};

	h.commands([=](command_buffer& cb) {
		cb.copy(a,b);
	});
});
```

But right now we just expose a vk::CommandBuffer:
```cpp
q.compute_pass([](handler& h) {
	accessor a{img, ...};
	accessor b{img2, ...};

	h.commands([=](vk::CommandBuffer& cb) {
		cb.copy(img,img2);
	});
});
```
So the accessors are never referenced again. The accessors and the whole outer lambda could be eliminated by explicitly passing
a list of accesses in the compute pass:
```cpp
q.compute_pass( 
	{access(img, ...), access(img, ...)},
	[&](vk::CommmandBuffer&) {
		//...
	}
)
```

That said, existing rendergraph APIs do use this pattern. (e.g. https://www.khronos.org/assets/uploads/developers/library/2019-reboot-develop-blue/SEED-EA_Rapid-Innovation-Using-Modern-Graphics_Apr19.pdf).
So don't change it for now.


## Fine-grained access tracking
Makes less sense to store it per-resource.

A big list of "facts" about memory accesses. Each fact concerns a "region", which can be an image subresource range, or a slice of a buffer.
The fact contains:
- the last write serial
- the last access serial (read or write)
- the last pipeline stage accessing the resource
- the last known image layout (for image subresources)
- current memory access flags (in what kind of memory the results of the last operation is visible)

The big issue is to update this list with new facts, given that the new regions *may overlap* regions of existing facts. 
- Split/merge regions
	- basically one tree per property, meh.
	- this moves the state tracking outside of resources, through, which is nice


## Submission refactor: access tracking v3
- Remove tracking fields from resource classes, put them into the queue.
- Remove per-resource access tracking, instead track accesses on _regions_ (parts of resource)
	- as a start, only allow regions that span whole resources
- to solve:
	- lookup using a region (resource + subresource for images, buffer + range for buffers)
	- what data structure for the "region->resource state" map?
		- fast lookup, insertion and deletion
		- don't care much about iteration
		- entries should be trivially constructible and destructible
	- when to remove entries from the map?
- note that D3D12 doesn't have memory barriers for ranges of buffers (the whole buffer is transitioned)

## Per-subresource state tracking?
- currently done at the resource level
- doing it for arbitrary buffer ranges seems overkill (for instance, D3D12 can't transition buffer ranges)
- however, both Vulkan and D3D12 can have subresources (image layer, mipmap) of the same resource in different states 

## Simplify barrier emission
- resource_state

## In which cases is it useful to transition individual subresources?
- generating mip maps
- computing an image pyramid
- downsampling
In all of those cases it seems that the individual subresource transitions could be "hidden" inside the pass,
and not exposed outside. 


# The problem with presentation
The main problem is that presentation doesn't support timeline semaphores, so it complicates
synchronization when accessing a swapchain image.

## Simplify swapchain-related things
Currently, 4 classes:
- swapchain, swapchain_impl, swapchain_image, swapchain_image_impl

Idea: make swapchain an image resource that represents the current backbuffer.
(Can't acquire two images at once, but why would that be useful?)
It would make more sense than the current swapchain_image: `swapchain_image` has shared
pointer semantics, but `present` is supposed to consume the image (ownership transfer),
which invalidates all other references.

## Resource tracking for swapchains?
- First use: image ready
- Before handoff to presentation: rendering finished
When presenting, immediately acquire the next image.


# Remove resource wrappers?
I.e. merge image and image_impl, etc. Expose a pointer type (user_resource_ptr) instead of a wrapper object.
Pros:
- less duplicated code
- null state (is that a pro?)
Cons:
- null state
- no pimpl (but that's already the case)
- no direct constructors (must use a factory function to create the pointer)

Why does SYCL have wrappers?
- is it for pImpl?
SYCL inspired wrappers. Object-like syntax (`.method`), but shared-ptr like semantics, except that there's no null state.
Remove and use `shared_ptr<image>` directly?

Don't use pointers at all?
- allow creation of images on the stack
- no shared semantics
- unique_ptr or shared_ptr as needed by the user
- Option A: resources must be alive at least as long as the current batch
	- annoying to ensure, error-prone
- Option B: resources must only be alive when creating a pass, can dropped immediately after
	- when using a resource in a queue, ownership is transferred to the queue
		- problem: the queue will eventually finish using the image, but this does not mean that the image should be deleted
			- ownership should be re-transferred to the image
			- 
	- if the resource object doesn't have ownership, destructor does nothing
	- otherwise, it is automatically released
- Option C:
	- destructors actually destroy the resource
		- they wait for the queue to finish using it
	- however, it's possible to explicitly transfer ownership of a resource to the queue
		- `queue.discard(resource)`: the queue takes on the responsibility of destroying the resource once the GPU finishes using it
	- the resource object itself has very little state
		- a reference to the queue that is using it, if any
		- the handle
		- the create info
	- tracking information is moved into the queue
		- map VK handle to index (handle->index)
		- stored in vectors
	- basically, decouple the resource classes from the queue
		- the user can have the queue track its own VkImage/VkBuffers

	- registering a resource
		- `queue_impl.add_resource_dependency(task, VkImage/VkBuffer, ...)`
			- lookup index of resource (VkImage->size_t)
			- lookup index of temporary (VkImage->size_t)
			- implicitly registers the resource to the queue
	- unregistering a resource
		- `queue.unregister_resource(VkImage)`
			- will wait for the GPU to finish using the resource, and 
	- transferring ownership to the queue:
		- `queue.discard(VkImage/VkBuffer)`
		- resource will be automatically freed when the GPU is finished with it
	- Problem: lots of map lookups (VkImage->...)
		- maybe cache the resource index?
	- All of this is only so that resource tracking can work on arbitrary handles
		- it is worth it?
			- maybe: users don't have to buy into the resources classes if they want to manage their objects in some other way
				- in which cases would it be more convenient for the user to have raw vulkan handles instead of our wrappers?
					- our wrappers also store a bunch of information like width, height, format, etc. format requied for image barriers.
					- also the allocation, which is useless for external images.

## Time for a survey: resource classes in frameworks that automatically handle synchronization
Look for:
- shared VS unique semantics
- can work with raw handles

- [Acid](https://github.com/EQMG/Acid)
	- shared_ptr, no raw handles, synchro unknown
- [Diligent engine]()
	- ref-counted ptr, state kept in resource
- [falcor](https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Core/API/Texture.h)
	- shared_ptr
- [bsf](https://github.com/GameFoundry/bsf/tree/master/Source/Plugins/bsfVulkanRenderAPI)
	- tracking info in resource
	- shared_ptr semantics

## What is the absolute minimum we need to track for automatic barriers?
- handle, obviously
- format of images (for image memory barriers: we need the aspect mask that corresponds to the format)
	- specify the aspect mask in the accessor instead?
- buffers: nothing?


# Formalize the submission problem
- Inputs: DAG of tasks that reference resources
- Outputs:
	- command buffer batches
		- barriers
		- synchronization
		- binary semaphores
	- final resource states


# Queue simplification
- determine barriers "on the fly" (incrementally, as the graph is built)
	- no need to store accesses in the task (directly update the half-barrier instead)
- do not reorder passes?
- usage flag deduction?
	- error prone with the current design, as the usage flags is "fixed" on resource creation (during first batch submission)
	  but the image can then be used after the batch is submitted
	  	- no real transients: all resources can be extended to live after the current batch
	  		- introduce real transients?
	  		- lifetime is unclear
- problem: build barriers after DAG building
	- need to "replay" the DAG (barriers need tracking of the current readers and writers) or store a vector of readers and writers per access

# No reordering
The tasks are submitted to the queue in the order they appear in the API.
This can lead to suboptimal scheduling.

Problems with reordering: 
- reordering makes things more complicated in the submission backend
- 

# Direct3D 12 port?
- resources: same, but don't allocate resource in advance
- VMA => D3D12 memory allocator (or just use commited resources for now)
- resource state => replace layout + access flags with resource state transitions
- remove render passes
- 

# Rust port?

# To fix:
- remove VkRenderPass creation responsibility from queue
- maybe rethink swapchain access?
- single command buffer per submission (except when requested, for parallel command recording)
- reclaim resources

# Unify queue and device
- no reason to have both, there's only one device, and there's only one "queue" (which does *not* correspond to a vulkan queue)
- rename to "context" or something
- create a device without a queue?
	- possible, use "device" only as a convenience class to create vulkan queues

## Log
28/11 : pipeline state tracking and "queue classes". Can deduce execution and memory dependencies.
29/11 : refactored scheduling
	- split scheduling, barrier deduction into several functions, under a common context (schedule_ctx)
	- interleave liveness analysis with scheduling so that it is correct w.r.t. pipeline usage
	- submission numbers (SNN)
	- laid down API for render passes
30/11 : cross-queue syncs
 	- track the current usage of each resource across queues (whether it's used in only one queue or possibly being read across multiple queues simultaneously). This is necessary for the determination of cross-queue syncs.
	- removed last_write_access. need only last_access and whether the last access was a read or a write
	- moved resource tracking variables (lastXXX) from resource to an array in schedule_ctx (more tightly-packed in memory). 
5/12 : submissions


