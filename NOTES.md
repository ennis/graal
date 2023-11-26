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

If a resource is used on multiple queues, require the user to synchronize with all queues before deleting the 
