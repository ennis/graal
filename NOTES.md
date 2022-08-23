# Module organization

Right now the `context` module is a mess that contains all of the following in a single file:
* frame pacing 
* autosync
* frame creation 
* submission

Queue submission can probably be moved to another module.
The "Pass" type, given that it's used by both frame creation, autosync and submission, should probably be moved into the root module.
Same for frame?

Right now, autosync locks the device objects so that resource tracking info can be updated.
Would it be better to store temporary tracking info somewhere else and then update the trackers afterwards?
=> why? it just makes it less efficient