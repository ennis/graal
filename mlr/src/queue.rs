use crate::device::Device;
use std::rc::Rc;

pub struct Queue {
    device: Rc<Device>,
    queue: graal::queue::Queue,
}

impl Queue {
    /*pub fn submit(&self, command_buffers: &[graal::command::CommandBuffer]) {
        self.queue.submit(command_buffers);
    }*/
}
