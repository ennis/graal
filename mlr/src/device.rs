//! Wrapper around device and queues.
use crate::sampler::{Sampler, SamplerCreateInfo};
use std::{cell::RefCell, collections::HashMap, rc::Rc};

pub struct Device {
    device: Rc<graal::device::Device>,
    sampler_cache: RefCell<HashMap<SamplerCreateInfo, Sampler>>,
}

impl Device {
    pub fn create_sampler(&self, info: &SamplerCreateInfo) -> Sampler {
        if let Some(sampler) = self.sampler_cache.borrow().get(info) {
            return sampler.clone();
        }

        let sampler = Sampler::new(self.device.clone(), info);
        self.sampler_cache.borrow_mut().insert(info.clone(), sampler.clone());
        sampler
    }
}
