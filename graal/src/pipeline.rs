use ash::vk;
use std::sync::Arc;

// Goals: cheap to clone, cheap to modify
// Partition in state groups, each group in Arc,
//
pub struct PipelineQuery {
    vertex: Option<Arc<[u32]>>,
    fragment: Option<Arc<[u32]>>,
}

/*fn make_arc_slice(spirv:&[u32]) -> Arc<[u32]> {
    unsafe {
        let mut arr = Arc::new_uninit_slice(spirv.len());
        ptr::copy(spirv.as_ptr(), arr.as_mut_ptr() as *mut u32, spirv.len());
        arr.assume_init()
    }
}

impl PipelineQuery {
    pub fn new() -> PipelineQuery {
        PipelineQuery {
            vertex: None,
            fragment: None
        }
    }

    pub fn vertex_shader(&mut self, spirv: &[u32]) -> &mut Self {
        self.vertex = make_arc_slice(spirv).into();
        self
    }

    pub fn fragment_shader(&mut self, spirv: &[u32]) -> &mut Self {
        self.fragment = make_arc_slice(spirv).into();
        self
    }

    // Those pipeline queries may be long-lived; e.g. stored in a struct

    // Solutions:
    // - borrows
}

*/
