use crate::MAX_QUEUES;
use crate::context::{SubmissionNumber, QueueSerialNumbers};
use ash::vk;
use std::mem;

#[derive(Copy, Clone, Debug)]
#[repr(usize)]
enum PipelineStageIndex {
    TopOfPipe,
    DI,
    VI,
    VS,
    TCS,
    TES,
    GS,
    TF,
    TS,
    MS,
    FSR,
    EFT,
    FS,
    LFT,
    CAO,
    FDP,
    CS,
    RTS,
    HST,
    CPR,
    ASB,
    TR,
    CR,
    BottomOfPipe,
    // This is not really a stage, but it is considered for tracking purposes as a pseudo-stage
    //AllCommands,
    //AllGraphics,
    MAX,
}

impl PipelineStageIndex {
    fn from_flags(flags: vk::PipelineStageFlags) -> Option<PipelineStageIndex> {
        use PipelineStageIndex::*;
        match flags {
            vk::PipelineStageFlags::TOP_OF_PIPE => Some(TopOfPipe),
            vk::PipelineStageFlags::BOTTOM_OF_PIPE => Some(BottomOfPipe),
            vk::PipelineStageFlags::DRAW_INDIRECT => Some(DI),
            vk::PipelineStageFlags::VERTEX_INPUT => Some(VI),
            vk::PipelineStageFlags::VERTEX_SHADER => Some(VS),
            vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER => Some(TCS),
            vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER => Some(TES),
            vk::PipelineStageFlags::GEOMETRY_SHADER => Some(GS),
            vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT => Some(TF),
            vk::PipelineStageFlags::TASK_SHADER_NV => Some(TS),
            vk::PipelineStageFlags::MESH_SHADER_NV => Some(MS),
            vk::PipelineStageFlags::SHADING_RATE_IMAGE_NV => Some(FSR),
            vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS => Some(EFT),
            vk::PipelineStageFlags::FRAGMENT_SHADER => Some(FS),
            vk::PipelineStageFlags::LATE_FRAGMENT_TESTS => Some(LFT),
            vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT => Some(CAO),
            vk::PipelineStageFlags::FRAGMENT_DENSITY_PROCESS_EXT => Some(FDP),
            vk::PipelineStageFlags::COMPUTE_SHADER => Some(CS),
            vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR => Some(RTS),
            vk::PipelineStageFlags::HOST => Some(HST),
            vk::PipelineStageFlags::COMMAND_PREPROCESS_NV => Some(CPR),
            vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR => Some(ASB),
            vk::PipelineStageFlags::TRANSFER => Some(TR),
            vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT => Some(CR),
            _ => None
        }
    }
}

const PIPELINE_STAGES_COUNT: usize = PipelineStageIndex::MAX as usize;

static LOGICALLY_EARLIER: &[&[PipelineStageIndex]] = {
    use PipelineStageIndex::*;

    &[
        /*DI*/  &[],
        /*VI*/  &[DI],
        /*VS*/  &[DI, VI],
        /*TCS*/ &[DI, VI, VS],
        /*TES*/ &[DI, VI, VS, TCS],
        /*GS*/  &[DI, VI, VS, TCS, TES],
        /*TF*/  &[DI, VI, VS, TCS, TES, GS],
        /*TS*/  &[DI],
        /*MS*/  &[DI, TS],
        /*FSR*/ &[DI, VI, VS, TCS, TES, GS, TF],
        /*EFT*/ &[DI, VI, VS, TCS, TES, GS, TF, FSR, FDP],
        /*FS*/  &[DI, VI, VS, TCS, TES, GS, TF, FSR, FDP, EFT],
        /*LFT*/ &[DI, VI, VS, TCS, TES, GS, TF, FSR, FDP, EFT, FS],
        /*CAO*/ &[DI, VI, VS, TCS, TES, GS, TF, FSR, FDP, EFT, FS, LFT],
        /*FDP*/ &[],
        /*CS*/  &[DI],
        /*RTS*/ &[DI],
        /*HST*/ &[],
        /*CPR*/ &[],
        /*ASB*/ &[],
        /*TR*/  &[],
        /*CR*/  &[],
    ]
};

static LOGICALLY_LATER: &[&[PipelineStageIndex]] = {
    use PipelineStageIndex::*;

    &[
        /*DI*/  &[CAO, LFT, FS, EFT, FSR, TF, GS, TES, TCS, VS, TS, MS, CS, RTS],
        /*VI*/  &[CAO, LFT, FS, EFT, FSR, TF, GS, TES, TCS, VS],
        /*VS*/  &[CAO, LFT, FS, EFT, FSR, TF, GS, TES, TCS],
        /*TCS*/ &[CAO, LFT, FS, EFT, FSR, TF, GS, TES],
        /*TES*/ &[CAO, LFT, FS, EFT, FSR, TF, GS],
        /*GS*/  &[CAO, LFT, FS, EFT, FSR, TF],
        /*TF*/  &[CAO, LFT, FS, EFT, FSR],
        /*TS*/  &[CAO, LFT, FS, EFT, FSR, TS],
        /*MS*/  &[CAO, LFT, FS, EFT, FSR],
        /*FSR*/ &[CAO, LFT, FS, EFT],
        /*EFT*/ &[CAO, LFT, FS],
        /*FS*/  &[CAO, LFT],
        /*LFT*/ &[CAO],
        /*CAO*/ &[],
        /*FDP*/ &[CAO, LFT, FS, EFT],
        /*CS*/  &[],
        /*RTS*/ &[],
        /*HST*/ &[],
        /*CPR*/ &[],
        /*ASB*/ &[],
        /*TR*/  &[],
        /*CR*/  &[],
    ]
};

/// Expands VkPipelineStageFlags to explicitly include logically earlier stages
/*fn expand_logically_later_stages(stage: vk::PipelineStageFlags) {

    const DI : u32 = vk::PipelineStageFlags::DRAW_INDIRECT.as_raw();
    const VI : u32 = vk::PipelineStageFlags::VERTEX_INPUT.as_raw();
    const VS : u32 = vk::PipelineStageFlags::VERTEX_SHADER.as_raw();
    const TCS : u32 = vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER.as_raw();
    const TES : u32 = vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw();
    const GS : u32 = vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw();
    const TF : u32 = vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT.as_raw();
    const TS : u32 = vk::PipelineStageFlags::TASK_SHADER_NV.as_raw();
    const MS : u32 = vk::PipelineStageFlags::MESH_SHADER_NV.as_raw();
    const FSR : u32 = vk::PipelineStageFlags::SHADING_RATE_IMAGE_NV.as_raw();
    const EFT : u32 = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw();
    const FS : u32 = vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw();
    const LFT : u32 = vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw();
    const CAO : u32 = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw();
    const FDP : u32 = vk::PipelineStageFlags::FRAGMENT_DENSITY_PROCESS_EXT.as_raw();
    const CS : u32 = vk::PipelineStageFlags::COMPUTE_SHADER.as_raw();
    const RTS : u32 = vk::PipelineStageFlags::RAY_TRACING_SHADER_KHR.as_raw();
    const HST : u32 = vk::PipelineStageFlags::HOST.as_raw();
    const CPR : u32 = vk::PipelineStageFlags::COMMAND_PREPROCESS_NV.as_raw();
    const ASB : u32 = vk::PipelineStageFlags::ACCELERATION_STRUCTURE_BUILD_KHR.as_raw();
    const TR : u32 = vk::PipelineStageFlags::TRANSFER.as_raw();
    const CR : u32 = vk::PipelineStageFlags::CONDITIONAL_RENDERING_EXT.as_raw();

    const ALL_COMMANDS: u32 =
        DI
        | VI
        | VS
        | TCS
        | TES
        | GS
        | TF
        | TS
        | MS
        | FSR
        | EFT
        | FS
        | LFT
        | CAO
        | FDP
        | CS
        | RTS
        | HST
        | CPR
        | ASB
        | TR
        | CR;

    const ALL_GRAPHICS: u32 = DI | VI | VS | TCS | TES | GS | TF | FSR | FDP | EFT | FS | LFT | CAO;


    match stage {
        vk::PipelineStageFlags::ALL_COMMANDS => { ALL_COMMANDS },
        vk::PipelineStageFlags::ALL_GRAPHICS => { ALL_GRAPHICS },
        vk::PipelineStageFlags::TOP_OF_PIPE => { ALL_COMMANDS },
        vk::PipelineStageFlags::BOTTOM_OF_PIPE => { 0 },
        _ => {
            let mut result = stage.as_raw();
            if stage & vk::PipelineStageFlags::DRAW_INDIRECT {
                result |= CAO | LFT | FS | EFT | FSR | TF | GS | TES | TCS | VS | TS | MS | CS | RTS;
            }
            if stage & vk::PipelineStageFlags::VERTEX_INPUT {
                result |= CAO | LFT | FS | EFT | FSR | TF | GS | TES | TCS | VS;
            }
            if stage & vk::PipelineStageFlags::VERTEX_SHADER {
                result |= CAO | LFT | FS | EFT | FSR | TF | GS | TES | TCS;
            }
            if stage & vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER {
                result |= CAO | LFT | FS | EFT | FSR | TF | GS | TES;
            }
            if stage & vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER {
                result |= CAO | LFT | FS | EFT | FSR | TF | GS;
            }
            if stage & vk::PipelineStageFlags::GEOMETRY_SHADER {
                result |= CAO | LFT | FS | EFT | FSR | TF;
            }
            if stage & vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT {
                result |= CAO | LFT | FS | EFT | FSR;
            }
            if stage & vk::PipelineStageFlags::TASK_SHADER_NV {
                result |= CAO | LFT | FS | EFT | FSR | TS;
            }
            if stage & vk::PipelineStageFlags::MESH_SHADER_NV {
                result |= CAO | LFT | FS | EFT | FSR;
            }
            if stage & vk::PipelineStageFlags::SHADING_RATE_IMAGE_NV {
                result |= CAO | LFT | FS | EFT;
            }
            if stage & vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS {
                result |= CAO | LFT | FS;
            }
            if stage & vk::PipelineStageFlags::FRAGMENT_SHADER {
                result |= CAO | LFT;
            }
            if stage & vk::PipelineStageFlags::LATE_FRAGMENT_TESTS {
                result |= CAO;
            }
            if stage & vk::PipelineStageFlags::FRAGMENT_DENSITY_PROCESS_EXT {
                result |= CAO | LFT | FS | EFT;
            }
            result
        }
    }
}*/


static GRAPHICS_STAGES: &[PipelineStageIndex] = {
    use PipelineStageIndex::*;
    &[DI, VI, VS, TCS, TES, GS, TF, FSR, FDP, EFT, FS, LFT, CAO]
};

// COMPUTE -> VERTEX:
// -> m[VERTEX] |= COMPUTE | m[COMPUTE]
// -> m[FRAGMENT] |= COMPUTE | m[COMPUTE]
//
// FRAGMENT -> TRANSFER
// -> m[TRANSFER] |= FRAGMENT | m[FRAGMENT]

/*impl StageBarriers {
    pub fn add_barrier(&mut self, src: vk::PipelineStageFlags, dst: vk::PipelineStageFlags) {

        let src_expanded = expand_logically_earlier_stages(src);

        if dst.contains(vk::PipelineStageFlags::TOP_OF_PIPE) || dst.contains(vk::PipelineStageFlags::ALL_COMMANDS) {
            for d in self.dst_stages.iter_mut() {
                *d |= src_expanded;
            }
        } else {
            if dst.contains(vk::PipelineStageFlags::ALL_GRAPHICS) {
                for &i in GRAPHICS_STAGES {
                    &mut self.mtx[i] |= src_expanded;
                }
            }

            let dst_expanded = expand_logically_later_stages(dst);

            macro_rules! apply_flags_on_dst {
                ($src_stage_flag:ident, $src_stage_index:ident => $dst_stage_flag:ident, $dst_stage_index:ident) => {
                    if src_expanded.contains(vk::PipelineStageFlags::$src_stage_flag)
                        && dst_expanded.contains(vk::PipelineStageFlags::$dst_stage_flag) {
                        self.m[PipelineStageIndex::$dst_stage_index as usize] |= src_expanded | self.m[];
                    }
                };
            }

            apply_flags_on_dst!(DRAW_INDIRECT, DI);
            apply_flags_on_dst!(VERTEX_INPUT, VI);
            apply_flags_on_dst!(VERTEX_SHADER, VS);
            apply_flags_on_dst!(TESSELLATION_CONTROL_SHADER, TCS);
            apply_flags_on_dst!(TESSELLATION_EVALUATION_SHADER, TES);
            apply_flags_on_dst!(GEOMETRY_SHADER, GS);
            apply_flags_on_dst!(TRANSFORM_FEEDBACK_EXT, TF);
            apply_flags_on_dst!(TASK_SHADER_NV, TS);
            apply_flags_on_dst!(MESH_SHADER_NV, MS);
            apply_flags_on_dst!(SHADING_RATE_IMAGE_NV, FSR);
            apply_flags_on_dst!(EARLY_FRAGMENT_TESTS, EFT);
            apply_flags_on_dst!(FRAGMENT_SHADER, FS);
            apply_flags_on_dst!(LATE_FRAGMENT_TESTS, LFT);
            apply_flags_on_dst!(COLOR_ATTACHMENT_OUTPUT, CAO);
            apply_flags_on_dst!(FRAGMENT_DENSITY_PROCESS_EXT, FDP);
            apply_flags_on_dst!(COMPUTE_SHADER, CS);
            apply_flags_on_dst!(RAY_TRACING_SHADER_KHR, RTS);
            apply_flags_on_dst!(HOST, HST);
            apply_flags_on_dst!(COMMAND_PREPROCESS_NV, CPR);
            apply_flags_on_dst!(ACCELERATION_STRUCTURE_BUILD_KHR, ASB);
            apply_flags_on_dst!(TRANSFER, TR);
            apply_flags_on_dst!(CONDITIONAL_RENDERING_EXT, CR);
        }
    }
}*/

// instead of a matrix, use a smaller table Stage->(Stage,Serial)
// - per-stage, store last sync serial and stage
// - problem: let's say VS (vertex shader) is currently synced on SN30+TR (transfer)
//   we add a sync on VS for SN45+CAO (color attachment output)
//   -> we lose all information on the previous sync
//   now we want to sync VS on SN20+TR: we currently have SN45+CAO, but no information about TR, so must add a barrier, which is unnecessary


/*/// Represents the state (last synchronized write) of each stage of the pipeline at a given point in
/// the frame.
#[derive(Copy,Clone,Debug)]
pub(crate) struct SyncTable {
    // FIXME this is BIG (~16Kbytes)
    // Keeps track of all possible pipeline stage barriers
    // I don't remember why this is better than a linear search for a barrier between source and destination...
    // -> the problem was transitive execution dependencies

    // given a sequence of execution barriers (bitflags), determine whether there's an exec dependency
    // e.g.
    //      COMPUTE -> VERTEX     implies  COMPUTE -> FRAGMENT
    //      FRAGMENT -> TRANSFER  implies  VERTEX -> TRANSFER
    // combine:
    //      (src set) COMPUTE -> VERTEX,...,FRAGMENT (dst set)
    //      next: if
    //      VERTEX..FRAGMENT -> TRANSFER
    //
    // -> implies COMPUTE -> TRANSFER

    //   C V F T
    // C . 1 1 .
    // V . . . 1
    // F . . . 1
    // T . . . .
    // then compute the transitive closure
    // (is there a path from compute to transfer? yes, multiple:
    //  - C->V->T
    //  - C->F->T


    // compute->vertex,fragment->transfer

    // src -> dst
    //
    //
}*/


/*fn for_each_flag(mask: vk::PipelineStageFlags, mut f: impl FnMut(PipelineStageIndex) -> bool) {
    if mask == vk::PipelineStageFlags::ALL_GRAPHICS {
        for &i in GRAPHICS_STAGES {
            if !f(i) { break }
        }
    }
    else if mask == vk::PipelineStageFlags::ALL_COMMANDS {
        for i in 0..PIPELINE_STAGES_COUNT {
            // safety: consecutive enum values, with repr(usize), starting at 0
            if !f(unsafe { mem::transmute(i) }) { break }
        }
    }
    else {
        let mut m = mask.as_raw();
        while m != 0 {
            let m_bit = m & (!m).wrapping_add(1);
            let i = PipelineStageIndex::from_flags(vk::PipelineStageFlags::from_raw(m_bit)).unwrap();
            if !f(i) { break }
            m ^= m_bit;
        }
    }
}

fn for_each_flag_combination(src_stage_mask: vk::PipelineStageFlags,
                             dst_stage_mask: vk::PipelineStageFlags,
                             mut f: impl FnMut(PipelineStageIndex,PipelineStageIndex) -> bool)
{
    for_each_flag(src_stage_mask, |i_src| {
        for_each_flag(dst_stage_mask, |i_dst| {
            f(i_src, i_dst)
        });
        true
    });
}

impl SyncTable {
    pub(crate) fn new() -> SyncTable {
        SyncTable {
            stages: Default::default(),
            xq: Default::default()
        }
    }

    /// Returns the last synchronized pass serials (sync with a semaphore wait) on the given queue
    pub(crate) fn get_last_xq_sync(&self, queue: u8) -> QueueSerialNumbers {
        self.xq[queue as usize]
    }

    /// Returns the serial of the last pass synchronized with a pipeline barrier on the given queue.
    pub(crate) fn get_last_barrier_sync(&self,
                     queue: usize,
                     base_serial: u64,
                     src_stage: vk::PipelineStageFlags,
                     dst_stage: vk::PipelineStageFlags) -> u64
    {
        let mut sn = u32::MAX;

        for_each_flag_combination(src_stage, dst_stage, |i_src, i_dst| {
            let v = self.stages[queue][i_dst as usize][i_src as usize];
            sn = sn.min(v as u32);
            v != 0
        });

        if sn == u32::MAX {
            sn = 0;
        }

        base_serial + sn as u64
    }

    pub(crate) fn apply_xq_barrier(&mut self, queue: u8, serials: QueueSerialNumbers) {
        self.xq[queue as usize].assign_max(&serials);
    }

    pub(crate) fn apply_pipeline_barrier(&mut self,
                              base_serial: u64,
                              src: SubmissionNumber,
                              src_stage: vk::PipelineStageFlags,
                              dst_stage: vk::PipelineStageFlags)
    {
        assert!(src.serial() > base_serial);
        let src_local = src.serial() - base_serial;
        assert!(src_local <= u16::MAX as u64);
        let src_local = src_local as u16;
        let q = src.queue();

        let assign_max = |a: &mut u16, b: u16| { *a = (*a).max(b) };

        for_each_flag_combination(src_stage, dst_stage, |i_src, i_dst| {
            //println!("apply_pipeline_barrier ({:?}->{:?})", i_src, i_dst);
            let i_src = i_src as usize;
            let i_dst = i_dst as usize;
            assign_max(&mut self.stages[q][i_dst][i_src], src_local);

            for &i in LOGICALLY_EARLIER[i_src] {
                assign_max(&mut self.stages[q][i_dst][i as usize], src_local);
            }
            for &i in LOGICALLY_LATER[i_dst] {
                assign_max(&mut self.stages[q][i as usize][i_src], src_local);
            }

            true
        });
    }

    //pub(crate) fn dump()
}
*/