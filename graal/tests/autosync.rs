#![feature(test)]

extern crate test;
use test::Bencher;

use ash::vk;
use graal::{ash::vk::PipelineStageFlags, QueueSerialNumbers, SubmissionNumber};
use std::{fmt, fmt::Formatter};

const EMPTY: vk::PipelineStageFlags = vk::PipelineStageFlags::empty();
const DI: vk::PipelineStageFlags = vk::PipelineStageFlags::DRAW_INDIRECT;
//const II : vk::Flags64 = vk::PipelineStageFlags2KHR::INDEX_INPUT;
//const VAI : vk::Flags64 = vk::PipelineStageFlags2KHR::VERTEX_ATTRIBUTE_INPUT;
const VS: vk::PipelineStageFlags = vk::PipelineStageFlags::VERTEX_SHADER;
const TCS: vk::PipelineStageFlags = vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER;
const TES: vk::PipelineStageFlags = vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER;
const GS: vk::PipelineStageFlags = vk::PipelineStageFlags::GEOMETRY_SHADER;
const TF: vk::PipelineStageFlags = vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT;
const FSR: vk::PipelineStageFlags = vk::PipelineStageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR;
const EFT: vk::PipelineStageFlags = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS;
const FS: vk::PipelineStageFlags = vk::PipelineStageFlags::FRAGMENT_SHADER;
const LFT: vk::PipelineStageFlags = vk::PipelineStageFlags::LATE_FRAGMENT_TESTS;
const CAO: vk::PipelineStageFlags = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT;
const CS: vk::PipelineStageFlags = vk::PipelineStageFlags::COMPUTE_SHADER;
//const TS : vk::PipelineStageFlags = 0x00080000; // VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
//const MS : vk::PipelineStageFlags = 0x00100000; // VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV
const TR: vk::PipelineStageFlags = vk::PipelineStageFlags::TRANSFER;

const I_DI: usize = 0;
const I_CS: usize = 1;
const I_VS: usize = 2;
const I_TCS: usize = 3;
const I_TES: usize = 4;
const I_GS: usize = 5;
const I_TF: usize = 6;
const I_FSR: usize = 7;
const I_EFT: usize = 8;
const I_FS: usize = 9;
const I_LFT: usize = 10;
const I_CAO: usize = 11;
const I_TR: usize = 12;


const STAGES_COUNT: usize = 13;

fn stage_index(flags: vk::PipelineStageFlags) -> usize {
    match flags {
        // draw & compute pipeline ordering
        DI => I_DI,
        CS => I_CS,
        VS => I_VS,
        TCS => I_TCS,
        TES => I_TES,
        GS => I_GS,
        TF => I_TF,
        FSR => I_FSR,
        EFT => I_EFT,
        FS => I_FS,
        LFT => I_LFT,
        CAO => I_CAO,
        TR => I_TR,
        _ => panic!("unexpected pipeline stage"),
    }
}

/// Returns whether stage a comes logically earlier than stage b.
#[rustfmt::skip]
fn is_logically_earlier(a: vk::PipelineStageFlags, b: vk::PipelineStageFlags) -> bool {

    const DI : vk::Flags64 = vk::PipelineStageFlags::DRAW_INDIRECT.as_raw() as u64;
    const II : vk::Flags64 = vk::PipelineStageFlags2KHR::INDEX_INPUT.as_raw() as u64;
    const VAI : vk::Flags64 = vk::PipelineStageFlags2KHR::VERTEX_ATTRIBUTE_INPUT.as_raw() as u64;
    const VS : vk::Flags64 = vk::PipelineStageFlags::VERTEX_SHADER.as_raw() as u64;
    const TCS : vk::Flags64 = vk::PipelineStageFlags::TESSELLATION_CONTROL_SHADER.as_raw() as u64;
    const TES : vk::Flags64 = vk::PipelineStageFlags::TESSELLATION_EVALUATION_SHADER.as_raw() as u64;
    const GS : vk::Flags64 = vk::PipelineStageFlags::GEOMETRY_SHADER.as_raw() as u64;
    const TF : vk::Flags64 = vk::PipelineStageFlags::TRANSFORM_FEEDBACK_EXT.as_raw() as u64;
    const FSR : vk::Flags64 = vk::PipelineStageFlags::FRAGMENT_SHADING_RATE_ATTACHMENT_KHR.as_raw() as u64;
    const EFT : vk::Flags64 = vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS.as_raw() as u64;
    const FS : vk::Flags64 = vk::PipelineStageFlags::FRAGMENT_SHADER.as_raw() as u64;
    const LFT : vk::Flags64 = vk::PipelineStageFlags::LATE_FRAGMENT_TESTS.as_raw() as u64;
    const CAO : vk::Flags64 = vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT.as_raw() as u64;
    const CS : vk::Flags64 = vk::PipelineStageFlags::COMPUTE_SHADER.as_raw() as u64;
    const TS : vk::Flags64 = 0x00080000; // VK_PIPELINE_STAGE_TASK_SHADER_BIT_NV
    const MS : vk::Flags64 = 0x00100000; // VK_PIPELINE_STAGE_MESH_SHADER_BIT_NV
    const TR : vk::Flags64 = vk::PipelineStageFlags::TRANSFER.as_raw() as u64;

    fn test(flags: vk::Flags64, mask: vk::Flags64) -> bool { (flags & mask) != 0 }

    let a = a.as_raw() as u64;
    let b = b.as_raw() as u64;

    match a {
        // draw & compute pipeline ordering
        DI  => test(b, DI | CS | TS | MS | II | VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        CS  => test(b,      CS                                                                             ),
        TS  => test(b,           TS | MS                                       | FSR | EFT | FS | LFT | CAO),
        MS  => test(b,                MS                                       | FSR | EFT | FS | LFT | CAO),
        II  => test(b,                     II | VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        VAI => test(b,                          VAI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        VS  => test(b,                                VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        TCS => test(b,                                     TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        TES => test(b,                                           TES | GS | TF | FSR | EFT | FS | LFT | CAO),
        GS  => test(b,                                                 GS | TF | FSR | EFT | FS | LFT | CAO),
        TF  => test(b,                                                      TF | FSR | EFT | FS | LFT | CAO),
        FSR => test(b,                                                           FSR | EFT | FS | LFT | CAO),
        EFT => test(b,                                                                 EFT | FS | LFT | CAO),
        FS  => test(b,                                                                       FS | LFT | CAO),
        LFT => test(b,                                                                            LFT | CAO),
        CAO => test(b,                                                                                  CAO),
        TR  => test(b, TR),
        _ => false,
    }
}

#[rustfmt::skip]
fn logically_later_stages(a: vk::PipelineStageFlags) -> &'static [usize] {
    match a {
        // draw & compute pipeline ordering
        DI  => & [I_DI , I_CS , I_VS , I_TCS , I_TES , I_GS , I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        CS  => & [       I_CS                                                                            ],
        VS  => & [              I_VS , I_TCS , I_TES , I_GS , I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        TCS => & [                     I_TCS , I_TES , I_GS , I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        TES => & [                             I_TES , I_GS , I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        GS  => & [                                     I_GS , I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        TF  => & [                                            I_TF , I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        FSR => & [                                                   I_FSR , I_EFT , I_FS , I_LFT , I_CAO],
        EFT => & [                                                           I_EFT , I_FS , I_LFT , I_CAO],
        FS  => & [                                                                   I_FS , I_LFT , I_CAO],
        LFT => & [                                                                          I_LFT , I_CAO],
        CAO => & [                                                                                  I_CAO],
        TR  => &[I_TR],
        _ => panic!("unexpected stage"),
    }
}


/// Per-stage sync tracking info.
///
/// Tells the source passes+stages that a particular destination stage is known to be synced with.
/// We track 3 execution sources on the current pipeline: draw sources (for the stages in the graphics pipeline),
/// compute (CS stage) and transfer (TR stage), plus 3 foreign sources (from other queues).
#[derive(Copy, Clone)]
struct PerStageTrackingInfo {
    /// Sync source on the draw pipeline.
    draw: (u64, vk::PipelineStageFlags),
    /// Sync source on the compute pipeline.
    compute: u64,
    transfer: u64,
    /// Sync sources from other queues.
    foreign: QueueSerialNumbers,
}

/// Execution dependency tracker.
#[derive(Copy,Clone)]
struct DependencyTracker {
    snn: SubmissionNumber,
    table: [PerStageTrackingInfo; STAGES_COUNT],
}

impl DependencyTracker {

    fn empty() -> DependencyTracker {
        DependencyTracker {
            snn: Default::default(),
            table: [PerStageTrackingInfo {
                draw: (0, vk::PipelineStageFlags::empty()),
                compute: 0,
                transfer: 0,
                foreign: Default::default()
            }; STAGES_COUNT]
        }
    }

    /// Creates a new dependency tracker from the last known state.
    pub fn new(last_known_state: &[DependencyTracker], this_snn: SubmissionNumber) -> DependencyTracker {
        let mut tracker = last_known_state.last().cloned().unwrap_or(DependencyTracker::empty());
        tracker.snn = this_snn;
        tracker
    } 
    
    /// Registers an execution dependency for the specified destination stage.
    ///
    /// Returns whether the execution dependency was already satisfied.
    pub fn add_execution_dependency(&mut self,
                                    prev_trackers: &[DependencyTracker],
                                    src_snn: SubmissionNumber,
                                    src_stage: vk::PipelineStageFlags,
                                    dst_stage: vk::PipelineStageFlags,
    ) -> bool
    {
        if src_snn.queue() != self.snn.queue() {
            // cross-queue dependency

        } else {

        }
        
        

        todo!()
    }

}


// 8+8+8+8 == 32b
// x20 stages => 640bytes per pass
// x1000 passes => 640kb of sync data
// complexity: O(number of dependencies)
// gives precise execution dependency information
// can be used to avoid redundant syncs

impl fmt::Debug for PerStageTrackingInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if self.draw.0 != 0 {
            write!(
                f,
                "{:>2}.{:<3}",
                self.draw.0,
                pipeline_stage_short_name(self.draw.1)
            )?;
        } else {
            write!(f, "      ")?;
        }
        if self.compute != 0 {
            write!(f, " ")?;
            write!(f, "{:>2}.CS", self.compute)?;
        } else {
            write!(f, "      ")?;
        }
        if self.transfer != 0 {
            write!(f, " ")?;
            write!(f, "{:>2}.TR", self.transfer)?;
        } else {
            write!(f, "      ")?;
        }
        Ok(())
    }
}

#[derive(Copy, Clone)]
struct PipelineSyncState {
}

impl PipelineSyncState {
    pub fn new() -> PipelineSyncState {
        PipelineSyncState {
            table: [PerStageTrackingInfo {
                draw: (0, vk::PipelineStageFlags::empty()),
                compute: 0,
                transfer: 0,
            }; STAGES_COUNT],
        }
    }
}

impl PipelineSyncState {
    fn add_execution_dependency(
        &mut self,
        prev_tables: &[PipelineSyncState],
        src: u64,
        src_stage: vk::PipelineStageFlags,
        dst_stage: vk::PipelineStageFlags,
    ) {
        for &i in logically_later_stages(dst_stage) {
            let stage = &mut self.table[i];
            let i_src = stage_index(src_stage);
            match src_stage {
                // syncing on any of these source stages => assume we're syncing on the
                DI | VS | TCS | TES | GS | TF | FSR | EFT | FS | LFT | CAO => {
                    // graphics pipeline
                    if src > stage.draw.0
                        && ((stage.draw.1 == vk::PipelineStageFlags::empty())
                            || is_logically_earlier(stage.draw.1, src_stage))
                    {
                        stage.draw = (src, src_stage);
                        stage.compute = stage
                            .compute
                            .max(prev_tables[src as usize].table[i_src].compute);
                        stage.transfer = stage
                            .transfer
                            .max(prev_tables[src as usize].table[i_src].transfer);
                    }
                }
                CS => {
                    // compute pipeline
                    if src > stage.compute {
                        stage.compute = src;
                        if prev_tables[src as usize].table[i_src].draw.0 > stage.draw.0
                            && ((stage.draw.1 == vk::PipelineStageFlags::empty())
                                || is_logically_earlier(
                            stage.draw.1,
                            prev_tables[src as usize].table[i_src].draw.1,
                                ))
                        {
                            stage.draw = prev_tables[src as usize].table[i_src].draw;
                        }
                        stage.transfer = stage
                            .transfer
                            .max(prev_tables[src as usize].table[i_src].transfer);
                    }
                }
                TR => {
                    // transfer pipeline
                    if src > stage.transfer {
                        stage.transfer = src;
                        stage.compute = stage
                            .compute
                            .max(prev_tables[src as usize].table[i_src].compute);
                        if prev_tables[src as usize].table[i_src].draw.0 > stage.draw.0
                            && ((stage.draw.1 == vk::PipelineStageFlags::empty())
                                || is_logically_earlier(
                            stage.draw.1,
                            prev_tables[src as usize].table[i_src].draw.1,
                                ))
                        {
                            stage.draw = prev_tables[src as usize].table[i_src].draw;
                        }
                    }
                }
                _ => panic!("unexpected pipeline stage"),
            }
        }
    }
}

impl fmt::Debug for PipelineSyncState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if f.alternate() {
            writeln!(f)?;
            for i in 0..STAGES_COUNT {
                write!(f, "|{:^20}", pipeline_stage_index_short_name(i))?;
            }
            writeln!(f)?;
        }
        for i in 0..STAGES_COUNT {
            let entry = &self.table[i];
            write!(f, "| {:?} ", entry)?;
        }
        Ok(())
    }
}

struct Pass {
    output_stage: vk::PipelineStageFlags,
    deps: Vec<(usize, vk::PipelineStageFlags)>,
}

impl Pass {
    fn new(output_stage: vk::PipelineStageFlags) -> Pass {
        Pass {
            output_stage,
            deps: vec![],
        }
    }
}

#[derive(Copy, Clone, Debug, Default)]
struct Dep(vk::PipelineStageFlags, vk::PipelineStageFlags);

#[derive(Clone)]
struct DepMatrix {
    num_passes: usize,
    matrix: Vec<Dep>,
}

impl DepMatrix {
    pub fn new(num_passes: usize) -> DepMatrix {
        let n = num_passes * num_passes;
        DepMatrix {
            num_passes,
            matrix: vec![Dep(EMPTY, EMPTY); n],
        }
    }

    pub fn add(
        &mut self,
        from: usize,
        src: vk::PipelineStageFlags,
        to: usize,
        dst: vk::PipelineStageFlags,
    ) {
        self.matrix[from * self.num_passes + to] = Dep(src, dst);
    }

    pub fn get(&self, src: usize, dst: usize) -> Dep {
        self.matrix[src * self.num_passes + dst]
    }

    pub fn get_mut(&mut self, src: usize, dst: usize) -> &mut Dep {
        &mut self.matrix[src * self.num_passes + dst]
    }

    pub fn propagate(&mut self) {
        for i in (1..self.num_passes).rev() {
            for j in i..self.num_passes {
                for m in i + 1..j {
                    let a = self.get(i, m);
                    let b = self.get(m, j);
                    if is_logically_earlier(a.1, b.0) {
                        self.add(i, a.0, j, b.1);
                    }
                }
            }
        }
    }
}

fn pipeline_stage_short_name(f: vk::PipelineStageFlags) -> &'static str {
    match f {
        EMPTY => ".",
        DI => "DI",
        VS => "VS",
        TCS => "TCS",
        TES => "TES",
        GS => "GS",
        TF => "TF",
        FSR => "FSR",
        EFT => "EFT",
        FS => "FS",
        LFT => "LFT",
        CAO => "CAO",
        CS => "CS",
        TR => "TR",
        _ => panic!("unsupported pipeline stage"),
    }
}
fn pipeline_stage_index_short_name(i: usize) -> &'static str {
    match i {
        //I_EMPTY => ".",
        I_DI => "DI",
        I_VS => "VS",
        I_TCS => "TCS",
        I_TES => "TES",
        I_GS => "GS",
        I_TF => "TF",
        I_FSR => "FSR",
        I_EFT => "EFT",
        I_FS => "FS",
        I_LFT => "LFT",
        I_CAO => "CAO",
        I_CS => "CS",
        I_TR => "TR",
        _ => panic!("unsupported pipeline stage"),
    }
}

impl fmt::Debug for DepMatrix {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        writeln!(f)?;
        // write column headers
        write!(f, "SRC↓DST>")?;
        for i in 0..self.num_passes {
            write!(f, "{:<8}", i)?;
        }
        writeln!(f)?;
        for i in 0..self.num_passes {
            write!(f, "{:>4} ", i)?;
            for j in 0..self.num_passes {
                let Dep(src, dst) = self.matrix[i * self.num_passes + j];
                let src = pipeline_stage_short_name(src);
                let dst = pipeline_stage_short_name(dst);
                write!(f, "{:>3}>{:<3} ", src, dst)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[bench]
fn bench_exec_dependencies_propagation(b: &mut Bencher) {
    let mut passes = Vec::new();

    passes.push(Pass::new(vk::PipelineStageFlags::empty()));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
    passes.push(Pass::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
    passes.push(Pass::new(vk::PipelineStageFlags::COMPUTE_SHADER));
    passes.push(Pass::new(vk::PipelineStageFlags::COMPUTE_SHADER));
    passes.push(Pass::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
    passes.push(Pass::new(vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    /*passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));
    passes.push(Pass::new(vk::PipelineStageFlags::TRANSFER));*/

    passes[2]
        .deps
        .push((1, vk::PipelineStageFlags::VERTEX_SHADER));
    passes[3]
        .deps
        .push((1, vk::PipelineStageFlags::FRAGMENT_SHADER));
    passes[4]
        .deps
        .push((2, vk::PipelineStageFlags::COMPUTE_SHADER));
    passes[5]
        .deps
        .push((3, vk::PipelineStageFlags::COMPUTE_SHADER));
    passes[5]
        .deps
        .push((4, vk::PipelineStageFlags::COMPUTE_SHADER));
    passes[6]
        .deps
        .push((4, vk::PipelineStageFlags::VERTEX_SHADER));
    passes[6]
        .deps
        .push((5, vk::PipelineStageFlags::FRAGMENT_SHADER));
    passes[7]
        .deps
        .push((4, vk::PipelineStageFlags::FRAGMENT_SHADER));
    passes[7]
        .deps
        .push((6, vk::PipelineStageFlags::FRAGMENT_SHADER));
    passes[8].deps.push((7, vk::PipelineStageFlags::TRANSFER));

    for i in 8..passes.len() - 1 {
        passes[i + 1]
            .deps
            .push((i, vk::PipelineStageFlags::TRANSFER));
    }

    let n = passes.len();
    let mut dm = DepMatrix::new(n);

    for (i, p) in passes.iter().enumerate() {
        for &(src, dst_stage) in p.deps.iter() {
            dm.add(src, passes[src].output_stage, i, dst_stage);
        }
    }

    let mut bench_dm = dm.clone();
    bench_dm.propagate();
    println!("propagated: {:?}", bench_dm);

    b.iter(move || {
        bench_dm.matrix[..].copy_from_slice(test::black_box(&dm.matrix[..]));
        bench_dm.propagate();
        bench_dm.get(4, 4)
    });

    //----------------------------------------------------------------------------------------------

    let mut tables = Vec::new();

    for (i, p) in passes.iter().enumerate() {
        let mut table = tables
            .last()
            .cloned()
            .unwrap_or(PipelineSyncState::new());
        for &(src, dst_stage) in p.deps.iter() {
            table.add_execution_dependency(
                &tables,
                src as u64,
                passes[src].output_stage,
                dst_stage,
            );
        }
        if i == 0 {
            println!("{:3} {:#?}", i, table);
        } else {
            println!("{:3} {:?}", i, table);
        }
        tables.push(table);
    }
}
