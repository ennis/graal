use ash::vk;
use graal::{ash::vk::PipelineStageFlags, SubmissionNumber};
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

/// Returns whether stage a comes logically earlier than stage b.
#[rustfmt::skip]
fn logically_earlier(a: vk::PipelineStageFlags, b: vk::PipelineStageFlags) -> bool {

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
        // transfer
        TR  => test(b, TR),
        _ => false,
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
                    if logically_earlier(a.1, b.0) {
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

fn main() {
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

    let n = passes.len();
    let mut dm = DepMatrix::new(n);

    for (i, p) in passes.iter().enumerate() {
        for &(src, dst_stage) in p.deps.iter() {
            dm.add(src, passes[src].output_stage, i, dst_stage);
        }
    }

    eprintln!("raw: {:#?}", dm);
    dm.propagate();
    eprintln!("propagated: {:#?}", dm);
}
