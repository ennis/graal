use winit::window::{WindowBuilder, Window};
use raw_window_handle::HasRawWindowHandle;
use winit::event_loop::EventLoop;
use graal::{vk, ImageId};

struct Fixture {
    context: graal::Context,
}

impl Fixture {
    pub fn new() -> Fixture {
        let device = graal::Device::new(None);
        let mut context = graal::Context::new(device);

        Fixture {
            context,
        }
    }
}

/// Helper to test frames.
/// Creates a test fixture and a frame, and runs the specified closure with the created frame.
fn frame_test(name: &str, f: impl FnOnce(&graal::Frame)) {
    let mut fixture = Fixture::new();
    let frame = fixture.context.start_frame(&graal::FrameCreateInfo {
        happens_after: None,
        collect_debug_info: true
    });
    f(&frame);

    frame.dump(Some(name));
    frame.finish();
}

/// Creates a dummy graphics pass. For testing automatic synchronization.
fn add_dummy_graphics_pass(frame: &graal::Frame,
                           name: &str,
                           image_accesses: &[(graal::ImageId, vk::AccessFlags, vk::PipelineStageFlags, vk::ImageLayout)],
                           buffer_accesses: &[(graal::BufferId, vk::AccessFlags, vk::PipelineStageFlags)])
{
    frame.add_graphics_pass(name, |pass| {
        for &(img, access_mask, stage_mask, layout) in image_accesses {
            pass.register_image_access_2(img, access_mask, stage_mask,  layout);
        }
        for &(buf, access_mask, stage_mask) in buffer_accesses {
            pass.register_buffer_access_2(buf, access_mask, stage_mask);
        }
    });
}

fn create_dummy_transient_image(frame: &graal::Frame, name: &str) -> ImageId {
    let graal::ImageInfo { id, .. } = frame.context().create_image(
        name,
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::ImageResourceCreateInfo {
            image_type: graal::vk::ImageType::TYPE_2D,
            usage: graal::vk::ImageUsageFlags::COLOR_ATTACHMENT
                | graal::vk::ImageUsageFlags::SAMPLED
                | graal::vk::ImageUsageFlags::TRANSFER_DST,
            format: graal::vk::Format::R8G8B8A8_SRGB,
            extent: graal::vk::Extent3D {
                width: 1280,
                height: 720,
                depth: 1,
            },
            mip_levels: 1,
            array_layers: 1,
            samples: 1,
            tiling: graal::vk::ImageTiling::OPTIMAL,
        },
        true,
    );
    id
}

macro_rules! test_image {
    ($f:expr, $n:ident) => { let $n = create_dummy_transient_image($f, stringify!($n)); };
}

macro_rules! test_graphics_pass {
    ($f:expr, $n:ident,
        IMAGE
            $([$img:ident, $($img_access:ident)|+, $($img_stages:ident)|+, $img_layout:ident])*
        BUFFER
            $([$buf:ident, $($buf_access:ident)|+, $($buf_stages:ident)|+])*
    ) => {
        add_dummy_graphics_pass($f, stringify!($n), &[
            $( ($img, $(graal::vk::AccessFlags::$img_access)|+, $(graal::vk::PipelineStageFlags::$img_stages)|+, graal::vk::ImageLayout::$img_layout) ),*
        ],&[
            $( ($buf, $(graal::vk::AccessFlags::$buf_access)|+, $(graal::vk::PipelineStageFlags::$buf_stages)|+) ),*
        ]);
    };
}


#[test]
fn test_pipeline_barrier() {
    frame_test("pipeline_barrier", |frame| {

        test_image!(frame, a);
        test_image!(frame, b);
        test_image!(frame, c);

        test_graphics_pass!(frame, p1,
            IMAGE
                [a, COLOR_ATTACHMENT_WRITE, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_OPTIMAL]
            BUFFER
        );

        test_graphics_pass!(frame, p2,
            IMAGE
                [a, SHADER_READ,            FRAGMENT_SHADER,         SHADER_READ_ONLY_OPTIMAL]
                [b, COLOR_ATTACHMENT_WRITE, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_OPTIMAL]
            BUFFER
        );

        test_graphics_pass!(frame, p3,
            IMAGE
                [b, SHADER_READ,            FRAGMENT_SHADER,         SHADER_READ_ONLY_OPTIMAL]
                [c, COLOR_ATTACHMENT_WRITE, COLOR_ATTACHMENT_OUTPUT, COLOR_ATTACHMENT_OPTIMAL]
            BUFFER
        );
    });
}