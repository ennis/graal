use graal::{platform::windows::DeviceExtWindows, vk, ImageResourceCreateInfo};
use std::ptr;

use windows::{
    core::Interface,
    Win32::{
        Foundation::HANDLE,
        Graphics::{
            Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_FEATURE_LEVEL, D3D_FEATURE_LEVEL_11_1},
            Direct3D11::{
                D3D11CreateDevice, ID3D11Device5, ID3D11Texture2D, D3D11_BIND_RENDER_TARGET, D3D11_CPU_ACCESS_FLAG,
                D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_CREATE_DEVICE_DEBUG, D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
                D3D11_RESOURCE_MISC_SHARED_NTHANDLE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT,
            },
            Dxgi::{
                Common::{DXGI_FORMAT_R8G8B8A8_UNORM, DXGI_SAMPLE_DESC},
                IDXGIResource1, DXGI_SHARED_RESOURCE_READ, DXGI_SHARED_RESOURCE_WRITE,
            },
        },
    },
};

#[test]
fn test_win32_share() {
    let (device, context) = unsafe { graal::create_device_and_context(None) };

    // create D3D11 device
    let (d3d11_device, _): (ID3D11Device5, _) = unsafe {
        let mut d3d11_device = None;
        let mut feature_level = D3D_FEATURE_LEVEL::default();
        let mut _d3d11_device_context = None;

        let feature_levels = [D3D_FEATURE_LEVEL_11_1];

        D3D11CreateDevice(
            // pAdapter:
            None,
            // DriverType:
            D3D_DRIVER_TYPE_HARDWARE,
            // Software:
            None,
            // Flags:
            D3D11_CREATE_DEVICE_BGRA_SUPPORT | D3D11_CREATE_DEVICE_DEBUG,
            // pFeatureLevels:
            &feature_levels,
            // SDKVersion
            D3D11_SDK_VERSION,
            // ppDevice:
            &mut d3d11_device,
            // pFeatureLevel:
            &mut feature_level,
            // ppImmediateContext:
            &mut _d3d11_device_context,
        )
        .unwrap();

        tracing::info!("Direct3D feature level: {:?}", feature_level);

        (
            d3d11_device.unwrap().cast::<ID3D11Device5>().unwrap(),
            _d3d11_device_context.unwrap(),
        )
    };

    // create D3D11 texture
    unsafe {
        let width = 512;
        let height = 512;
        let d3d11_texture_desc = D3D11_TEXTURE2D_DESC {
            Width: width,
            Height: height,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_R8G8B8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: D3D11_BIND_RENDER_TARGET,
            CPUAccessFlags: D3D11_CPU_ACCESS_FLAG(0),
            MiscFlags: D3D11_RESOURCE_MISC_SHARED_NTHANDLE | D3D11_RESOURCE_MISC_SHARED_KEYEDMUTEX,
        };

        let d3d11_texture: ID3D11Texture2D = d3d11_device
            .CreateTexture2D(&d3d11_texture_desc, ptr::null())
            .expect("CreateTexture2D failed");
        // create shared handle
        let dxgi_resource: IDXGIResource1 = d3d11_texture.cast::<IDXGIResource1>().unwrap();

        let mut handle: HANDLE = dxgi_resource
            .CreateSharedHandle(
                ptr::null(),
                DXGI_SHARED_RESOURCE_READ | DXGI_SHARED_RESOURCE_WRITE,
                None,
            )
            .expect("CreateSharedHandle failed");

        let imported_image = device.create_imported_image_win32(
            "image",
            &graal::ImageResourceCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                usage: vk::ImageUsageFlags::SAMPLED
                    | vk::ImageUsageFlags::TRANSFER_DST
                    | vk::ImageUsageFlags::COLOR_ATTACHMENT,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D {
                    width,
                    height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: 1,
                tiling: Default::default(),
            },
            vk::MemoryPropertyFlags::default(),
            vk::MemoryPropertyFlags::default(),
            vk::ExternalMemoryHandleTypeFlags::D3D11_TEXTURE,
            handle.0 as vk::HANDLE,
            None,
        );
    }
}
