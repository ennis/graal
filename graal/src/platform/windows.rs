use ash::vk::HANDLE;


#[derive(Copy, Clone, Debug)]
pub struct Win32Handle<'a> {
    pub handle: HANDLE,
    pub name: &'a str,
}