//! Scene management
use crate::{
    bounding_box::BoundingBox,
    mesh::{MeshData, Vertex3D},
};
use glam::{vec2, vec3, Mat4, Vec3, Vec3A, Vec4};
use graal::{ash::version::DeviceV1_0, vk, TypedBufferInfo};
use slotmap::{new_key_type, SlotMap};
use std::{mem, path::Path, ptr};
use std::sync::{Arc, RwLock};
use tobj::Mesh;

new_key_type! {
    pub struct ObjectId;
    pub struct MaterialId;
    pub struct MeshId;
}

/// Information about the material of an object in a scene.
/// TODO for now it's just a few hard-coded properties but we might want a more sophisticated
/// material system in the future.
pub struct MaterialData {
    pub color: Vec4,
}

/// Represents a transform.
pub struct Transform {
    pub translate: Vec3,
    pub rotate: Vec3,
    pub scale: Vec3,
}

impl Transform {

    const IDENTITY: Transform  = Transform {
        translate: glam::const_vec3!([0.0, 0.0, 0.0]),
        rotate: glam::const_vec3!([0.0, 0.0, 0.0]),
        scale: glam::const_vec3!([1.0, 1.0, 1.0]),
    };

    /// Returns the identity transform.
    pub fn identity() -> Transform {
        Self::IDENTITY
    }

    /// Returns the matrix corresponding to the transform.
    pub fn to_mat4(&self) -> Mat4 {
        todo!()
    }
}

impl Default for Transform {
    fn default() -> Self {
        Transform::identity()
    }
}

/// An object in a scene.
pub struct ObjectData {
    /// Object transform.
    pub transform: Transform,
    /// Associated material, can be null.
    pub material: MaterialId,
    /// Mesh
    pub mesh: MeshId,
}

type ObjectMap = SlotMap<ObjectId, ObjectData>;
type MaterialMap = SlotMap<MaterialId, MaterialData>;
type MeshMap = SlotMap<MeshId, MeshData>;

/// Contains a hierarchy of objects, with associated transforms, and that reference mesh data
/// and material information.
// TODO actual hierarchy
pub struct Scene {
    //context: Arc<RwLock<graal::Context>>,
    objects: ObjectMap,
    materials: MaterialMap,
    meshes: MeshMap,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new(/*context: Arc<RwLock<graal::Context>>*/) -> Scene {
        Scene {
            //context,
            objects: SlotMap::with_key(),
            materials: SlotMap::with_key(),
            meshes: SlotMap::with_key(),
        }
    }

    pub fn objects(&self) -> &ObjectMap {
        &self.objects
    }

    pub fn materials(&self) -> &MaterialMap {
        &self.materials
    }

    pub fn meshes(&self) -> &MeshMap {
        &self.meshes
    }

    pub fn mesh(&self, id: MeshId) -> Option<&MeshData> {
        self.meshes.get(id)
    }

    /// Imports objects from an obj file.
    /// Vertex data will be uploaded to the GPU as a part of the specified frame.
    pub fn import_obj(&mut self, frame: &graal::Frame, obj_file_path: &Path) {
        let (models, materials) = match tobj::load_obj(obj_file_path, true) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("error loading OBJ file: {}", e);
                return;
            }
        };

        for m in models.iter() {
            let mesh_data = load_mesh_from_obj(frame, m);

            eprintln!("loaded object {}, {} vertices, {} indices", m.name, mesh_data.vertex_count, mesh_data.index_count);

            let mesh_id = self.meshes.insert(mesh_data);
            self.objects.insert(ObjectData {
                transform: Transform::identity(),
                material: Default::default(),
                mesh: mesh_id,
            });
        }
    }
}

fn load_mesh_from_obj(frame: &graal::Frame, model: &tobj::Model) -> MeshData {
    let mesh = &model.mesh;
    // allocate staging buffers in advance
    let vertex_count = mesh.positions.len() / 3;
    let index_count = mesh.indices.len();
    let vertex_byte_size = mem::size_of::<Vertex3D>() * vertex_count;
    let index_byte_size = mem::size_of::<u32>() * index_count;

    let staging_vbo = frame.alloc_upload_slice::<Vertex3D>(
        vk::BufferUsageFlags::TRANSFER_SRC,
        vertex_count,
        Some("staging vertex buffer"),
    );
    let staging_ibo = frame.alloc_upload_slice::<u32>(
        vk::BufferUsageFlags::TRANSFER_SRC,
        index_count,
        Some("staging index buffer"),
    );

    let mut p_min = Vec3A::splat(f32::INFINITY);
    let mut p_max = Vec3A::splat(-f32::INFINITY);

    for i in 0..vertex_count {
        let position = vec3(
            mesh.positions[i * 3],
            mesh.positions[i * 3 + 1],
            mesh.positions[i * 3 + 2],
        );

        // update bounds
        p_min = p_min.min(position.into());
        p_max = p_max.max(position.into());

        let normal = if !mesh.normals.is_empty() {
            vec3(
                mesh.normals[i * 3],
                mesh.normals[i * 3 + 1],
                mesh.normals[i * 3 + 2],
            )
        } else {
            vec3(0.0f32, 0.0f32, 0.0f32)
        };

        let texcoords = if !mesh.texcoords.is_empty() {
            vec2(mesh.texcoords[i * 2], mesh.texcoords[i * 2 + 1])
        } else {
            vec2(0.0f32, 0.0f32)
        };

        let v = Vertex3D {
            position,
            normal,
            tangent: normal,
        };

        unsafe {
            ptr::write((staging_vbo.mapped_ptr as *mut Vertex3D).add(i), v);
        }
    }

    for (i, &index) in mesh.indices.iter().enumerate() {
        unsafe {
            ptr::write((staging_ibo.mapped_ptr as *mut u32).add(i), index);
        }
    }

    // create the device local vertex/index buffers
    let device_vbo = frame.context().create_buffer(
        &model.name,
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            byte_size: vertex_byte_size as u64,
            map_on_create: false,
        },
        false,
    );

    let device_ibo = frame.context().create_buffer(
        &model.name,
        &graal::ResourceMemoryInfo::DEVICE_LOCAL,
        &graal::BufferResourceCreateInfo {
            usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
            byte_size: index_byte_size as u64,
            map_on_create: false,
        },
        false,
    );

    // upload
    frame.add_transfer_pass("upload mesh", false, |pass| {

        pass.register_buffer_access(staging_vbo.id, graal::AccessType::TransferRead);
        pass.register_buffer_access(device_vbo.id, graal::AccessType::TransferWrite);
        pass.register_buffer_access(staging_ibo.id, graal::AccessType::TransferRead);
        pass.register_buffer_access(device_ibo.id, graal::AccessType::TransferWrite);

        //pass.flush_buffer(device_vbo, vk::AccessFlags::MEMORY_READ);
        //pass.flush_buffer(device_ibo, vk::AccessFlags::MEMORY_READ);

        pass.set_commands(move |context, command_buffer| unsafe {
            let device = context.device();

            device.cmd_copy_buffer(
                command_buffer,
                staging_vbo.handle,
                device_vbo.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: vertex_byte_size as u64,
                }],
            );

            device.cmd_copy_buffer(
                command_buffer,
                staging_ibo.handle,
                device_ibo.handle,
                &[vk::BufferCopy {
                    src_offset: 0,
                    dst_offset: 0,
                    size: index_byte_size as u64,
                }],
            );
        });
    });

    MeshData {
        vertex_buffer: graal::TypedBufferInfo {
            id: device_vbo.id,
            handle: device_vbo.handle,
            mapped_ptr: ptr::null_mut(),
        },
        index_buffer: TypedBufferInfo {
            id: device_ibo.id,
            handle: device_ibo.handle,
            mapped_ptr: ptr::null_mut(),
        },
        index_count,
        vertex_count,
        bounds: BoundingBox {
            min: p_min,
            max: p_max,
        },
    }
}

/*impl Default for Scene {
    fn default() -> Self {
        Scene::new()
    }
}*/
