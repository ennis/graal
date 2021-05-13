//! Scene management
use crate::{
    bounding_box::BoundingBox,
    mesh::{MeshData, Vertex3D},
};
use glam::{vec2, vec3, Mat4, Vec3, Vec3A, Vec4};
use graal::{ash::version::DeviceV1_0, vk, GpuFuture, TypedBufferInfo};
use slotmap::{new_key_type, SlotMap};
use std::{mem, path::Path, ptr};

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
    const IDENTITY: Transform = Transform {
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
pub struct Scene {
    objects: ObjectMap,
    materials: MaterialMap,
    meshes: MeshMap,
    last_upload: GpuFuture,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new(/*context: Arc<RwLock<graal::Context>>*/) -> Scene {
        Scene {
            //context,
            objects: SlotMap::with_key(),
            materials: SlotMap::with_key(),
            meshes: SlotMap::with_key(),
            last_upload: Default::default(),
        }
    }

    /// Returns the bounds of the scene.
    ///
    /// This function returns the union of the bounds of all objects in the scene.
    pub fn bounds(&self) -> BoundingBox {
        // TODO handle transforms
        let mut b = BoundingBox::new();
        for m in self.meshes.values() {
            b = b.union(&m.bounds);
        }
        b
    }

    /// Returns the objects in the scene.
    ///
    /// Each object may in turn reference an associated mesh, a materials,
    /// and child objects.
    ///
    /// ## Example
    /// TODO
    pub fn objects(&self) -> &ObjectMap {
        &self.objects
    }

    /// Returns the materials of the scene.
    pub fn materials(&self) -> &MaterialMap {
        &self.materials
    }

    /// Returns the meshes of the scene.
    pub fn meshes(&self) -> &MeshMap {
        &self.meshes
    }

    /// Returns the mesh with the specified ID. Equivalent to `self.meshes.get(id)`
    pub fn mesh(&self, id: MeshId) -> Option<&MeshData> {
        self.meshes.get(id)
    }

    pub fn start_upload<'a>(&'a mut self, context: &'a mut graal::Context) -> SceneUploader<'a> {
        SceneUploader {
            scene: self,
            frame: context.start_frame(graal::FrameCreateInfo {
                happens_after: Default::default(),
                collect_debug_info: true,
            }),
            upload_items: vec![],
        }
    }
}

/// An item for an upload operation (copy from CPU staging to GPU buffer).
struct SceneUploadItem {
    staging_buffer: graal::BufferInfo,
    device_buffer: graal::BufferInfo,
    byte_size: usize,
}

pub struct SceneUploader<'a> {
    scene: &'a mut Scene,
    frame: graal::Frame<'a>,
    upload_items: Vec<SceneUploadItem>,
}

impl<'a> SceneUploader<'a> {
    /// Imports objects and meshes from an obj file.
    pub fn import_obj<P: AsRef<Path>>(&mut self, obj_file_path: P) {
        self.import_obj_internal(obj_file_path.as_ref());
    }

    fn load_mesh_from_obj(&mut self, model: &tobj::Model) -> MeshId {
        // Loading mesh data on the GPU is done in two steps:
        //
        // - first, in this function (`load_mesh_from_obj`), we allocate CPU staging buffers
        //   for vertex and index data, and load data from the file onto those buffers.
        //   We also allocate the vertex and index buffers on the GPU, but we don't transfer the mesh
        //   data to them yet. Instead, we add a `SceneUploadItem` for both vertex and index buffers
        //   to `self.upload_items`, which will be processed later.
        //
        // - then, in `finish`, we create the transfer pass that performs the transfer from
        //   staging to device buffer for each `SceneUploadItem`.

        let mesh = &model.mesh;
        let vertex_count = mesh.positions.len() / 3;
        let index_count = mesh.indices.len();
        let vertex_byte_size = mem::size_of::<Vertex3D>() * vertex_count;
        let index_byte_size = mem::size_of::<u32>() * index_count;

        // Allocate staging buffers.
        let staging_vbo = self.frame.alloc_upload_slice::<Vertex3D>(
            vk::BufferUsageFlags::TRANSFER_SRC,
            vertex_count,
            Some("staging vertex buffer"),
        );
        let staging_ibo = self.frame.alloc_upload_slice::<u32>(
            vk::BufferUsageFlags::TRANSFER_SRC,
            index_count,
            Some("staging index buffer"),
        );

        // Compute bounds, reformat vertex attributes a little bit, and write them to the staging buffer.
        let mut p_min = Vec3A::splat(f32::INFINITY);
        let mut p_max = Vec3A::splat(-f32::INFINITY);

        for i in 0..vertex_count {
            let position = vec3(
                mesh.positions[i * 3],
                mesh.positions[i * 3 + 1],
                mesh.positions[i * 3 + 2],
            );

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

        // Copy indices to the staging index buffer.
        for (i, &index) in mesh.indices.iter().enumerate() {
            unsafe {
                ptr::write((staging_ibo.mapped_ptr as *mut u32).add(i), index);
            }
        }

        // Create the device local vertex/index buffers.
        let device_vbo = self.frame.context().create_buffer(
            &model.name,
            &graal::ResourceMemoryInfo::DEVICE_LOCAL,
            &graal::BufferResourceCreateInfo {
                usage: vk::BufferUsageFlags::VERTEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                byte_size: vertex_byte_size as u64,
                map_on_create: false,
            },
            /* transient */ false,
        );

        let device_ibo = self.frame.context().create_buffer(
            &model.name,
            &graal::ResourceMemoryInfo::DEVICE_LOCAL,
            &graal::BufferResourceCreateInfo {
                usage: vk::BufferUsageFlags::INDEX_BUFFER | vk::BufferUsageFlags::TRANSFER_DST,
                byte_size: index_byte_size as u64,
                map_on_create: false,
            },
            /* transient */ false,
        );

        // Push upload items for the vertex and index buffers.
        self.upload_items.push(SceneUploadItem {
            staging_buffer: staging_vbo.into(),
            device_buffer: device_vbo.into(),
            byte_size: vertex_byte_size,
        });
        self.upload_items.push(SceneUploadItem {
            staging_buffer: staging_ibo.into(),
            device_buffer: device_ibo.into(),
            byte_size: index_byte_size,
        });

        // Build and insert the mesh data entry in the scene.
        let mesh_data = MeshData {
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
        };

        eprintln!(
            "loaded mesh {}, {} vertices, {} indices",
            model.name, mesh_data.vertex_count, mesh_data.index_count
        );

        let mesh_id = self.scene.meshes.insert(mesh_data);
        mesh_id
    }

    /// Non-polymorphic version of the above.
    fn import_obj_internal(&mut self, obj_file_path: &Path) {
        let (models, _materials) = match tobj::load_obj(obj_file_path, true) {
            Ok(x) => x,
            Err(e) => {
                eprintln!("error loading OBJ file: {}", e);
                return;
            }
        };

        // Load all meshes in the OBJ file and create an object for each of them.
        for m in models.iter() {
            let mesh_id = self.load_mesh_from_obj(m);

            self.scene.objects.insert(ObjectData {
                transform: Transform::identity(),
                material: Default::default(),
                mesh: mesh_id,
            });
        }
    }

    /// Finishes uploading stuff to the scene.
    pub fn finish(self) {
        let upload_items = self.upload_items;

        self.frame
            .add_transfer_pass("upload scene data", false, |pass| {
                // register accesses
                for item in upload_items.iter() {
                    // Even though we don't need synchronization here, we still need to we use coarse-grained (per-frame) synchronization for scene uploads.
                    pass.register_buffer_access_2(
                        item.staging_buffer.id,
                        vk::AccessFlags::TRANSFER_READ,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                    pass.register_buffer_access_2(
                        item.device_buffer.id,
                        vk::AccessFlags::TRANSFER_WRITE,
                        vk::PipelineStageFlags::TRANSFER,
                    );
                }

                pass.set_commands(move |context, command_buffer| unsafe {
                    for item in upload_items {
                        context.vulkan_device().cmd_copy_buffer(
                            command_buffer,
                            item.staging_buffer.handle,
                            item.device_buffer.handle,
                            &[vk::BufferCopy {
                                src_offset: 0,
                                dst_offset: 0,
                                size: item.byte_size as u64,
                            }],
                        );
                    }
                });
            });

        self.frame.dump(Some("scene_upload"));
        // Finish the frame.
        let future = self.frame.finish();
        // We now need to wait for the frame to complete before being able to access the
        // GPU mesh buffers. Update the future accordingly.
        self.scene.last_upload = self.scene.last_upload.join(future);
    }
}
