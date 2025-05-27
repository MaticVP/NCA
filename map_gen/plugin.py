import bpy
import numpy as np
import math
from mathutils import Vector
from bpy import context

bl_info = {
    "name": "3D Mesh from Height Map",
    "blender": (2, 9, 0),
    "category": "Object",
}


import numpy as np
import torch
import torchvision.transforms.functional as TF

from CellularAutomata import CA, to_rgb, NoiseCA, FullCA

def gen_hightmap(type, start_seed=None,numSteps=10,steps=12, res=64, seed_size=8):

    if type == "Perlin" or type == "FBM":
        ca = CA()
    elif type == "Noise FBM" or type == "Noise Perlin" or type == "Chunk Perlin":
        ca = NoiseCA(noise_level=0.2)
    else:
        ca = FullCA(noise_level=0.2)

    if type == "Perlin":
        ca.load_state_dict(torch.load("./ca_model_pearl_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "FBM":
        ca.load_state_dict(torch.load("./ca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise Perlin":
        ca.load_state_dict(torch.load("./nca_model_perlin_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise FBM":
        ca.load_state_dict(torch.load("./nca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Full Perlin":
        ca.load_state_dict(torch.load("./fca_model_perlin_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Full FBM":
        ca.load_state_dict(torch.load("./fca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Chunk Perlin":
        ca.load_state_dict(torch.load("./fca_model_perlin_ero_chunks.pt", weights_only=True, map_location=torch.device('cpu')))


    if start_seed is None:
        x = ca.seed(1, res)
    else:
        x = ca.seed(1, res)
        ref_x, ref_y, ref_xy = start_seed

        def apply_ref(ref, h_slice, w_slice):
            if ref is not None:
                ref = torch.from_numpy(ref.transpose(2, 0, 1))
                x[0, :3, h_slice, w_slice] = ref[:, h_slice, w_slice]

        chunk_size = seed_size

        apply_ref(ref_x, slice(0, res), slice(0, chunk_size))

        apply_ref(ref_y, slice(0, chunk_size), slice(0, res))

        apply_ref(ref_xy, slice(0, chunk_size), slice(0, chunk_size))


    with torch.no_grad():
        for i in range(numSteps):

            for k in range(steps):

                x[:] = ca(x)

        #x = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        x = to_rgb(x).permute([0, 2, 3, 1])

        x = x.squeeze(0)

        x = x.detach().cpu().numpy()

        x = np.clip(x, 0, 1)

        return x



class HeightMapToMeshOperator(bpy.types.Operator):
    bl_idname = "object.height_map_to_mesh_operator"
    bl_label = "Generate 3D Mesh from Height Map"

    def execute(self, context):
        # Load the height map image (grayscale)
        height_map_image = gen_hightmap("Chunk Perlin", numSteps=14,steps=5, seed_size=32)

        # Get image size (width and height)
        width, height = height_map_image.size

        # Create a new mesh (a plane) to apply height data
        mesh_data = bpy.data.meshes.new("height_map_mesh")
        obj = bpy.data.objects.new("HeightMapObject", mesh_data)
        bpy.context.collection.objects.link(obj)

        # Create vertices
        vertices = []
        for y in range(height):
            for x in range(width):
                # Get pixel value from the height map (in grayscale)
                pixel = height_map_image.pixels[(y * width + x) * 4]  # Only use the Red channel (grayscale)

                # Convert pixel to a height (scale the value)
                z = pixel * 10  # You can adjust the scale factor (10) to fit your needs

                # Create a vertex at (x, y, z)
                vertices.append((x, y, z))

        # Create faces (quad-based mesh)
        faces = []
        for y in range(height - 1):
            for x in range(width - 1):
                # Each quad is defined by 4 corners (vertices)
                v1 = y * width + x
                v2 = y * width + (x + 1)
                v3 = (y + 1) * width + (x + 1)
                v4 = (y + 1) * width + x
                faces.append((v1, v2, v3, v4))

        # Assign vertices and faces to the mesh
        mesh_data.from_pydata(vertices, [], faces)

        # Update the mesh to reflect changes
        mesh_data.update()

        # Optionally smooth the shading of the mesh
        bpy.ops.object.shade_smooth()

        return {'FINISHED'}


class HeightMapPanel(bpy.types.Panel):
    bl_label = "Height Map to 3D Mesh"
    bl_idname = "OBJECT_PT_height_map"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Tool'

    def draw(self, context):
        layout = self.layout
        layout.operator("object.height_map_to_mesh_operator")  # Button to run the operator


def register():
    bpy.utils.register_class(HeightMapToMeshOperator)
    bpy.utils.register_class(HeightMapPanel)


def unregister():
    bpy.utils.unregister_class(HeightMapToMeshOperator)
    bpy.utils.unregister_class(HeightMapPanel)


if __name__ == "__main__":
    register()
