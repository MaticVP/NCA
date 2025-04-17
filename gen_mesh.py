import numpy as np
import trimesh
from PIL import Image
from scipy.ndimage import gaussian_filter

def gen_map(img):
    image_path = './fbm_noise_erosion.png'  # Path to your image
    resize_shape = (128, 128)  # Resize for performance & smoother mesh
    pixel_size = 20.0  # Horizontal spacing between points
    height_scale = 1.0  # Elevation scaling factor
    smoothing_sigma = 2.0  # Gaussian blur strength

    # === LOAD IMAGE ===
    img = Image.open(img).convert('L')  # Grayscale
    img = img.resize(resize_shape, resample=Image.BILINEAR)
    height_map = np.array(img).astype(np.float32)

    # === NORMALIZE AND SCALE HEIGHT ===
    smoothed_map = np.clip(height_map, np.percentile(height_map, 5), np.percentile(height_map, 95))
    smoothed_map -= smoothed_map.min()
    smoothed_map /= smoothed_map.max()
    z_flat = (height_map.flatten()) * height_scale

    # === GENERATE GRID COORDINATES ===
    h, w = height_map.shape
    x = np.arange(w) * pixel_size
    y = np.arange(h) * pixel_size
    x_grid, y_grid = np.meshgrid(x, y)
    x_flat = x_grid.flatten()
    y_flat = y_grid.flatten()

    # === CREATE VERTICES AND FACES ===
    vertices = np.vstack((x_flat, y_flat, z_flat)).T
    faces = []
    for i in range(h - 1):
        for j in range(w - 1):
            v0 = i * w + j
            v1 = v0 + 1
            v2 = v0 + w
            v3 = v2 + 1
            faces.append([v0, v1, v2])
            faces.append([v1, v3, v2])
    faces = np.array(faces)

    # === COLORING BASED ON HEIGHT ===
    # Normalize the height values to [0, 1] for colormap usage
    norm_height = (height_map.flatten() - height_map.min()) / (height_map.max() - height_map.min())

    # Create custom color map: blue to red based on height
    face_colors = np.zeros((len(faces), 3))  # RGB color array for faces
    for i, face in enumerate(faces):
        height_value = np.mean(height_map[face // w, face % w])  # Average height for the face
        norm_value = (height_value - height_map.min()) / (height_map.max() - height_map.min())
        if norm_value > 0.1:
            face_colors[i] = [0.58, 0.878, 0.588]
        if norm_value > 0.5:
            face_colors[i] = [0.78,0.678, 0.416]
        if norm_value > 0.8:
            face_colors[i] = [0.902,0.882, 0.843]
        #face_colors[i] = [norm_value, 0, 1 - norm_value]  # Blue to Red gradient

    # === CREATE AND EXPORT MESH ===
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    # Assign the custom colors to the mesh (per face, not per vertex)
    mesh.visual.face_colors = (face_colors * 255).astype(np.uint8)  # Trimesh expects [0, 255]

    return mesh, face_colors