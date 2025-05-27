import io
import subprocess
import time
from scipy.ndimage import median_filter
import streamlit as st
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from stpyvista import stpyvista
from gen_heightmaps import gen_hightmap
from gen_mesh import gen_map

torch.classes.__path__ = []

# is_xvfb_running = subprocess.run(["pgrep", "Xvfb"], capture_output=True)
# if is_xvfb_running.returncode == 1:
#     pv.start_xvfb()

pv.start_xvfb()

st.title("Upload Image & Render Heightmap")

uploaded_file = st.file_uploader("Upload a heightmap image (.png, .jpg)", type=["png", "jpg", "jpeg"])

st.sidebar.header("Mesh Parameters")
resolution = st.sidebar.slider("Resolution (number of samples)", min_value=64, max_value=128, value=128, step=1)
pixel_scale = st.sidebar.number_input("Pixel Scale (e.g., meters per pixel)", min_value=1, max_value=50, value=20, step=1)
height_scale = st.sidebar.number_input("Height Scale", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
st.sidebar.header("NCA Parameters")
numSteps = st.sidebar.slider("number of steps", min_value=1, max_value=300, value=30, step=1)
steps = st.sidebar.number_input("Iterations per step", min_value=1, max_value=96, value=32, step=1)
res = st.sidebar.slider("Resolution (number of samples)", min_value=32, max_value=256, value=32, step=1)
map_type = st.sidebar.selectbox("Map Type", ["Perlin", "FBM","Noise Perlin","Noise FBM","Full Perlin","Full FBM"])



img = None

elapsed_map = 0.0
elapsed_mesh = 0.0

if st.sidebar.button("Generate Heightmap"):

    start = time.time()
    uploaded_file = gen_hightmap(map_type,numSteps,steps,res)

    img = uploaded_file
    img *= 255

    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.convert('L')

    elapsed_map = time.time() - start

elif uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')

if uploaded_file is not None:
    st.session_state.hmap = img

if uploaded_file is not None or "hmap" in st.session_state.keys():
    buf = io.BytesIO()
    st.session_state.hmap.save(buf, format="PNG")
    buf.seek(0)

    st.sidebar.download_button(
        label="Download Heightmap as PNG",
        data=buf,
        file_name="heightmap.png",
        mime="image/png"
    )





if uploaded_file is not None or "hmap" in st.session_state.keys():
    start = time.time()
    mesh, color = gen_map(st.session_state.hmap, (resolution,resolution),pixel_scale,height_scale)

    # Convert Trimesh to PyVista
    vertices = mesh.vertices
    faces = mesh.faces
    faces_pv = np.hstack([[3] + list(face) for face in faces])  # add face sizes (3 = triangle)
    faces_pv = np.array(faces_pv)

    pv_mesh = pv.PolyData(vertices, faces_pv)
    pv_mesh.cell_data["colors"] = color

    # Plot in Streamlit
    #stpyvista.set_plot_theme("document")  # optional
    plotter = pv.Plotter(window_size=(600, 600))
    plotter.add_mesh(pv_mesh, scalars="colors", rgb=True)
    plotter.set_background("white")
    plotter.view_isometric()
    stpyvista(plotter)
    elapsed_mesh = time.time() - start

st.sidebar.header("Performance")
st.sidebar.text(f"Heightmap generated in {elapsed_map:.2f} seconds.")
st.sidebar.text(f"Mesh generated and rendered in {elapsed_mesh:.2f} seconds.")

