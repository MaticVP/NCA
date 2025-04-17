import streamlit as st
import numpy as np
import pyvista as pv
import torch
from PIL import Image
from stpyvista import stpyvista
from gen_heightmaps import gen_hightmap

torch.classes.__path__ = []

from gen_mesh import gen_map

st.title("Upload Image & Render Heightmap")

uploaded_file = st.file_uploader("Upload a heightmap image (.png, .jpg)", type=["png", "jpg", "jpeg"])

st.sidebar.header("Mesh Parameters")
resolution = st.sidebar.slider("Resolution (number of samples)", min_value=64, max_value=128, value=128, step=1)
pixel_scale = st.sidebar.number_input("Pixel Scale (e.g., meters per pixel)", min_value=1, max_value=50, value=20, step=1)
height_scale = st.sidebar.number_input("Height Scale", min_value=0.01, max_value=5.0, value=1.0, step=0.1)
st.sidebar.header("NCA Parameters")
numSteps = st.sidebar.slider("number of steps", min_value=1, max_value=300, value=300, step=1)
steps = st.sidebar.number_input("Iterations per step", min_value=1, max_value=96, value=32, step=1)
res = st.sidebar.slider("Resolution (number of samples)", min_value=32, max_value=256, value=32, step=1)

map_type = st.sidebar.selectbox("Map Type", ["Perlin", "FBM"])

img = None

if st.sidebar.button("Generate Heightmap"):
    uploaded_file = gen_hightmap(map_type,numSteps,steps,res)
    img = uploaded_file
    img *= 255
    img = img.astype(np.uint8)
    img = Image.fromarray(img)
    img = img.convert('L')

elif uploaded_file is not None:
    img = Image.open(uploaded_file).convert('L')

if uploaded_file is not None:
    st.session_state.hmap = img



pyvista.start_xvfb()

if uploaded_file is not None or "hmap" in st.session_state.keys():
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

