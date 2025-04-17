import pyvista
import streamlit as st
import numpy as np
import pyvista as pv
import trimesh
from stpyvista import stpyvista

from gen_mesh import gen_map

st.title("Upload Image & Render Heightmap")

uploaded_file = st.file_uploader("Upload a heightmap image (.png, .jpg)", type=["png", "jpg", "jpeg"])

st.sidebar.header("Mesh Parameters")
resolution = st.sidebar.slider("Resolution (number of samples)", min_value=64, max_value=256, value=128, step=1)
pixel_scale = st.sidebar.number_input("Pixel Scale (e.g., meters per pixel)", min_value=1, max_value=50, value=20, step=1)
height_scale = st.sidebar.number_input("Height Scale", min_value=0.01, max_value=5.0, value=1.0, step=0.1)


if uploaded_file is not None:
    #pyvista.start_xvfb()
    mesh, color = gen_map(uploaded_file,(resolution,resolution),pixel_scale,height_scale)


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

