import pyvista
import streamlit as st
import numpy as np
import pyvista as pv
import trimesh
from stpyvista import stpyvista

from gen_mesh import gen_map

st.title("Upload Image & Render Heightmap")

uploaded_file = st.file_uploader("Upload a heightmap image (.png, .jpg)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    pyvista.start_xvfb()
    mesh, color = gen_map(uploaded_file)


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

