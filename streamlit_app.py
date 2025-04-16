import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import streamlit as st
from io import BytesIO

def generate_heightmap(seed, scale, resolution):
    torch.manual_seed(seed)
    res = int(resolution)
    heightmap = scale * torch.randn(res, res).numpy()

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111, projection='3d')
    X, Y = np.meshgrid(np.arange(res), np.arange(res))
    ax.plot_surface(X, Y, heightmap, cmap='terrain')

    buf = BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)

    return Image.open(buf)

# Streamlit UI
st.title("Heightmap Generator")

seed = st.slider("Random Seed", 0, 1000, 42)
scale = st.slider("Scale", 0.1, 5.0, 1.0)
resolution = st.slider("Resolution", 32, 128, 64)

# Generate and display heightmap
st.image(generate_heightmap(seed, scale, resolution))
