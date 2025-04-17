import numpy as np
from matplotlib import pyplot as plt

from DP.NCA.gen_heightmaps import gen_hightmap

uploaded_file = gen_hightmap("Perlin")

plt.imshow(uploaded_file)
plt.show()