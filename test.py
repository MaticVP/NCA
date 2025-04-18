import numpy as np
from matplotlib import pyplot as plt
import time
from DP.NCA.gen_heightmaps import gen_hightmap
start = time.time()
uploaded_file = gen_hightmap("Noise Perlin")
end = time.time()
print(end - start)

plt.imshow(uploaded_file)
#plt.imsave("./maps/NCA_FBM")
#plt.imsave(f"./maps/NCA_perlin_noise_erosion.png", uploaded_file, cmap='gray')
plt.show()

