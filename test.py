import numpy as np
from matplotlib import pyplot as plt
import time
from DP.NCA.gen_heightmaps import gen_hightmap

grid_size = 1

map = [None for i in range(grid_size*grid_size)]

for i in range(grid_size):
    seed = [None, None, None]
    for j in range(grid_size):

        if i > 0:
            seed[1] = map[(i-1) * grid_size + j]
        if j > 0 and i > 0:
            index = (i-1) * grid_size + (j-1)
            seed[2] = map[index]
        now = time.time()
        uploaded_file = gen_hightmap("Perlin",seed, res=128, numSteps=14,steps=5, seed_size=32)
        delta = time.time() - now
        print(delta)
        seed[0] = uploaded_file

        map[i * grid_size + j] = uploaded_file

rows = []
for i in range(grid_size):
    row = [map[i * grid_size + j] for j in range(grid_size)]
    row_combined = np.hstack(row)
    rows.append(row_combined)

full_image = np.vstack(rows)

plt.imshow(full_image, cmap='gray')
plt.axis('off')
plt.savefig("compare.png", bbox_inches='tight', pad_inches=0)
plt.show()

