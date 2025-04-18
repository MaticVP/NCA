import numpy as np
from matplotlib import pyplot as plt
import time
from DP.NCA.gen_heightmaps import gen_hightmap
start = time.time()
uploaded_file = gen_hightmap("Perlin")
end = time.time()
print(end - start)

