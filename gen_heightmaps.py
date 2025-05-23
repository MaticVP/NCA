import numpy as np
import torch

from CellularAutomata import CA, to_rgb, NoiseCA, FullCA


def gen_hightmap(type,numSteps=10,steps=12, res=128):

    if type == "Perlin" or type == "FBM":
        ca = CA()
    elif type == "Noise FBM" or type == "Noise Perlin":
        ca = NoiseCA(noise_level=0.2)
    else:
        ca = FullCA(noise_level=0.2)

    if type == "Perlin":
        ca.load_state_dict(torch.load("./ca_model_pearl_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "FBM":
        ca.load_state_dict(torch.load("./ca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise Perlin":
        ca.load_state_dict(torch.load("./nca_model_perlin_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise FBM":
        ca.load_state_dict(torch.load("./nca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Full Perlin":
        ca.load_state_dict(torch.load("./fca_model_perlin_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Full FBM":
        ca.load_state_dict(torch.load("./fca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    x = ca.seed(1,res)

    with torch.no_grad():
        for i in range(numSteps):

            for k in range(steps):

                x[:] = ca(x)

        #x = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        x = to_rgb(x).permute([0, 2, 3, 1])

        x = x.squeeze(0)

        x = x.detach().numpy()

        x = np.clip(x, 0, 1)

        return x
