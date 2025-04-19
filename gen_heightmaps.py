import torch

from CellularAutomata import CA, to_rgb, NoiseCA


def gen_hightmap(type,numSteps=20,steps=12, res=512):

    if type == "Perlin" or type == "FBM":
        ca = CA()
    elif type == "Noise FBM" or type == "Noise Perlin":
        ca = NoiseCA(noise_level=0.2)

    if type == "Perlin":
        ca.load_state_dict(torch.load("./ca_model_pearl_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "FBM":
        ca.load_state_dict(torch.load("./ca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise Perlin":
        ca.load_state_dict(torch.load("./nca_model_perlin_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    if type == "Noise FBM":
        ca.load_state_dict(torch.load("./nca_model_fbm_ero.pt", weights_only=True, map_location=torch.device('cpu')))

    x = ca.seed(1,res)

    with torch.no_grad():
        for i in range(numSteps):

            print(i / numSteps)

            for k in range(steps):

                x[:] = ca(x)

        #x = to_rgb(x).permute([0, 2, 3, 1]).cpu()
        x = to_rgb(x).permute([0, 2, 3, 1])

        x = x.squeeze(0)

        x = x.detach().numpy()

        return x
