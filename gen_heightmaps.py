import torch

from CellularAutomata import CA, to_rgb

def gen_hightmap(type,numSteps=300,steps=32, res=64):

    ca = CA()

    if type == "Perlin":
        ca.load_state_dict(torch.load("./ca_model_pearl_ero.pt", weights_only=True))

    if type == "FBM":
        ca.load_state_dict(torch.load("./ca_model_fbm_ero.pt", weights_only=True))

    x = None

    x = ca.seed(1,res)

    with torch.no_grad():
        for i in range(numSteps):

            print(i / numSteps)

            for k in range(steps):

                x[:] = ca(x)



        x = to_rgb(x).permute([0, 2, 3, 1]).cpu()

        x = x.squeeze(0)

        x = x.detach().numpy()

        return x
