from all_in_one import AllInOneModel
import os
from skimage.io import imsave
import torch
import numpy as np
from imageio import volread

paths = [
    "./data_example/e-0479_c-1_siRNA-11_pos-10_RI.tiff",
    "./data_example/e-0479-c-61-untreated-test_RI.tiff",
]

img = torch.cat([torch.Tensor([volread(path)]).float() for path in paths], dim=0)
focus_frame_idxs = 41

model = AllInOneModel(focus_frame_idx=focus_frame_idxs)
model.load_state_dict(torch.load("./weights/nucleoli_weights.tar", map_location=torch.device('cpu')))

pred = model(img).data.cpu().numpy()

imsave(f"./results/test1.png", pred[0, 0].astype("uint8") * 255)
imsave(f"./results/test2.png", pred[1, 0].astype("uint8") * 255)
