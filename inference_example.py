from all_in_one import AllInOneModel, read_image
import os
from skimage.io import imsave
import torch


paths = [
    "./data_example/example_tiffs/e-0479_c-1_siRNA-11_pos-10_RI.tiff",
    "./data_example/example_tiffs/e-0479-c-61-untreated-test_RI.tiff",
]
focus_frame_idxs = [41, 48]
model = AllInOneModel("./weights/seq_lrs4_lrt5_lrd5_nucl.tar")

img = torch.cat([read_image(path) for path in paths], axis=0)
pred = model(img, focus_frame_idxs)

imsave(f"./results/test1.png", pred[0, 0].astype("uint8") * 255)
imsave(f"./results/test2.png", pred[1, 0].astype("uint8") * 255)
