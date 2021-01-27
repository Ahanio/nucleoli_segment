from all_in_one import AllInOneModel
import os
from skimage.io import imsave


path = './data_example/example_tiffs/e-0479_c-1_siRNA-11_pos-10_RI.tiff'
model = AllInOneModel('./weights/seq_lrs4_lrt5_lrd5_nucl.tar')
pred = model(path, 41)

imsave(f"./results/test.png", pred.astype('uint8')*255)