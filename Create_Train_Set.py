import numpy as np
import os
import glob
import h5py
from scipy.misc import imread
from scipy.misc import imresize
from TEST import rgb2ycbcr
from TEST import cropping

train_list = []
data_path = "./data/291"    # YOU MUST CHECK THIS PATH, 291, sharp_3, sharp_5
data_dir =  data_path
data = glob.glob(os.path.join(data_dir, "*.jpg"))
data += glob.glob(os.path.join(data_dir, "*.bmp"))
data.sort()

img = []
for z in range(4):
    img.append([])

for i in range(len(data)):
    img_raw = imread(data[i], mode='RGB')
    img[0] = np.round(rgb2ycbcr(img_raw)).astype('uint8')[:,:,0]/255.0

    for z in range(2, 5):
        img[z-1] = imresize(imresize(img[0], 1/z, interp='bicubic', mode='F'), (img[0].shape[0], img[0].shape[1]), interp='bicubic', mode='F')

    width = img[0].shape[1]
    height = img[0].shape[0]
    for z in range(4):
        img[z] = cropping(img[z],height, width).astype(np.float32)

    patch_size = 41
    stride = 41
    x_size = int((img[0].shape[1]-patch_size)/stride+1)
    y_size = int((img[0].shape[0]-patch_size)/stride+1)

    for x in range(x_size):
        for y in range(y_size):
            x_coord = x*stride
            y_coord = y*stride

            patch = img[0][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size]
            for z in range(1, 4):
                train_list.append([patch, img[z][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size]])

            patch = np.rot90(img[0][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size], 1)
            for z in range(1, 4):
                train_list.append([patch, np.rot90(img[z][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size], 1)])

            patch = np.fliplr(img[0][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size])
            for z in range(1, 4):
                train_list.append([patch, np.fliplr(img[z][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size])])

            patch = np.fliplr(np.rot90(img[0][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size], 1))
            for z in range(1, 4):
                train_list.append([patch, np.fliplr(np.rot90(img[z][y_coord:y_coord + patch_size, x_coord:x_coord + patch_size], 1))])
    print(i)    # up to 291 ( 0~291 )
with h5py.File('./data/train.h5','w') as hf:
    hf.create_dataset("train", data=train_list)
print("Success process")