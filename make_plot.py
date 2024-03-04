
# This file is to draw polygon for mask overlay for better visisbility

import torch
import re
import os
from PIL import Image
import glob
import numpy as np

from plot import create_mask_from_polygon

if __name__ == '__main__': 
    pred_dirs = 'output/'
    image_path = '/home/data_repo/INSTANCE_ICH/data2D/images/test/'
    mask_path = '/home/data_repo/INSTANCE_ICH/data2D/masks/test/'
    
    os.makedirs('plots_ch', exist_ok = True)
    for path in os.listdir(image_path):
        im_ = image_path + path
        label_ = mask_path + path
        pred_ =  pred_dirs +'seg_' +path
        im = np.array(Image.open(im_))
        lab = np.array(Image.open(label_))
        lab = lab * 255 if lab.min() == 1 else lab
        pred = np.array(Image.open(pred_))
        pred = pred * 255 if pred.min() == 1 else pred

        fname = path[:-4]
        contour_img = create_mask_from_polygon(im, lab, outline=1)
        contour_img = create_mask_from_polygon(np.array(contour_img), pred, outline=0)
        contour_img.save(f'plots_ch/{fname + "cntr.png"}', dpi=(300,300))


        contour_img = create_mask_from_polygon(im, lab, outline=1)
        contour_img.save(f'plots_ch/{fname + "cntr_gt.png"}', dpi=(300,300))


