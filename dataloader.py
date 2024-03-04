



import os
import glob
import pydicom
import torch
from torch.utils.data import Dataset
import albumentations as A
import nrrd
import sys
from albumentations.pytorch import ToTensorV2
import re
import io
from PIL import Image
import numpy as np
import cv2
import random
from torchvision.ops import masks_to_boxes

class LoadPhysioNet():

    '''
    PhysioNet, INSTANCE dataset has been stored into PNG
    Turn mask into polygon contour.

    '''
    def __init__(self, 
                 path=None, 
                 mode='abnormal', 
                 label=False, 
                 img_size=256, 
                 max_seq_len =512 , 
                 eps = 0.001,
                 train = True,
                 healthy_path = '/home/data_repo/physionetICH/data2D/healthy'):
        '''
        Args:
        path: path to folder of abnormal 
        max_seq_len: The sequence length for each polygon 
        eps: epsilon distance for rendering cv2 contour.
        rotate_polygon: Augment the contour points, any point can be starting point.
        '''
        assert mode in ['abnormal', 'normal', 'all'] , 'Mode should be `abnormal`, `normal` or `all`'
        self.mode = mode
        self.img_size = img_size
        self.label = label
        self.train = train
        healthy_images = glob.glob(healthy_path+ '/*')
        if self.mode == 'all':
            self.images = glob.glob(path + '/*')
            self.images = healthy_images + self.images
        elif self.mode == 'normal':
            self.images = healthy_images
        else:
            self.images = glob.glob(path + '/*')
        self.transform_labeled = A.Compose([
              A.HorizontalFlip(p=0.5),              
              A.Affine(scale=(0.90,1.0),
                       translate_percent=(0.,0.2),
                       rotate=(-15,15),
                       shear=(-8,8)
                  ),
              #A.RandomResizedCrop(height=img_size,
              #                    width=img_size, 
              #                    scale=(0.85,1.0),
              #                    p=0.7),
              A.RandomBrightnessContrast(),
              A.GaussNoise(),
              ToTensorV2(), #not normalized
              ])


        #self.aug1 = PixelShuffling(30)
        self.max_seq_len = max_seq_len
        self.eps = eps

    def __len__(self):

        return len(self.images)

    def __getitem__(self, idx):

        """
        Actual function to load public annotated dataset. 

        Returns:
            im : `Tensor`
            boxes: `Tensor` (n 4) Absolute coordinates.
            contours: `List[Tensor]` A of list of lists of polygon coordinates.
            where tensor in shape ( 256, 1, 3) where 3 refers to (x, y, contour_ind) , one box can contain more than 1 contour.
            List is the number of boxes.
                    
        """

        im = np.array(Image.open(self.images[idx])) #[h,w,3]
        if len(im.shape) == 3:
            im = im[:,:,0]

        h0, w0 = im.shape[:2]  # orig hw
        r = self.img_size / max(h0, w0)  # ratio
        if r != 1:  # if sizes are not equal
            interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
            im = cv2.resize(im, (self.img_size, self.img_size), interpolation=interp)           

        if (self.mode == 'abnormal') & self.label:
            p = re.sub( 'images','masks',self.images[idx])
            mask = np.array(Image.open(p))

            if r != 1:
                interp = cv2.INTER_AREA  # random.choice(self.rand_interp_methods)
                mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=interp)           
            mask = (mask >0).astype(int)

            assert mask.sum() > 0 , f'Empty Mask before transform! {self.images[idx]} {mask.shape}'
            if self.train:
                pass_  = 0
                while pass_ == 0:
                    augmented = self.transform_labeled(image=im, mask=mask)
                    mask_attempt = augmented['mask']
                    if mask_attempt.sum() > 0 : 
                        pass_ = 1
                        im = augmented['image']
                        mask = augmented['mask']
            else:
                #im = torch.tensor(im).permute(2,0,1)
                im = torch.tensor(im)
                mask = torch.tensor(mask)
            im = (im - im.min())/np.ptp(im)

            # Knowing that the mask is assumed to be single lesion only.
            assert mask.sum() > 0 , f'Empty Mask after transform! {self.images[idx]} {mask.shape}'
            boxes = masks_to_boxes(mask[None,]) #xyxy

            if len(boxes) == 0:
                nz = mask.nonzero()
                xmin , ymin = nz[:,1].min() -2 , nz[:,0].min() -2
                xmax, ymax = max(nz[:,1].max() +2, xmin+2) , max(nz[:,0].max()+2, ymin+2)
                boxes = torch.tensor([[xmin,ymin, xmax, ymax]])
                assert len(boxes) > 0 

            
            if self.train:
                return im, mask, boxes
            else:
                return im[None,...], mask, boxes



        im = (im - im.min())/np.ptp(im)
        return torch.from_numpy(im).to(torch.float32)[None,...]



