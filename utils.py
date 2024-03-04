import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
#import SimpleITK as sitk
import cv2
import os
import math
from PIL import Image
#from plot import create_mask_from_polygon
import time


class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            #print('before:',input_tensor.shape,input_tensor)
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss
      

   
    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
            #print('NEW INPUT',inputs)
        target = self._one_hot_encoder(target)

        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes



def dice(pred, true, k = 1):
    intersection = ((pred == k)*(true==k)).sum() * 2.0
    dice = intersection / ((pred == k).sum() + (true == k).sum() + 1e-6)
    return dice

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
  
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        iou = metric.binary.jc(pred,gt)
        return dice, hd95, iou
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0 , 0
    else:
        return 0, 0 ,0 




def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,base_dir=None,split=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    

    batch_inference_time = []  

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                outputs = net(input)
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            start = time.time()
            out = net(input)
            batch_inference_time += [time.time() - start]
            out = torch.argmax(torch.softmax(out, dim=1), dim=1).squeeze(0)
            test_bs = input.shape[0]
            prediction = out.cpu().detach().numpy()
            
            
            
 
    
    image_path = os.path.join(base_dir,'images',split,case+'.png')
    bbox_path = os.path.join(base_dir,'labels',split,case+'.txt')
    mask_path = os.path.join(base_dir,'masks',split,case+'.png')
    
    #if os.stat(bbox_path).st_size != 0:
    mask = Image.open(mask_path).convert('L')
    image = Image.open(image_path).convert('L')
    image = np.array(image)
    mask = np.array(mask)
    image= image.astype(np.float32) / 255.
    mask = mask/255.
    
    with open(bbox_path) as f:
            lines = [line.rstrip() for line in f]
            left,top,right,bottom = 512,512,0,0
            for bbox in lines:

                central_x = float(bbox.split(' ')[1])*512
                central_y = float(bbox.split(' ')[2])*512
                box_width = float(bbox.split(' ')[3])*512
                box_height = float(bbox.split(' ')[4])*512
                min_y= math.floor(central_y-box_height/2)
                max_y = math.ceil(central_y+box_height/2)
                min_x =math.floor(central_x-box_width/2)
                max_x =math.ceil(central_x+box_width/2)
                left = min(left,min_x)
                top = min(top,min_y)
                right = max(right,max_x)
                bottom = max(bottom,max_y)

    

    
    #print(left,top,right,bottom)
    #print(right-left,bottom-top)
    #print(prediction.shape)
    #print(abs(right-left),abs(bottom-top))
    predicted_mask_resized =  cv2.resize(prediction, dsize=(right-left,bottom-top), interpolation=cv2.INTER_NEAREST)
    
    
    # Assume that the original image size is (H, W) and the bounding box coordinates are (left, top, right, bottom)
    H, W = 512,512

    # Create a binary mask for the bounding box area
    bbox_mask = np.zeros((H, W))

    # Overlay the predicted mask on the ground truth mask
    bbox_mask[top:bottom, left:right] = predicted_mask_resized  # overlay the predicted mask on the bounding box area
                
    prediction = bbox_mask
    label = mask
            
    uniquelabels= np.unique(label)
    if uniquelabels.any():
        actual_label = (uniquelabels[np.nonzero(uniquelabels)])[0]    
    else:
        actual_label = 10
    new_dice= dice(prediction, label, k = actual_label)
    dice_list= [actual_label,new_dice]

    
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction ==i , label==i ))
    

    if test_save_path is not None:
        contour_img = create_mask_from_polygon(image *255, label*255, outline=1)
        contour_img = create_mask_from_polygon(np.array(contour_img), prediction*255, outline=0)
        # img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        # prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        # lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        #print(image)
        #print(prediction.shape)
        #image = Image.fromarray(np.uint8(image * 255), 'RGB')
        #image.save(test_save_path + '/'+case + "_image.png")
        
        # np.save()
        
        # image.save(test_save_path + '/'+case + "_image.png")
        cv2.imwrite(test_save_path + '/'+case + "_image.png", image*255)
        cv2.imwrite(test_save_path + '/'+case + "_label.png", label*255)
        cv2.imwrite(test_save_path + '/'+case + "_pred.png", prediction*255)
        contour_img.save(test_save_path + '/'+case + "_cntr.png")
        #print(lab_itk)
        # img_itk.SetSpacing((1, 1, z_spacing))
        # prd_itk.SetSpacing((1, 1, z_spacing))
        # lab_itk.SetSpacing((1, 1, z_spacing))
        #sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        #sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        #sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")

    return metric_list,dice_list, batch_inference_time, test_bs
