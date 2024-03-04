

# trying out INSTANCE dataset
import math
import time
import glob 
import os
from PIL import Image
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from torchmetrics import Dice
join = os.path.join


def compute_tp_fp_fn_tn(mask_ref: np.ndarray, mask_pred: np.ndarray, ignore_mask: np.ndarray = None):
    if ignore_mask is None:
        use_mask = np.ones_like(mask_ref, dtype=bool)
    else:
        use_mask = ~ignore_mask
    tp = np.sum((mask_ref & mask_pred) & use_mask)
    fp = np.sum(((~mask_ref) & mask_pred) & use_mask)
    fn = np.sum((mask_ref & (~mask_pred)) & use_mask)
    tn = np.sum(((~mask_ref) & (~mask_pred)) & use_mask)
    return tp, fp, fn, tn


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, H, W):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]  # (B, 1, 4)

    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None,
        boxes=box_torch,
        masks=None,
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,  # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),  # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings,  # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings,  # (B, 256, 64, 64)
        multimask_output=False,
    )

    low_res_pred = torch.sigmoid(low_res_logits)  # (1, 1, 256, 256)

    low_res_pred = F.interpolate(
        low_res_pred,
        size=(H, W),
        mode="bilinear",
        align_corners=False,
    )  # (1, 1, gt.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  # (256, 256)
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg

image_path = '/home/data_repo/INSTANCE_ICH/data2D/images/test/'
mask_path = '/home/data_repo/INSTANCE_ICH/data2D/masks/test/'


labels_path_map = glob.glob('/home/lchin004/INSTANCE_data/new_split/labels/test/*') + glob.glob('/home/lchin004/INSTANCE_data/new_split/labels/val/*') + glob.glob('/home/lchin004/INSTANCE_data/new_split/labels/train/*')

labels_path_map = dict([(p.split('/')[-1][:-4], p) for p in labels_path_map])

processed = []


sz = Image.open(image_path+ os.listdir(image_path)[0]).size

for p in os.listdir(image_path):
    lp = labels_path_map[p[:-4]]
    current_im_lab = []
    with open(lp, 'r') as f:
        lines = [line.rstrip() for line in f]
    #a = np.zeros((512,512)) 
    for bbox in lines:
        image_class = bbox.split(' ')[0]
        central_x = float(bbox.split(' ')[1])* sz[1]
        central_y = float(bbox.split(' ')[2])* sz[0]
        box_width = float(bbox.split(' ')[3])* sz[1]
        box_height = float(bbox.split(' ')[4])* sz[0]
        min_y= math.floor(central_y-box_height/2)
        max_y = math.ceil(central_y+box_height/2)
        min_x =math.floor(central_x-box_width/2)
        max_x =math.ceil(central_x+box_width/2)
        current_im_lab += [(min_x, min_y, max_x, max_y)]


    processed += [(p,  image_path + p ,current_im_lab , mask_path + p)]

print('Done gathering data...\n')


device = 'cuda'

#MedSAM model
#medsam_model = sam_model_registry["vit_b"](checkpoint='work_dir/MedSAM/medsam_vit_b.pth')

#SAM Model
medsam_model = sam_model_registry["vit_b"](checkpoint='work_dir/SAM/sam_vit_b_01ec64.pth')
import pdb; pdb.set_trace()

medsam_model = medsam_model.to(device)
medsam_model.eval()
seg_path = 'output'
dice = Dice(average='micro')
total_dice = []
dataset = []

for im_name , im_path, label, mask in processed:
    img_np = io.imread(im_path)
    mask = io.imread(mask)

    if len(img_np.shape) == 2:
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
    else:
        img_3c = img_np
    H, W, _ = img_3c.shape
    # %% image preprocessing
    img_1024 = transform.resize(
        img_3c, (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
    ).astype(np.uint8)
    img_1024 = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )  # normalize to [0, 1], (H, W, 3)
    # convert the shape to (3, H, W)
    img_1024_tensor = (
        torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
    )
    
    box_np = np.array(label)
    # transfer box_np t0 1024x1024 scale
    box_1024 = box_np / np.array([W, H, W, H]) * 1024
    with torch.no_grad():
        image_embedding = medsam_model.image_encoder(img_1024_tensor)  # (1, 256, 64, 64)
    
    start = time.time()
    medsam_seg = medsam_inference(medsam_model, image_embedding, box_1024, H, W)
    end = time.time()

    tp, fp, fn, tn = compute_tp_fp_fn_tn(mask,medsam_seg, ignore_mask=None)

    results = {}
    if tp + fp + fn == 0:
        results['Dice'] = np.nan
        results['IoU'] = np.nan
    else:
        results['Dice'] = 2 * tp / (2 * tp + fp + fn + 1)
        results['IoU'] = tp / (tp + fp + fn + 1)
        results['FP'] = fp
        results['TP'] = tp
        results['FN'] = fn
        results['TN'] = tn
        results['n_pred'] = fp + tp
        results['n_ref'] = fn + tp

    #met = dice(torch.tensor(medsam_seg), torch.tensor(mask)).item()
    print(f'Predict {im_name} ... Dice score {results["Dice"]} ,  Time : {end - start}')
    total_dice += [results['Dice']]

    io.imsave(
        join(seg_path, "seg_" + im_name),
        medsam_seg,
        check_contrast=False,
    )
    results['name'] = im_name
    dataset += [results]

print(f'Average Dice Score {np.mean(total_dice)}')


import pandas as pd
from matplotlib import pyplot as plt
data = pd.DataFrame(dataset)
data.to_csv('sam_test_instance.csv',index=False)
data[['n_ref','Dice']].plot.scatter(x='n_ref',y='Dice')
plt.xlabel('Lesion size')
plt.ylabel('Dice Score')
plt.title('SAM inference on INSTANCE dataset')
plt.savefig('sam_test_instance.png')

