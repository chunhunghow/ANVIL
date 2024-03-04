import os
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
import torchvision
import numpy as np
import torch
from segment_anything import sam_model_registry
from skimage import io, transform
import torch.nn.functional as F
from torchmetrics import Dice
from pytorch_lightning.loggers import CSVLogger
import re
from PIL import Image, ImageDraw
from dataloader import LoadPhysioNet
from utils import DiceLoss
from plot import get_contour
from skimage import io, transform
import logging

colors = ['red','blue','green']
colors = [(255,0,0,255), (0,0,255,255), (0,255,0,255)]
colors_trans = [(255,0,0,50), (0,0,255,50), (0,255,0,50)]


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



class Model(pl.LightningModule):
    def __init__(self, model, args):
        super().__init__()
        self.net = model
        self.multimask_output = args.multimask_output
        self.num_classes = 1 # in diceloss forward taking pred[:,0] , pred should only be [b,1,h,w]
        self.dice = DiceLoss(self.num_classes)
        self.ce = torch.nn.CrossEntropyLoss() #have to bcelogit? mask output is 3 , how do handle all three output? average? good to start with just 1 like medsam, 
        #otherwise all 3 masks will contain same information
        self.ce = torch.nn.BCEWithLogitsLoss()
        #blob loss
        # just to test whether other param updated
        self.tmp = self.net.mask_decoder.iou_token.weight.clone()
        if hasattr(self.net.mask_decoder,'anvil_prompt_embeddings'):
            self.tmp2 = self.net.mask_decoder.anvil_prompt_embeddings.weight.clone()

        self.val_score = []
        self.lr = args.lr

    
    def training_step(self, batch, batch_idx):
        im, mask, lb = batch
        if im.shape[1] == 1:
            im = im.repeat(1,3,1,1)
        B,_,H,W = im.shape
        #batched_input = [{"image": im[i], "original_size": im[i].shape[-2:] } for i in range(B)]
        #pred = self.net(batched_input, multimask_output=True)
        img_1024 = torchvision.transforms.functional.resize(im, (1024,1024))
        with torch.no_grad():
            image_embeddings = self.net.image_encoder(img_1024)
            sparse_embeddings, dense_embeddings = self.net.prompt_encoder(
                points=None,
                boxes= None,
                masks=None,
            )
        low_res_masks, iou_predictions = self.net.mask_decoder(
            #image_embeddings=curr_embedding.unsqueeze(0),
            image_embeddings=image_embeddings,
            image_pe=self.net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        #first try with single mask output
        if self.multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        pred_mask = low_res_masks[:, mask_slice, :, :]
        iou_pred = iou_predictions[:, mask_slice]
        # do evaluation on upsampled 1024 or low_res?
        #pred_mask = self.net.postprocess_masks(pred_mask, input_size=(pred_mask.shape[-2],pred_mask.shape[-1]), original_size=(args.img_size,args.img_size))
        pred_mask = F.interpolate(pred_mask, (H,W), mode="bilinear", align_corners=False)

        dice_loss = self.dice(torch.sigmoid(pred_mask),mask.to(float))
        ce_loss = self.ce(pred_mask[:,0],mask.to(float) )


        loss = dice_loss + ce_loss
        self.logger.log_metrics({"loss": loss})
        #loss = ce_loss
        return loss

    def training_epoch_end(self, output):
        print(f" Weight changed : {(self.tmp.detach().cpu() - self.net.mask_decoder.iou_token.weight.detach().cpu()).max()}")
        if hasattr(self.net.mask_decoder,'anvil_prompt_embeddings'):
            print(f" Weight changed of prompt: {(self.tmp2.detach().cpu() - self.net.mask_decoder.anvil_prompt_embeddings.weight.detach().cpu()).max()}")
            self.tmp2 = self.net.mask_decoder.anvil_prompt_embeddings.weight.clone()

        #save just prompt embedding manually

        filename = 'epoch='+str(self.current_epoch) + '.pt'
        self.output_path = os.path.join(self.logger._save_dir, self.logger._name, 'version_'+str(self.logger._version))
        weight_dir = os.path.join(self.output_path ,'checkpoints')
        os.makedirs(weight_dir, exist_ok=True)
        torch.save(self.net.mask_decoder.anvil_prompt_embeddings.weight , os.path.join(weight_dir,filename  ))

    def validation_step(self, batch, batch_idx):
        im, mask, lb = batch
        if im.shape[1] == 1:
            im = im.repeat(1,3,1,1)
        B,_,H,W = im.shape
        #batched_input = [{"image": im[i], "original_size": im[i].shape[-2:] } for i in range(B)]
        #pred = self.net(batched_input, multimask_output=True)

        # %% image preprocessing
        img_1024 = torchvision.transforms.functional.resize(im, (1024,1024))
        #im, mask, lb = batch
        #if im.shape[1] == 1:
        #    im = im.repeat(1,3,1,1)
        #B,_,H,W = im.shape

        image_embeddings = self.net.image_encoder(img_1024)
        sparse_embeddings, dense_embeddings = self.net.prompt_encoder(
            points=None,
            boxes= None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.net.mask_decoder(
            #image_embeddings=curr_embedding.unsqueeze(0),
            image_embeddings=image_embeddings,
            image_pe=self.net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        #first try with single mask output
        if self.multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        pred_mask = low_res_masks[:, mask_slice, :, :]
        iou_pred = iou_predictions[:, mask_slice]

        pred_mask = torch.sigmoid(pred_mask)

        #reason to do here like MedSAM did is the original SAM implemetaton padded empty pixels for preprocessing where we simply resize, so we do not follow
        # SAM postprocessing to drop padding, simply upsize twice to 1024 as the input size
        pred_mask = F.interpolate(pred_mask, (H,W), mode="bilinear", align_corners=False)

        # again threshold after interpolation
        pred_mask = (pred_mask > 0.5).cpu().to(int).numpy()  #original SAM use logit >0.0
        mask = mask.cpu().to(int).numpy()


        for i in range(B):
            tp, fp, fn, tn = compute_tp_fp_fn_tn(mask[i], pred_mask[i], ignore_mask=None)

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
            self.val_score += [results]

 
            if self.current_epoch > 0: 
                savedir = os.path.join(self.output_path ,'val_plots/')
                os.makedirs(savedir, exist_ok=True)
                contours = get_contour(pred_mask[i][0].astype(np.uint8))
                image = Image.fromarray((im[i] * 255).permute(1,2,0).cpu().numpy().astype(np.uint8)).convert('RGBA')
                img = Image.new('RGBA',image.size, (255,255,255,0))

                outline = 0
                for contour in contours:
                    x = contour[:, 0]
                    y = contour[:, 1]
                    polygon_tuple = list(zip(y, x))
                    ImageDraw.Draw(img).polygon(polygon_tuple, outline=colors[outline], fill=colors_trans[outline])
                image = Image.alpha_composite(image,img)

                contours = get_contour(mask[i].astype(np.uint8))
                outline = 1
                for contour in contours:
                    x = contour[:, 0]
                    y = contour[:, 1]
                    polygon_tuple = list(zip(y, x))
                    ImageDraw.Draw(img).polygon(polygon_tuple, outline=colors[outline], fill=colors_trans[outline])
                Image.alpha_composite(image,img).save(savedir + str(batch_idx) + f'_{i}.png')

    def validation_epoch_end(self,out):
        if len(self.trainer.lr_scheduler_configs) > 0:
            print("\nAverage Dice score: ",np.mean([r['Dice'] for r in self.val_score]), f"lr : {self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]}")
        else:
            print("\nAverage Dice score: ",np.mean([r['Dice'] for r in self.val_score]))
        self.logger.log_metrics({"Average Dice": np.mean([r['Dice'] for r in self.val_score])})


    def test_step(self, batch, batch_idx):
        im, mask, lb = batch
        if im.shape[1] == 1:
            im = im.repeat(1,3,1,1)
        B,_,H,W = im.shape
        #batched_input = [{"image": im[i], "original_size": im[i].shape[-2:] } for i in range(B)]
        #pred = self.net(batched_input, multimask_output=True)

        # %% image preprocessing
        img_1024 = torchvision.transforms.functional.resize(im, (1024,1024))
        #img_1024 = [torch.tensor(transform.resize(
        #    im[i].cpu().numpy(), (1024, 1024), order=3, preserve_range=True, anti_aliasing=True
        #).astype(np.uint8)).to(im.device)[None,] for i in range(B)] 


        image_embeddings = self.net.image_encoder(img_1024)
        sparse_embeddings, dense_embeddings = self.net.prompt_encoder(
            points=None,
            boxes= None,
            masks=None,
        )
        low_res_masks, iou_predictions = self.net.mask_decoder(
            #image_embeddings=curr_embedding.unsqueeze(0),
            image_embeddings=image_embeddings,
            image_pe=self.net.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=self.multimask_output,
        )

        #first try with single mask output
        if self.multimask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)

        pred_mask = low_res_masks[:, mask_slice, :, :]
        iou_pred = iou_predictions[:, mask_slice]

        pred_mask = torch.sigmoid(pred_mask)
        #pred_mask = (pred_mask > 0.5).cpu().to(float)  #original SAM use logit >0.0

        #pred_mask = self.net.postprocess_masks(pred_mask, input_size=(pred_mask.shape[-2],pred_mask.shape[-1]), original_size=(args.img_size,args.img_size))

        #reason to do here like MedSAM did is the original SAM implemetaton padded empty pixels for preprocessing where we simply resize, so we do not follow
        # SAM postprocessing to drop padding, simply upsize twice to 1024 as the input size
        pred_mask = F.interpolate(pred_mask, (H,W), mode="bilinear", align_corners=False)

        # again threshold after interpolation
        pred_mask = (pred_mask > 0.5).cpu().to(int).numpy()  #original SAM use logit >0.0
        mask = mask.cpu().to(int).numpy()


        for i in range(B):
            tp, fp, fn, tn = compute_tp_fp_fn_tn(mask[i], pred_mask[i], ignore_mask=None)

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
            self.val_score += [results]



    def test_epoch_end(self,out):
        print("\nAverage Dice score: ",np.mean([r['Dice'] for r in self.val_score]))
        self.logger.log_metrics({"Average Dice": np.mean([r['Dice'] for r in self.val_score])})
        self.val_score = []

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.net.mask_decoder.anvil_prompt_embeddings.parameters(), lr=self.lr)
        #opt = torch.optim.SGD(self.net.mask_decoder.anvil_prompt_embeddings.parameters(), lr=self.lr)
        #sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10 , eta_min=1e-3)
        #sched = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.5, total_iters=5)
        #sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,T_0=10, eta_min=1e-3)
        #logging.warning('Tuning the whole decoder now....\n')
        #opt = torch.optim.AdamW(self.net.mask_decoder.parameters(), lr=self.lr)
        return opt
        #return [opt],[sched]


def get_args():
    parser = argparse.ArgumentParser()
    #parser.add_argument('--dataset', type=str,
    #                    default='../data/Synapse', help='root dir for data')
    parser.add_argument('--bs', type=int,
                        default=8, help='batch size')
    parser.add_argument('--device', type=str,
                        default='0', help='Index of cuda device')
    parser.add_argument('--multimask_output', action='store_true',
                         help='SAM multimask')

    parser.add_argument('--img_size', type=int,
                        default=512, help='Image size for SAM')

    parser.add_argument('--num_tokens', type=int,
                        default=1, help='ANVIL Number of Nodes')
    parser.add_argument('--lr', type=float,
                        default=0.01, help='Learning rate')
    args = parser.parse_args()
    return args
    
if __name__ == '__main__':
    args = get_args()   

    image_path = '/home/data_repo/INSTANCE_ICH/data2D/images/train/'
    mask_path = '/home/data_repo/INSTANCE_ICH/data2D/masks/train/'
    dataset = LoadPhysioNet(path = image_path, train=True, label=True, img_size=args.img_size) #1024 following SAM paper and MedSAM
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.bs ,num_workers=24)
    testdataset = LoadPhysioNet(path = re.sub('train','test',image_path), train=False, label=True, img_size=args.img_size) #1024 following SAM paper and MedSAM
    testloader = torch.utils.data.DataLoader(testdataset, batch_size=args.bs, num_workers=24)


    device = [int(args.device)] if ',' not in args.device else [int(i) for i in args.device.split(',')]
    trainer = Trainer(accelerator='gpu', devices=device , max_epochs=100)
    #model = sam_model_registry["vit_b"](checkpoint='../MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth')
    model = sam_model_registry["vit_b"](checkpoint='../MedSAM/work_dir/MedSAM/medsam_vit_b.pth')
    model = Model(model, args)
    print('First we have a baseline of how original SAM performs on the test dataset without any prompt.')
    trainer.test(model, testloader)


    # trying original SAM
    model = Model(
                #sam_model_registry["anvil_b"](checkpoint='../MedSAM/work_dir/MedSAM/medsam_vit_b.pth',num_tokens=args.num_tokens)
                sam_model_registry["anvil_b"](checkpoint='../MedSAM/work_dir/SAM/sam_vit_b_01ec64.pth',num_tokens=args.num_tokens)
                ,args )

    logger = CSVLogger('logs', name='ANVIL')
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = Trainer(accelerator='gpu', devices=device, logger=logger, max_epochs=100, enable_checkpointing=False, callbacks=[lr_monitor])
    trainer.logger.log_hyperparams(args.__dict__)
    #trainer.logger.log_hyperparams({'bs': args.bs , 'num_tokens': args.num_tokens})
    trainer.fit(model, loader,testloader)
