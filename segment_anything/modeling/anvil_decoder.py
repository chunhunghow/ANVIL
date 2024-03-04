

from .mask_decoder import MaskDecoder
import torch
from torch import nn
from torch.nn import Dropout
from typing import List, Tuple, Type
import math
from functools import partial, reduce
from operator import mul

class ANVILDecoder(MaskDecoder):
    def __init__(self, num_tokens, **kwargs):

        # TODO
        # prompt_config should contain DEEP,LOCATION,NUM_TOKENS,INITIATION, DROPOUT
        super().__init__(**kwargs)
        
        self.embed_dim = self.transformer_dim
        #self.prompt_config = prompt_config
        #self.prompt_dropout = Dropout(self.prompt_config.DROPOUT)
        self.prompt_dropout = Dropout(0.0)
        patch_size = (16,16)
        #num_tokens = self.prompt_config.NUM_TOKENS
        #intialization
        val = math.sqrt(6. / float(3 * reduce(mul, patch_size, 1) + self.embed_dim))  # noqa
        #self.anvil_prompt_embeddings = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        self.anvil_prompt_embeddings = nn.Embedding(num_tokens, self.embed_dim)
        #nn.init.uniform_(self.anvil_prompt_embeddings.data, -val, val)
        nn.init.uniform_(self.anvil_prompt_embeddings.weight, -val, val)




    #def forward(self, x: torch.Tensor) -> torch.Tensor:
    #    B = x.shape[0]
    #    x = self.patch_embed(x)
    #    if self.pos_embed is not None:
    #        x = x + self.pos_embed

    #    # does SAM embedding has pos_drop? pos_drop in VPT is from timm VisionTransformer _pos_emb function, 
    #    # vpt concat class token in embedding layer, add pos_emb and dropout
    #    # vpt patch_embed is using timm ViT patch_embed, which has a conv layer and norm, takes in (B,C,H,W) -> BLC
    #    # vpt has a cls_token concat in front of patch embedding, following timm implementtation beforre pos emb
    #    # for sam , cls_token is not needed, pos_drop not used
    #
    #    #use prepend first
    #    # prompt_embeddings if not initialized during train, should be provided
    #    x = torch.cat(
    #            (   
    #                self.prompt_dropout(self.prompt_embeddings.expand(B, -1,-1)), 
    #                x,
    #            )   
    #            ,dim=1)       

    #    for blk in self.blocks:
    #        x = blk(x)

    #    x = self.neck(x.permute(0, 3, 1, 2))

    #    return x


    def predict_masks(
        self,
        image_embeddings: torch.Tensor,
        image_pe: torch.Tensor,
        sparse_prompt_embeddings: torch.Tensor,
        dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        output_tokens = torch.cat(
            [self.iou_token.weight, self.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(
            sparse_prompt_embeddings.size(0), -1, -1
        )
        #tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        #chunhung: sparse embedding is torch empty if no point or box provided, for mask its no_mask_embed, if theres box, the coord turned to points and pe_encoded
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings, self.anvil_prompt_embeddings.weight[None,] ), dim=1)
        # Expand per-image data in batch direction to be per-mask
        if image_embeddings.shape[0] != tokens.shape[0]:
            src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        else:
            src = image_embeddings
        src = src + dense_prompt_embeddings
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)
        b, c, h, w = src.shape

        # Run the transformer
        hs, src = self.transformer(src, pos_src, tokens)

        # chunhung: iou token is similar to a cls_token to predict mask quality,  
        iou_token_out = hs[:, 0, :]
        mask_tokens_out = hs[:, 1 : (1 + self.num_mask_tokens), :]


        # Upscale mask embeddings and predict masks using the mask tokens
        #chunhung : output scaling and output hypernetworks has same output dim , each mask token is dot prod with each pixel,-> (B , num_token, image_len)
        src = src.transpose(1, 2).view(b, c, h, w)
        upscaled_embedding = self.output_upscaling(src)
        hyper_in_list: List[torch.Tensor] = []
        for i in range(self.num_mask_tokens):
            hyper_in_list.append(
                self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :])
            )
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


