# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 12:43:18 2023

@author: rw17789
"""

import torch
import torch.nn as nn
from einops.einops import rearrange
import torch.nn.functional as F
from .main_model import ImageRegistration
from .backbone import Feature_Extraction_2
from .loftr_transformer import LocalFeatureTransformer, PositionEncodingSine

from torch.nn.utils import spectral_norm
import math

from .xacgan import Generator#, ssim_loss
from itertools import chain

def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _compute_ssim(img1, img2, window, window_size, channel, R = 1, size_average = True):
    assert img1.dim() == 4
    assert img1.shape == img2.shape
    
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = (0.01 * R) ** 2
    C2 = (0.03 * R) ** 2
    

    output = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
    return output.mean() if size_average else output.mean(dim=(1,2,3))

def ssim_loss(img1, img2, range = (-1,1), window_size = 11, size_average = True):
    assert img1.dim() == 4
    assert img1.shape == img2.shape
    R = range[1] - range[0]
    img1 = img1 - range[0]
    img2 = img2 - range[0]
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return 1 - _compute_ssim(img1, img2, window, window_size, channel, R, size_average)

class Layer(nn.Module):
    def __init__(self, inc, outc, trans_type = "in"):
        super().__init__()
        conv = nn.Conv2d if trans_type != "up" else nn.ConvTranspose2d
        layer = [
            spectral_norm(nn.Conv2d(inc, outc, 3, 1, 1, bias = False)),
            # nn.BatchNorm2d(outc),
            nn.ReLU() if trans_type != "down" else nn.LeakyReLU(0.2),
            spectral_norm(conv(outc, outc, 3 if trans_type == "in" else 4, 1 if trans_type == "in" else 2, 1, bias = False)),
            # nn.BatchNorm2d(outc),
            nn.ReLU() if trans_type != "down" else nn.LeakyReLU(0.2)
        ]
        if trans_type == "down":
            layer.insert(2, nn.Dropout(0.5))
            layer.insert(-1, nn.Dropout(0.5))
        self.layer = nn.Sequential(*layer)
        
        if trans_type == "in" and inc == outc:
            # assert inc == outc
            self.bypass = None
        else:
            self.bypass = nn.Sequential(spectral_norm(nn.Conv2d(inc, outc, 1, bias = False)))
            if trans_type != "in":
                self.bypass.add_module(
                    "resize",
                    nn.MaxPool2d(2,2) if trans_type == "down" else  nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
                    )
        
    def forward(self, x):
        return self.layer(x) + self.bypass(x) if self.bypass is not None else x
    

class UNet_Trans(nn.Module):
    def __init__(self, in_channel, out_channel = 1, hidden_channel = 8, num = 3, out_type = "tanh"):
        super().__init__()
        
        self.down_layer = nn.ModuleList(
            [
                nn.Sequential(
                    spectral_norm(nn.Conv2d(in_channel, hidden_channel, 3, 1, 1, bias = False)),
                    # nn.BatchNorm2d(hidden_channel),
                    nn.ReLU()
                )
            ] + \
            [
                # self._make_layer(
                Layer(
                    hidden_channel * (2 ** i),
                    hidden_channel * (2 ** (i + 1)),
                    "down"
                ) for i in range(num)
            ]
        )
        # self.bottle_neck = self._make_layer(hidden_channel * (2 ** (num)) , hidden_channel * (2 ** (num)))
        self.bottle_neck = Layer(hidden_channel * (2 ** (num)) , hidden_channel * (2 ** (num)))
        
        self.up_layer = nn.ModuleList(
            [
                # self._make_layer(
                Layer(
                    hidden_channel * 2 ** min(num, num - i + 1),
                    hidden_channel * (2 ** (num - 1 - i)),
                    "up"
                ) for i in range(num)
            ] + [
                nn.Sequential(
                    spectral_norm(nn.Conv2d(hidden_channel * 2, hidden_channel, 3, 1, 1, bias = False)),
                    # nn.BatchNorm2d(hidden_channel),
                    nn.ReLU(),
                    spectral_norm(nn.Conv2d(hidden_channel, out_channel, 3, 1, 1, bias = False)),
                )
            ]
        )
        assert out_type in ["tanh", "sigmoid"]
        self.out_layer = nn.Tanh() if out_type == "tanh" else nn.Sigmoid()
        
    def forward(self, x):
        down_out = []
        for layer in self.down_layer:
            x = layer(x)
            down_out.append(x)
        x = self.bottle_neck(x)
        for i, layer in enumerate(self.up_layer):
            if i == 0:
                x = layer(x)
            else:
                x = layer(torch.cat([x, down_out[-i-1]], dim = 1))        
        return self.out_layer(x)
    
class DuelDomainImageRegistration(ImageRegistration):
    def __init__(self, config):
        super().__init__(config)
        
        self.domain_type = config["Domain_type"]
        assert isinstance(self.domain_type, str) and len(self.domain_type) == 2
        assert self.domain_type in ["bf", "fb"]
        
        # config["Backbone"]["in_channel"] = config["Backbone"]["in_channel"] * 2
        
        self.unetbf = UNet_Trans(config["In_channel"], config["In_channel"], out_type="sigmoid")
        
        self.unetfr = UNet_Trans(config["In_channel"], config["In_channel"], out_type="tanh")
        
    def forward(self, data_dict):
        
        return_mode = self.training or "Template_image_bf" in data_dict.keys()
        
        final_mode = "Template_image_bf" not in data_dict.keys() and "Template_image" in data_dict.keys()
        
        if self.domain_type == "bf":
            gen_fr = self.unetbf(data_dict["Template_image_bf" if not final_mode else "Template_image"])
            image_init = torch.cat([data_dict["Template_image_bf" if not final_mode else "Template_image"], gen_fr], dim = 1)
            gen_bf = self.unetfr(data_dict["Target_image_fr" if not final_mode else "Target_image"])
            image_term = torch.cat([gen_bf, data_dict["Target_image_fr" if not final_mode else "Target_image"]], dim = 1)
        else:
            gen_bf = self.unetfr(data_dict["Template_image_fr" if not final_mode else "Template_image"])
            image_term = torch.cat([gen_bf, data_dict["Template_image_fr" if not final_mode else "Template_image"]], dim = 1)
            gen_fr = self.unetbf(data_dict["Target_image_bf" if not final_mode else "Target_image"])
            image_init = torch.cat([data_dict["Target_image_bf" if not final_mode else "Target_image"], gen_fr], dim = 1)
        
        if return_mode:
            
            if self.domain_type == "bf":
                gen_bf_aff = self.unetfr(data_dict["Template_image_fr"])
                gen_fr_aff = self.unetbf(data_dict["Target_image_bf"])
                
                sim_template = (
                    F.l1_loss(data_dict["Template_image_fr"], gen_fr, reduction="none") * \
                    data_dict["Template_mask"].float().mul(0.8).add(0.2)
                    ).mean() + \
                    F.l1_loss(data_dict["Template_image_bf"], gen_bf_aff) * 0.5 + \
                    ssim_loss(data_dict["Template_image_fr"], gen_fr, (0,1)) * 4 + \
                    ssim_loss(data_dict["Template_image_bf"], gen_bf_aff) * 4
                    
                sim_target = (
                    F.l1_loss(data_dict["Target_image_fr"], gen_fr_aff, reduction="none") * \
                    data_dict["Target_mask"].float().mul(0.8).add(0.2)
                    ).mean() + \
                    F.l1_loss(data_dict["Target_image_bf"], gen_bf) * 0.5 + \
                    ssim_loss(data_dict["Target_image_fr"], gen_fr_aff, (0,1)) * 4 + \
                    ssim_loss(data_dict["Target_image_bf"], gen_bf) * 4
                
            else:
                gen_fr_aff = self.unetbf(data_dict["Template_image_bf"])
                gen_bf_aff = self.unetfr(data_dict["Target_image_fr"])
                
                sim_template = (
                    F.l1_loss(data_dict["Template_image_fr"], gen_fr_aff, reduction="none") * \
                    data_dict["Template_mask"].float().mul(0.8).add(0.2)
                    ).mean() + \
                    F.l1_loss(data_dict["Template_image_bf"], gen_bf) * 0.5 + \
                    ssim_loss(data_dict["Template_image_fr"], gen_fr_aff, (0,1)) * 4 + \
                    ssim_loss(data_dict["Template_image_bf"], gen_bf) * 4
                    
                sim_target = (
                    F.l1_loss(data_dict["Target_image_fr"], gen_fr, reduction="none") * \
                    data_dict["Target_mask"].float().mul(0.8).add(0.2)
                    ).mean() + \
                    F.l1_loss(data_dict["Target_image_bf"], gen_bf_aff) * 0.5 + \
                    ssim_loss(data_dict["Target_image_fr"], gen_fr, (0,1)) * 4 + \
                    ssim_loss(data_dict["Target_image_bf"], gen_bf_aff) * 4               
            
            trans_similarity = sim_template + sim_target
        
        feat_init = self.backbone(image_init)
        feat_term = self.backbone(image_term)
        
        feat_init = rearrange(self.pos_encoding(feat_init), 'n c1 c2 h w -> n (h w) c1 c2')
        feat_term = rearrange(self.pos_encoding(feat_term), 'n c1 c2 h w -> n (h w) c1 c2')
            
        if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            mask_init = rearrange(data_dict['Template_square_mask'], 'n h w -> n (h w)')
            mask_term = rearrange(data_dict['Target_square_mask'], 'n h w -> n (h w)')
        else:
            mask_init = None
            mask_term = None
            
        feat_init, feat_term = self.feature_transformer(feat_init, feat_term, mask_init, mask_term)
        
        Ch = feat_init.shape[-1]
        
        feat_init = feat_init.div(Ch**0.5)
        feat_term = feat_term.div(Ch**0.5)
        
        matching_map = torch.einsum("nlec, nsec -> nlse", feat_init, feat_term)
        
        matching_map = self.map_projection(matching_map)
        
        score_map = matching_map[...,0]
            
        if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            score_map.masked_fill_(~(torch.einsum('ij, ik -> ijk', mask_init, mask_term)).bool(), -1e9)
            
        if self.matching_algorithm == 'sinkhorn':
            score_map = self.log_optimal_transport(score_map, self.bin_score, self.skh_iter) # shape: (Bs L+1 S+1) 
        else:
            matching_map.mul_(10)
            score_map = F.softmax(score_map, 1) * F.softmax(score_map, 2) # shape: (Bs L S)
            
        angle_map = F.normalize(matching_map[...,1:3], dim=-1)
        scale_map = matching_map[...,3:4]# if matching_map.size(-1) == 6 else None
        trans_map = matching_map[...,-2:]
        
        if return_mode:
            return {'score_map': score_map, 
                    'angle_map': angle_map,
                    'scale_map': scale_map,
                    'trans_map': trans_map}, trans_similarity             
        return {'score_map': score_map, 
                'angle_map': angle_map,
                'scale_map': scale_map,
                'trans_map': trans_map}
    
class GANDomainTransImageRegistration(ImageRegistration):
    def __init__(self, config, param_path = None, param_freeze = True):
        super().__init__(config)
        
        self.domain_type = config["Domain_type"]
        assert isinstance(self.domain_type, str) and len(self.domain_type) == 2
        assert self.domain_type in ["bf", "fb"]
        
        # config["Backbone"]["in_channel"] = config["Backbone"]["in_channel"] * 2
        
        self.unetbf = Generator(**config["Xacgan"])#(config["In_channel"], config["In_channel"], "00010", False, 32, down_step = 4)
        
        self.unetfr = Generator(**config["Xacgan"])#(config["In_channel"], config["In_channel"], "00010", False, 32, down_step = 4)
        
        if param_path is not None:
            self.unetbf.load_state_dict(torch.load(param_path)["bf"])
            self.unetfr.load_state_dict(torch.load(param_path)["fr"])
            
            if param_freeze:
                for param in chain(self.unetbf.parameters(), self.unetfr.parameters()):
                    param.requires_grad = False
        self.loss_pass = not param_freeze
            
    def forward(self, data_dict, return_loss = True):

        final_mode = "Template_image_bf" not in data_dict.keys() and "Template_image" in data_dict.keys()
        
        if self.domain_type == "bf":
            gen_fr = self.unetbf(data_dict["Template_image_bf" if not final_mode else "Template_image"])["image"]
            image_init = torch.cat([data_dict["Template_image_bf" if not final_mode else "Template_image"], gen_fr], dim = 1)
            gen_bf = self.unetfr(data_dict["Target_image_fr" if not final_mode else "Target_image"])["image"]
            image_term = torch.cat([gen_bf, data_dict["Target_image_fr" if not final_mode else "Target_image"]], dim = 1)
        else:
            gen_bf = self.unetfr(data_dict["Template_image_fr" if not final_mode else "Template_image"])["image"]
            image_term = torch.cat([gen_bf, data_dict["Template_image_fr" if not final_mode else "Template_image"]], dim = 1)
            gen_fr = self.unetbf(data_dict["Target_image_bf" if not final_mode else "Target_image"])["image"]
            image_init = torch.cat([data_dict["Target_image_bf" if not final_mode else "Target_image"], gen_fr], dim = 1)
            
        if (self.training or return_loss) and self.loss_pass:
            if self.domain_type == "bf":
                gen_fr_aff = self.unetbf(data_dict["Target_image_bf"])["image"]
                gen_bf_aff = self.unetfr(data_dict["Template_image_fr"])["image"]
            else:
                gen_fr_aff = self.unetbf(data_dict["Template_image_bf"])["image"]
                gen_bf_aff = self.unetfr(data_dict["Target_image_fr"])["image"]
                
            similarity_loss = \
                ((F.l1_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf, reduction = "none") + \
                F.l1_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff, reduction = "none")) * \
                 data_dict["Template_mask"].float().mul(9).add(1)).mean() * 0.2 + \
                ((F.l1_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff, reduction = "none") + \
                F.l1_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr, reduction = "none")) * \
                 data_dict["Target_mask"].float().mul(9).add(1)).mean() * 0.2 + \
                ssim_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff) * 0.8 + \
                ssim_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff) * 0.8 + \
                ssim_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf) * 0.8 + \
                ssim_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr) * 0.8                
            # 
            # similarity_loss = \
            #     F.l1_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff) * 0.2 + \
            #     (F.l1_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff, reduction = "none") * \
            #      data_dict["Template_mask"].float().mul(9).add(1)).mean() * 0.2 + \
            #     F.l1_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf) * 0.2 + \
            #     (F.l1_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr, reduction = "none") * \
            #      data_dict["Target_mask"].float().mul(9).add(1)).mean() * 0.2 + \
            #     ssim_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff) * 0.8 + \
            #     ssim_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff) * 0.8 + \
            #     ssim_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf) * 0.8 + \
            #     ssim_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr) * 0.8
        else:
            similarity_loss = None
            
        
        feat_init = self.backbone(image_init)
        feat_term = self.backbone(image_term)
        
        feat_init = rearrange(self.pos_encoding(feat_init), 'n c1 c2 h w -> n (h w) c1 c2')
        feat_term = rearrange(self.pos_encoding(feat_term), 'n c1 c2 h w -> n (h w) c1 c2')
            
        if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            mask_init = rearrange(data_dict['Template_square_mask'], 'n h w -> n (h w)')
            mask_term = rearrange(data_dict['Target_square_mask'], 'n h w -> n (h w)')
        else:
            mask_init = None
            mask_term = None
            
        feat_init, feat_term = self.feature_transformer(feat_init, feat_term, mask_init, mask_term)
        
        Ch = feat_init.shape[-1]
        
        feat_init = feat_init.div(Ch**0.5)
        feat_term = feat_term.div(Ch**0.5)
        
        matching_map = torch.einsum("nlec, nsec -> nlse", feat_init, feat_term)
        
        matching_map = self.map_projection(matching_map)
        
        score_map = matching_map[...,0]
            
        if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            score_map.masked_fill_(~(torch.einsum('ij, ik -> ijk', mask_init, mask_term)).bool(), -1e9)
            
        if self.matching_algorithm == 'sinkhorn':
            score_map = self.log_optimal_transport(score_map, self.bin_score, self.skh_iter) # shape: (Bs L+1 S+1) 
        else:
            matching_map.mul_(10)
            score_map = F.softmax(score_map, 1) * F.softmax(score_map, 2) # shape: (Bs L S)
            
        angle_map = F.normalize(matching_map[...,1:3], dim=-1)
        scale_map = matching_map[...,3:4]# if matching_map.size(-1) == 6 else None
        trans_map = matching_map[...,-2:]
        
        if similarity_loss is not None:
            return {'score_map': score_map, 
                    'angle_map': angle_map,
                    'scale_map': scale_map,
                    'trans_map': trans_map}, similarity_loss           
        return {'score_map': score_map, 
                'angle_map': angle_map,
                'scale_map': scale_map,
                'trans_map': trans_map}

class XaCGANDomainTransImageRegistration(nn.Module):
    def __init__(self, config, param_path = None, param_freeze = True):
        super().__init__()
        
        self.domain_type = config["Domain_type"]
        assert isinstance(self.domain_type, str) and len(self.domain_type) == 2
        assert self.domain_type in ["bf", "fb"]
        
        config["Backbone"]["up_progress"] =  True
        
        hidden_channel = config["Backbone"]["hidden_channel"] // config["Backbone"]["n_rotation"]
        
        self.backbone = Feature_Extraction_2(**config["Backbone"])
        
        self.pos_encoding = PositionEncodingSine(
            d_model = hidden_channel,
            # **config["max_shape"]
            max_shape=(256, 256)
        )
        
        self.feature_transformer_coarse = LocalFeatureTransformer(
            d_model = hidden_channel,
            nhead = 8,
            **config["Transformer"]
        )
        
        self.feature_transformer_fine = LocalFeatureTransformer(
            d_model = hidden_channel,
            nhead = 4,
            **config["Transformer"]
        )
        
        self.matching_algorithm = config["Matching_algorithm"]["Type"]
        alpha = config["Matching_algorithm"]["alpha"]
        self.bin_score_coarse = nn.Parameter(torch.tensor(float(alpha), requires_grad=True)) if self.matching_algorithm == 'sinkhorn' else None
        self.bin_score_fine = nn.Parameter(torch.tensor(float(alpha), requires_grad=True))
        self.skh_iter = config["Matching_algorithm"]["iters"]
        
        # if self.matching_algorithm == 'sinkhorn':
        #     alpha = config["Matching_algorithm"]["alpha"]
        #     self.bin_score_coarse = nn.Parameter(torch.tensor(float(alpha), requires_grad=True))
        #     self.bin_score_fine = nn.Parameter(torch.tensor(float(alpha), requires_grad=True))
        #     self.skh_iter = config["Matching_algorithm"]["iters"]

            
        self.map_projection_coarse = nn.Sequential(
            nn.Linear(8, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 6)
        )
        
        self.map_projection_fine = nn.Sequential(
            nn.Linear(4, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 6)
        )
        
        self.unetbf = Generator(**config["Xacgan"])
        
        self.unetfr = Generator(**config["Xacgan"])
        
        if param_path is not None:
            self.unetbf.load_state_dict(torch.load(param_path)["bf"])
            self.unetfr.load_state_dict(torch.load(param_path)["fr"])
            
            if param_freeze:
                for param in chain(self.unetbf.parameters(), self.unetfr.parameters()):
                    param.requires_grad = False
        self.loss_pass = not param_freeze
            
    def forward(self, data_dict, return_loss = True):

        final_mode = "Template_image_bf" not in data_dict.keys() and "Template_image" in data_dict.keys()
        
        if self.domain_type == "bf":
            gen_fr = self.unetbf(data_dict["Template_image_bf" if not final_mode else "Template_image"])["image"]
            image_init = torch.cat([data_dict["Template_image_bf" if not final_mode else "Template_image"], gen_fr], dim = 1)
            gen_bf = self.unetfr(data_dict["Target_image_fr" if not final_mode else "Target_image"])["image"]
            image_term = torch.cat([gen_bf, data_dict["Target_image_fr" if not final_mode else "Target_image"]], dim = 1)
        else:
            gen_bf = self.unetfr(data_dict["Template_image_fr" if not final_mode else "Template_image"])["image"]
            image_term = torch.cat([gen_bf, data_dict["Template_image_fr" if not final_mode else "Template_image"]], dim = 1)
            gen_fr = self.unetbf(data_dict["Target_image_bf" if not final_mode else "Target_image"])["image"]
            image_init = torch.cat([data_dict["Target_image_bf" if not final_mode else "Target_image"], gen_fr], dim = 1)
            
        if (self.training or return_loss) and self.loss_pass:
            if self.domain_type == "bf":
                gen_fr_aff = self.unetbf(data_dict["Target_image_bf"])["image"]
                gen_bf_aff = self.unetfr(data_dict["Template_image_fr"])["image"]
            else:
                gen_fr_aff = self.unetbf(data_dict["Template_image_bf"])["image"]
                gen_bf_aff = self.unetfr(data_dict["Target_image_fr"])["image"]
                
            similarity_loss = \
                ((F.l1_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf, reduction = "none") + \
                F.l1_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff, reduction = "none")) * \
                 data_dict["Template_mask"].float().mul(9).add(1)).mean() * 0.2 + \
                ((F.l1_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff, reduction = "none") + \
                F.l1_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr, reduction = "none")) * \
                 data_dict["Target_mask"].float().mul(9).add(1)).mean() * 0.2 + \
                ssim_loss(data_dict["Target_image_bf"], gen_bf if self.domain_type == "bf" else gen_bf_aff) * 0.8 + \
                ssim_loss(data_dict["Template_image_fr"], gen_fr if self.domain_type == "bf" else gen_fr_aff) * 0.8 + \
                ssim_loss(data_dict["Template_image_bf"], gen_bf_aff if self.domain_type == "bf" else gen_bf) * 0.8 + \
                ssim_loss(data_dict["Target_image_fr"], gen_fr_aff if self.domain_type == "bf" else gen_fr) * 0.8                

        else:
            similarity_loss = None
            
        
        feat_init_coarse, feat_init_fine = self.backbone(image_init)
        feat_term_coarse, feat_term_fine = self.backbone(image_term)
        
        output_dict = {}
        
        for name, feat_init, feat_term, feature_transformer, map_projection, bin_score in zip(
                ["coarse", "fine"],
                [feat_init_coarse, feat_init_fine], 
                [feat_term_coarse, feat_term_fine],
                [self.feature_transformer_coarse, self.feature_transformer_fine],
                [self.map_projection_coarse, self.map_projection_fine],
                [self.bin_score_coarse, self.bin_score_fine]
                ):
        
            feat_init = rearrange(self.pos_encoding(feat_init), 'n c1 c2 h w -> n (h w) c1 c2')
            feat_term = rearrange(self.pos_encoding(feat_term), 'n c1 c2 h w -> n (h w) c1 c2')
            
            # if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            #     mask_init = rearrange(data_dict['Template_square_mask'], 'n h w -> n (h w)')
            #     mask_term = rearrange(data_dict['Target_square_mask'], 'n h w -> n (h w)')
            # else:
            #     mask_init = None
            #     mask_term = None
            
            feat_init, feat_term = feature_transformer(feat_init, feat_term, None, None)# mask_init, mask_term)
        
            Ch = feat_init.shape[-1]
        
            feat_init = feat_init.div(Ch**0.5)
            feat_term = feat_term.div(Ch**0.5)
        
            matching_map = torch.einsum("nlec, nsec -> nlse", feat_init, feat_term)
        
            matching_map = map_projection(matching_map)
        
            score_map = matching_map[...,0]
                
            # if 'Template_square_mask' in data_dict and 'Target_square_mask' in data_dict:
            #     score_map.masked_fill_(~(torch.einsum('ij, ik -> ijk', mask_init, mask_term)).bool(), -1e9)
            
            # if self.matching_algorithm == 'sinkhorn':
            if bin_score is None:
                score_map.mul_(10)
                score_map = F.softmax(score_map, 1) * F.softmax(score_map, 2) # shape: (Bs L S)
            else:
                score_map = self.log_optimal_transport(score_map, bin_score, self.skh_iter) 
                # shape: (Bs L+1 S+1) 
                
                
            angle_map = F.normalize(matching_map[...,1:3], dim=-1)
            scale_map = matching_map[...,3:4]# if matching_map.size(-1) == 6 else None
            trans_map = matching_map[...,-2:]
            
            output_dict[name] = {
                
            'score_map': score_map, 
            'angle_map': angle_map,
            'scale_map': scale_map,
            'trans_map': trans_map
            }
        
        if similarity_loss is not None:
            return output_dict, similarity_loss           
        return output_dict
    
    def log_optimal_transport(self, scores, alpha, iters=3):
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)
    
        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)
    
        couplings = torch.cat([torch.cat([scores, bins0], -1),
                               torch.cat([bins1, alpha], -1)], 1)
    
        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    
        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z.exp()
    
    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters):
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)
