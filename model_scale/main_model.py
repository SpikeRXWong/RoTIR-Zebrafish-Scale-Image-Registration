# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:07:16 2023

@author: rw17789
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import Feature_Extraction
from .loftr_transformer import LocalFeatureTransformer, PositionEncodingSine
from einops.einops import rearrange

class ImageRegistration(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        hidden_channel = config["Backbone"]["hidden_channel"] // config["Backbone"]["n_rotation"]
        
        self.backbone = Feature_Extraction(**config["Backbone"])
        
        self.pos_encoding = PositionEncodingSine(
            d_model = hidden_channel,
            # **config["max_shape"]
            max_shape=(256, 256)
        )
        
        self.feature_transformer = LocalFeatureTransformer(
            d_model = hidden_channel,
            **config["Transformer"]
        )
        
        self.matching_algorithm = config["Matching_algorithm"]["Type"]
        if self.matching_algorithm == 'sinkhorn':
            alpha = config["Matching_algorithm"]["alpha"]
            self.bin_score = nn.Parameter(torch.tensor(float(alpha), requires_grad=True))
            self.skh_iter = config["Matching_algorithm"]["iters"]
            
        layer_depth = config["Transformer"]["nhead"]
            
        self.map_projection = nn.Sequential(
            nn.Linear(layer_depth, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 64),
            nn.Dropout(0.25),
            nn.Linear(64, 6)
        )
        
    def forward(self, data_dict):
        image_init = data_dict["Template_image"]
        image_term = data_dict["Target_image"]
        
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
         
        return {'score_map': score_map, 
                'angle_map': angle_map,
                'scale_map': scale_map,
                'trans_map': trans_map}
    
            
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
            
          