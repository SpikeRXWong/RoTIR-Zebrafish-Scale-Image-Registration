# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:10:58 2023

@author: rw17789
"""

import torch
import torch.nn as nn
# from einops.einops import rearrange, reduce, repeat

class Matching_Loss(nn.Module):
    def __init__(self, weight, score_weight, matching_type = 'sinkhorn', loss_type = None):
        super().__init__()

        self.weight = weight if weight is not None else \
                            {
                                "Score_map": 1,
                                "Angle": 1,
                                "Scale": 1,
                                "Translation": 1,
                            }
        self.score_weight = score_weight if score_weight is not None else \
                            {
                                "Positive_loss": 20,
                                # "Negative_near_loss": 10,
                                "Negative_mask_loss": 10,
                                "Negative_back_loss": 1,
                            }
        
        for k in ["Score_map", "Angle", "Scale", "Translation"]:
            assert k in self.weight, "{} weight is not in loss weight".format(k)
        for k in ["Positive_loss", "Negative_mask_loss", "Negative_back_loss"]:
            assert k in self.score_weight, "{} weight is not in score loss weight".format(k)
        
        self.matching_algorithm = matching_type

        if loss_type in ["L1", "MAE"]:
            self.loss_func = nn.L1Loss(reduction='none')

        elif loss_type in ["L2", "MSE"]:
            self.loss_func = nn.MSELoss(reduction='none')

        else:
            self.loss_func = nn.SmoothL1Loss(reduction='none')

    
    def forward(self, score_map, angle_map, scale_map, trans_map, data_dict): #gt_matching_map, gt_matrix):
        gt_score_map = data_dict["Matching_map"][:,0,...]
        gt_trans_map = data_dict["Matching_map"][:,1:,...].permute(0,2,3,1)
        
        gt_cos_sin_scale = torch.cat([
            torch.cos(data_dict["Transformation_Rotation_Angle"]),
            torch.sin(data_dict["Transformation_Rotation_Angle"]),
            torch.log(data_dict["Transformation_Scale"])
            ], dim = -1)
        
        if "Template_square_mask" in data_dict and "Target_square_mask" in data_dict:
            focus_mask = {}
            focus_mask["mask_init"] = data_dict["Template_square_mask"].flatten(1)
            focus_mask["mask_term"] = data_dict["Target_square_mask"].flatten(1)
                    
        else:
            focus_mask = None
        
        loss_score_dict = self.compute_score_loss(score_map, gt_score_map, focus_mask)
        loss_score = 0
        for k in loss_score_dict:
            if not loss_score_dict[k].isnan():
                loss_score = loss_score + self.score_weight[k] * loss_score_dict[k]
                
        gt_cos_sin_scale = repeat(gt_cos_sin_scale, "i j -> i h w j", h = angle_map.size(1), w = angle_map.size(2))        
        loss_angle = self.loss_func(angle_map, gt_cos_sin_scale[...,:2])[gt_score_map.bool()].mean()
        # loss_scale = self.loss_func(scale_map, gt_cos_sin_scale[...,-1:])[gt_score_map.bool()].mean()
        
        if self.weight["Scale"] == 0:
            loss_scale = torch.tensor(0)
        else:
            loss_scale = self.loss_func(scale_map, gt_cos_sin_scale[...,-1:])[gt_score_map.bool()].mean()
        
        if self.weight["Translation"] == 0:
            loss_trans = torch.tensor(0)
        else:
            loss_trans = self.loss_func(trans_map, gt_trans_map)[gt_score_map.bool()].mean()
        
        loss = loss_score * self.weight["Score_map"] + \
               loss_angle * self.weight["Angle"] + \
               loss_scale * self.weight["Scale"] + \
               loss_trans * self.weight["Translation"]
        loss_dict = {}
        loss_dict.update(loss_score_dict)
        loss_dict.update({
            "Angle": loss_angle,
            "Scale": loss_scale,
            "Translation": loss_trans,
        })
        for k,v in loss_dict.items():
            loss_dict[k] = v.item()
        
        return loss, loss_dict
            
    def compute_score_loss(self, conf, conf_gt, focus_mask):
        pos_mask = conf_gt == 1 
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        
        if self.matching_algorithm == 'sinkhorn':
            pos_conf = conf[:,:-1, :-1][pos_mask]
            loss_pos = - 0.25 * torch.pow(1 - pos_conf, 2.0) * pos_conf.log()
            
            if focus_mask is not None:
                neg0m = torch.logical_and(conf_gt.sum(-1) == 0, focus_mask["mask_init"])
                neg1m = torch.logical_and(conf_gt.sum(1) == 0, focus_mask["mask_term"])
                neg_conf_mask = torch.cat([conf[:, :-1, -1][neg0m], conf[:,-1,:-1][neg1m]], 0)
            
                loss_neg_mask = - 0.25 * torch.pow(1 - neg_conf_mask, 2.0) * neg_conf_mask.log()
               
                neg0b = torch.logical_not(focus_mask["mask_init"])
                neg1b = torch.logical_not(focus_mask["mask_term"])
                neg_conf_back = torch.cat([conf[:, :-1, -1][neg0b], conf[:,-1,:-1][neg1b]], 0)                
                
                loss_neg_back = - 0.25 * torch.pow(1 - neg_conf_back, 2.0) * neg_conf_back.log()
            else:
                neg0 = conf_gt.sum(-1) == 0
                neg1 = conf_gt.sum(1) == 0
                neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:,-1,:-1][neg1]], 0)
                loss_neg = - 0.25 * torch.pow(1 - neg_conf, 2.0) * neg_conf.log()
            
        else:
            neg_mask_all = conf_gt == 0
            loss_pos = - torch.log(conf[pos_mask])
            if focus_mask is not None:
                neg_mask = focus_mask["mask_init"][...,None] * focus_mask["mask_term"][:,None]
                neg_back = torch.logical_not(neg_mask)
                neg_mask = torch.logical_and(neg_mask, neg_mask_all)
                
                loss_neg_mask = - torch.log(1 - conf[neg_mask])
                loss_neg_back = - torch.log(1 - conf[neg_back])
            else:
                loss_neg = - torch.log(1 - conf[neg_mask_all])
                
        # loss_neg = 20 * loss_neg_high.mean() + 10 * loss_neg_mid.mean() + loss_neg_low.mean()
        
        loss_dict = {
            "Positive_loss": loss_pos.mean(),
            "Negative_mask_loss": loss_neg_mask.mean() if focus_mask is not None else loss_neg.mean(),
            "Negative_back_loss": loss_neg_back.mean() if focus_mask is not None else torch.tensor(float('nan')),
        }
        
        return loss_dict#loss_pos.mean() * 50 + loss_neg
    