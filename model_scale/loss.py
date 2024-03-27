# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 20:10:58 2023

@author: rw17789
"""

import torch
import torch.nn as nn
from einops.einops import rearrange, reduce, repeat

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
    
class Duel_Matching_loss(nn.Module):
    def __init__(self, weight_coarse, weight_fine, score_weight_coarse, score_weight_fine, matching_type = 'sinkhorn', loss_type = None):
        assert matching_type == 'sinkhorn', "Use sinkhorn for loss calculation!"
        super().__init__()
        # super().__init__(weight_fine, score_weight_fine, matching_type, loss_type)
        
        self.weight_coarse = weight_coarse if weight_coarse is not None else \
                            {
                                "Score_map": 1,
                                "Angle": 1,
                                "Scale": 1,
                                "Translation": 1,
                            }                            
        self.weight_fine = weight_fine if weight_fine is not None else \
                            {
                                "Score_map": 1,
                                "Angle": 1,
                                "Scale": 1,
                                "Translation": 1,
                            }
        self.score_weight_coarse = score_weight_coarse if score_weight_coarse is not None else \
                            {
                                "Positive_loss": 20,
                                "Positive_edge_loss": 10,
                                "Negative_near_loss": 10,
                                "Negative_mask_loss": 10,
                                "Negative_back_loss": 1,
                            }
                            
        self.score_weight_fine = score_weight_fine if score_weight_fine is not None else \
                            {
                                "Positive_loss": 20,
                                "Positive_edge_loss": 10,
                                "Negative_near_loss": 10,
                                "Negative_mask_loss": 10,
                                "Negative_back_loss": 1,
                            }
        for k in ["Score_map", "Angle", "Scale", "Translation"]:
            assert k in self.weight_coarse, "{} weight is not in loss weight coarse".format(k)
            assert k in self.weight_fine, "{} weight is not in loss weight fine".format(k)
        # for k in ["Positive_loss", "Negative_edge_loss", "Positive_edge_loss"]:
        #     assert k in self.score_weight_coarse, "{} weight is not in score loss weight coarse".format(k)
        for k in ["Positive_loss", "Positive_edge_loss", "Negative_near_loss", "Negative_mask_loss", "Negative_back_loss"]:
            assert k in self.score_weight_coarse, "{} weight is not in score loss weight coarse".format(k)
            assert k in self.score_weight_fine, "{} weight is not in score loss weight fine".format(k)
            
        self.matching_algorithm = matching_type

        if loss_type in ["L1", "MAE"]:
            self.loss_func = nn.L1Loss(reduction='none')

        elif loss_type in ["L2", "MSE"]:
            self.loss_func = nn.MSELoss(reduction='none')

        else:
            self.loss_func = nn.SmoothL1Loss(reduction='none')
            
    # def forward(self, score_map_8, score_map_16, angle_map, scale_map, trans_map, data_dict, **kwargs): 
    def forward(self, output_dict, data_dict, t = 1):#, **kwargs): 
        self.t = t
        
        # coarse_score_map = data_dict["Matching_map_8"][:,0,...]
        # focus_mask = {}
        # focus_mask["mask_init"] = self._enlarge_mask(coarse_score_map.sum(-1))
        # focus_mask["mask_term"] = self._enlarge_mask(coarse_score_map.sum(-2))
        
        total_loss_dict = {}
        loss = 0
        
        for level, name, weight, score_weight in zip(
                ["coarse", "fine"], 
                ["Matching_map_8", "Matching_map_16"],
                [self.weight_coarse, self.weight_fine],
                [self.score_weight_coarse, self.score_weight_fine]
                ):
            
            gt_score_map = data_dict[name][:,0,...]
            gt_trans_map = data_dict[name][:,1:,...].permute(0,2,3,1)
            
            focus_mask = {
                "mask_init": data_dict["Matching_map_8"][:,0,...].sum(-1),
                "mask_term": data_dict["Matching_map_8"][:,0,...].sum(-2),
                }
            if level == "fine":
                for k, v in focus_mask.items():
                    focus_mask[k] = self._enlarge_mask(v)
            
            gt_cos_sin_scale = torch.cat([
                torch.cos(data_dict["Transformation_Rotation_Angle"]),
                torch.sin(data_dict["Transformation_Rotation_Angle"]),
                torch.log(data_dict["Transformation_Scale"])
                ], dim = -1)
            
            loss_score_dict = self.compute_score_loss(
                output_dict[level]['score_map'], 
                gt_score_map, 
                focus_mask
                )
            loss_score = 0
            ####
            # print(score_weight.keys())
            # print(loss_score_dict.keys())
            ####
            for k in loss_score_dict:
                if not loss_score_dict[k].isnan():
                    loss_score = loss_score + score_weight[k] * loss_score_dict[k]
            
            if weight["Angle"] == 0:
                loss_angle =  torch.tensor(0)
            else:
                gt_cos_sin_scale = repeat(gt_cos_sin_scale, "i j -> i h w j", h = output_dict[level]['angle_map'].size(1), w = output_dict[level]['angle_map'].size(2))        
                loss_angle = self.loss_func(
                    output_dict[level]['angle_map'], 
                    gt_cos_sin_scale[...,:2]
                    )[gt_score_map.bool()].mean()
            
            if weight["Scale"] == 0:
                loss_scale = torch.tensor(0)
            else:
                loss_scale = self.loss_func(output_dict[level]['scale_map'], gt_cos_sin_scale[...,-1:])[gt_score_map.bool()].mean()
            
            if weight["Translation"] == 0:
                loss_trans = torch.tensor(0)
            else:
                loss_trans = self.loss_func(output_dict[level]['trans_map'], gt_trans_map)[gt_score_map.bool()].mean()
            
            loss = loss + \
                   loss_score * weight["Score_map"] + \
                   loss_angle * weight["Angle"] + \
                   loss_scale * weight["Scale"] + \
                   loss_trans * weight["Translation"]
                   
            loss_dict = {}
            loss_dict.update(loss_score_dict)
            loss_dict.update({
                "Angle": loss_angle,
                "Scale": loss_scale,
                "Translation": loss_trans,
            })
            for k,v in loss_dict.items():
                loss_dict[k] = v.item()
            total_loss_dict[level] = loss_dict
        return loss, total_loss_dict
            
    def compute_score_loss(self, conf, conf_gt, focus_mask):
        pos_mask = conf_gt == 1 
        conf = torch.clamp(conf, 1e-6, 1 - 1e-6)
        
        if self.matching_algorithm == 'sinkhorn':
            # Positive_loss
            pos_conf = conf[:,:-1, :-1][pos_mask]
            loss_pos = - 0.25 * torch.pow(1 - pos_conf, 2.0) * pos_conf.log()
            
            # if focus_mask is not None:
            # Negative_near_loss
            neg_near = torch.logical_and(
                focus_mask["mask_init"][..., None] * focus_mask["mask_term"][:, None],
                torch.logical_not(pos_mask)
            )
            neg_near_conf = conf[:,:-1,:-1][neg_near]
            ## different choices of losses for negative near positive area loss
            if self.t == 0:
                loss_neg_near = - 0.25 * torch.pow(neg_near_conf, 2.0) * (1 - neg_near_conf).log()
            elif self.t == 1:
                loss_neg_near = 4 * torch.pow(neg_near_conf, -0.5) * (1 + neg_near_conf).log() # 0.25 * torch.pow(1 - neg_near_conf, 2.0) * neg_near_conf.log()#- 0.25 * torch.pow(neg_near_conf, 2.0) * (1 - neg_near_conf).log()
            elif self.t == 2:
                loss_neg_near = - 4 * (1 - neg_near_conf).log()
            elif self.t == 3:
                loss_neg_near = 4 * torch.pow(1 - neg_near_conf.log(), -1)
            else:
                loss_neg_near = torch.tensor(0)
            
            # Negative_mask_loss
            neg0m = torch.logical_and(conf_gt.sum(-1) == 0, focus_mask["mask_init"])
            neg1m = torch.logical_and(conf_gt.sum(1) == 0, focus_mask["mask_term"])
            neg_conf_mask = torch.cat([conf[:, :-1, -1][neg0m], conf[:,-1,:-1][neg1m]], 0)
            
            loss_neg_mask = - 0.25 * torch.pow(1 - neg_conf_mask, 2.0) * neg_conf_mask.log()
            
            # Negative_back_loss
            neg0b = torch.logical_not(focus_mask["mask_init"])
            neg1b = torch.logical_not(focus_mask["mask_term"])
            neg_conf_back = torch.cat([conf[:, :-1, -1][neg0b], conf[:,-1,:-1][neg1b]], 0)                
            
            loss_neg_back = - 0.25 * torch.pow(1 - neg_conf_back, 2.0) * neg_conf_back.log()

            # else:
            #     neg0 = conf_gt.sum(-1) == 0
            #     neg1 = conf_gt.sum(1) == 0
            #     neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:,-1,:-1][neg1]], 0)
            #     loss_neg = - 0.25 * torch.pow(1 - neg_conf, 2.0) * neg_conf.log()
            
            # Positive_edge_loss
            pos0 = conf_gt.sum(-1) > 0
            pos1 = conf_gt.sum(1) > 0
            pos_conf = torch.cat([conf[:, :-1, -1][pos0], conf[:,-1,:-1][pos1]], 0)
            loss_edge_pos = - 0.25 * torch.pow(pos_conf, 2.0) * (1 - pos_conf).log()
            
        else:
            raise Exception("Use Sinkhorn")
            # neg_mask_all = conf_gt == 0
            # loss_pos = - torch.log(conf[pos_mask])
            # if focus_mask is not None:
            #     neg_mask = focus_mask["mask_init"][...,None] * focus_mask["mask_term"][:,None]
            #     neg_back = torch.logical_not(neg_mask)
            #     neg_mask = torch.logical_and(neg_mask, neg_mask_all)
                
            #     loss_neg_mask = - torch.log(1 - conf[neg_mask])
            #     loss_neg_back = - torch.log(1 - conf[neg_back])
            # else:
            #     loss_neg = - torch.log(1 - conf[neg_mask_all])
                
        # loss_neg = 20 * loss_neg_high.mean() + 10 * loss_neg_mid.mean() + loss_neg_low.mean()
        
        # if focus_mask is not None:
        loss_dict = {
            "Positive_loss": loss_pos.mean(),
            "Positive_edge_loss": loss_edge_pos.mean(),
            "Negative_near_loss": loss_neg_near.mean(),
            "Negative_mask_loss": loss_neg_mask.mean(),
            "Negative_back_loss": loss_neg_back.mean(),
        }
        # else:
        #     loss_dict = {
        #         "Positive_loss": loss_pos.mean(),
        #         "Negative_edge_loss": loss_neg.mean(),
        #         "Positive_edge_loss": loss_edge_pos.mean(),
        #     }
        
        return loss_dict#loss_pos.mean() * 50 + loss_neg
    
    def _enlarge_mask(self, mask_line, scale = 2):
        L1 = mask_line.shape[-1] ** 0.5
        H = torch.zeros_like(mask_line).repeat(1, scale **2)
        batch, position = torch.where(mask_line)
        for i, p in zip(batch, position):
            h, w = p // L1, p % L1
            for j in range(scale ** 2):
                j1 = j // scale
                j2 = j % scale
                H[i, int((h * scale + j1) * (L1 * scale) + (w * scale + j2))] = 1
        return H
    
class Parallel_Matching_Loss(Matching_Loss):
    def __init__(self, weight, score_weight, matching_type = 'sinkhorn', loss_type = None):
        super().__init__(weight, score_weight, matching_type, loss_type)
        # super().__init__()

        # self.weight = weight if weight is not None else \
        #                     {
        #                         "Score_map": 1,
        #                         "Angle": 1,
        #                         "Scale": 1,
        #                         "Translation": 1,
        #                     }
        # self.score_weight = score_weight if score_weight is not None else \
        #                     {
        #                         "Positive_loss": 20,
        #                         "Negative_mask_loss": 10,
        #                         "Negative_back_loss": 1,
        #                     }
        
        # for k in ["Score_map", "Angle", "Scale", "Translation"]:
        #     assert k in self.weight, "{} weight is not in loss weight".format(k)
        # for k in ["Positive_loss", "Negative_mask_loss", "Negative_back_loss"]:
        #     assert k in self.score_weight, "{} weight is not in score loss weight".format(k)
        
        # self.matching_algorithm = matching_type

        # if loss_type in ["L1", "MAE"]:
        #     self.loss_func = nn.L1Loss(reduction='none')

        # elif loss_type in ["L2", "MSE"]:
        #     self.loss_func = nn.MSELoss(reduction='none')

        # else:
        #     self.loss_func = nn.SmoothL1Loss(reduction='none')

    
    def forward(self, score_map, angle_map, scale_map, trans_map, data_dict): #gt_matching_map, gt_matrix):
        
        if self.matching_algorithm == 'sinkhorn':
            assert score_map.size(1) == angle_map.size(1) + 1
            Mp = "Matching_map_16"
        else:
            assert score_map.size(1) == angle_map.size(1)
            Mp = "Matching_map_8"
        
        gt_score_map = data_dict[Mp][:,0,...]
        gt_trans_map = data_dict[Mp][:,1:,...].permute(0,2,3,1)
        
        gt_cos_sin_scale = torch.cat([
            torch.cos(data_dict["Transformation_Rotation_Angle"]),
            torch.sin(data_dict["Transformation_Rotation_Angle"]),
            torch.log(data_dict["Transformation_Scale"])
            ], dim = -1)
        
        focus_mask = {
            "mask_init": data_dict["Matching_map_8"][:,0,...].sum(-1),
            "mask_term": data_dict["Matching_map_8"][:,0,...].sum(-2),
            }
        if self.matching_algorithm == 'sinkhorn':
            for k, v in focus_mask.items():
                focus_mask[k] = self._enlarge_mask(v)
                         
        loss_score_dict = self.compute_score_loss(score_map, gt_score_map, focus_mask)

        loss_score = 0
        for k in loss_score_dict:
            if not loss_score_dict[k].isnan():
                loss_score = loss_score + self.score_weight[k] * loss_score_dict[k]
        
        if self.weight["Angle"] == 0:
            loss_angle = torch.tensor(0)
        else:
            gt_cos_sin_scale = repeat(gt_cos_sin_scale, "i j -> i h w j", h = angle_map.size(1), w = angle_map.size(2))        
            # loss_angle = self.loss_func(angle_map, gt_cos_sin_scale[...,:2])[gt_score_map.bool()].mean()
            loss_angle = torch.nn.functional.mse_loss(angle_map, gt_cos_sin_scale[...,:2], reduction='none')[gt_score_map.bool()].mean()

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
            
            # if focus_mask is not None:
            neg0m = torch.logical_and(conf_gt.sum(-1) == 0, focus_mask["mask_init"])
            neg1m = torch.logical_and(conf_gt.sum(1) == 0, focus_mask["mask_term"])
            neg_conf_mask = torch.cat([conf[:, :-1, -1][neg0m], conf[:,-1,:-1][neg1m]], 0)
        
            loss_neg_mask = - 0.25 * torch.pow(1 - neg_conf_mask, 2.0) * neg_conf_mask.log()
           
            neg0b = torch.logical_not(focus_mask["mask_init"])
            neg1b = torch.logical_not(focus_mask["mask_term"])
            neg_conf_back = torch.cat([conf[:, :-1, -1][neg0b], conf[:,-1,:-1][neg1b]], 0)                
            
            loss_neg_back = - 0.25 * torch.pow(1 - neg_conf_back, 2.0) * neg_conf_back.log()
            # else:
            #     neg0 = conf_gt.sum(-1) == 0
            #     neg1 = conf_gt.sum(1) == 0
            #     neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:,-1,:-1][neg1]], 0)
            #     loss_neg = - 0.25 * torch.pow(1 - neg_conf, 2.0) * neg_conf.log()
            
        else:
            pos_conf = conf[pos_mask]
            weight_mask = self._create_weight_map(pos_mask)[pos_mask]
            loss_pos = - (1 + (pos_conf - weight_mask)).log() - (1 - (pos_conf - weight_mask)).log()
            loss_pos = loss_pos * (1 / weight_mask)
            
            neg_mask_all = conf_gt == 0

            # if focus_mask is not None:
            neg_mask = focus_mask["mask_init"][...,None] * focus_mask["mask_term"][:,None]
            neg_back = torch.logical_not(neg_mask)
            neg_mask = torch.logical_and(neg_mask, neg_mask_all)
            
            loss_neg_mask = - 0.25 * torch.log(1 - conf[neg_mask])
            loss_neg_back = - 0.25 * torch.log(1 - conf[neg_back])
            # else:
            #     loss_neg = - torch.log(1 - conf[neg_mask_all])
                
        # loss_neg = 20 * loss_neg_high.mean() + 10 * loss_neg_mid.mean() + loss_neg_low.mean()
        
        loss_dict = {
            "Positive_loss": loss_pos.mean(),
            "Negative_mask_loss": loss_neg_mask.mean(),# if focus_mask is not None else loss_neg.mean(),
            "Negative_back_loss": loss_neg_back.mean(),# if focus_mask is not None else torch.tensor(float('nan')),
        }
        
        return loss_dict

    def _create_weight_map(self, mask):
        c, h, w = torch.where(mask)
        weight_mask = torch.zeros_like(mask).float()
        for cc, hh, ww in zip(c, h, w):
            weight_mask[cc, hh, ww] = 1 / (mask[cc, :, ww].sum() * mask[cc, hh, :].sum())
        return weight_mask#[mask]
    
    def _enlarge_mask(self, mask_line, scale = 2):
        L1 = mask_line.shape[-1] ** 0.5
        H = torch.zeros_like(mask_line).repeat(1, scale **2)
        batch, position = torch.where(mask_line)
        for i, p in zip(batch, position):
            h, w = p // L1, p % L1
            for j in range(scale ** 2):
                j1 = j // scale
                j2 = j % scale
                H[i, int((h * scale + j1) * (L1 * scale) + (w * scale + j2))] = 1
        return H