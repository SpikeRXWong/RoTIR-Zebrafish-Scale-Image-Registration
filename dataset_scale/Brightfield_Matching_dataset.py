# -*- coding: utf-8 -*-
"""
Created on Thu Sep 14 11:48:19 2023

@author: rw17789
"""

import os
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

from einops.einops import reduce

from .utils import affine_matrix, matrix_transform_image2coordinate
import random
from math import pi

class BrightFieldMatchingImageDataset(Dataset):
    def __init__(self,
                 image_path,
                 image_size = 256,
                 image_type = "bf",
                 train = True,
                 grid_size = 16,
                 value_restriction = False
                 ):
        self.image_path = image_path
        assert image_size in [256, 128]
        self.image_size = image_size
        assert image_type in ["bf", "fb", "bb", "ff"]
        self.image_type = image_type
        self.train = train
        self.grid_size = grid_size
        self.value_restriction = value_restriction
    
    def _tv_affine_transform(self, img1, img2, img3, angle, translate, scale=1, shear = [0, 0]):
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0)
        if img3.dim() == 2:
            img3 = img3.unsqueeze(0)
        re_img1 = T.functional.affine(
            img1, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST,#BILINEAR, 
            fill = 0
        )
        re_img2 = T.functional.affine(
            img2, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        re_img3 = T.functional.affine(
            img3, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        return re_img1, re_img2, re_img3
    
    def __getitem__(self, index):
        folder = os.path.join(self.image_path, os.listdir(self.image_path)[index])
        for f in os.listdir(folder):
            if "brightfiled" in f:
                file = os.path.join(folder, f)
            if "fluorescent_1" in f:
                file_f = os.path.join(folder, f)
            if "mask" in f:
                file_m = os.path.join(folder, f)
                
        # bfimage = torch.from_numpy(np.load(file))
        # frimage = torch.from_numpy(np.load(file_f)).float().div(255)
        # frmask = torch.from_numpy(np.load(file_m)).float().sum(dim=0, keepdim=True)
        
        ind = 5#random.randint(0, bfimage.shape[0]-1)
        
        bfimage = torch.from_numpy(np.load(file)).float().div(255/2).sub(1)
        frimage = torch.from_numpy(np.load(file_f)).float().div(255/2).sub(1)
        frmask = torch.from_numpy(np.load(file_m)).float().sum(dim=0, keepdim=True)
        
        assert bfimage.shape[-1] == 512#self.image_size * 2, bfimage.shape[-1]
        assert frimage.shape[-1] == 512 #self.image_size * 2, frimage.shape[-1]
        assert frmask.shape[-1] == 512#self.image_size * 2, frmask.shape[-1]
        
        bfimage = bfimage[ind : ind + 1,...].unsqueeze(0)
        frimage = frimage.unsqueeze(0).unsqueeze(0)        
        frmask = frmask.unsqueeze(0)
        
        result = {}
        for item in ['template', 'target']:
            _angle = random.uniform(-1, 1) * (1 if not self.value_restriction else 0.25)
            _t_1 = random.uniform(-1, 1) / 2#(2 if not self.value_restriction else 8)
            _t_2 = random.uniform(-1, 1) / 2#(2 if not self.value_restriction else 8)
            
            image1, image2, image3 = self._tv_affine_transform(
                bfimage,
                frimage,
                frmask,
                angle=_angle * 180,
                translate = [_t_1 * (256 // 2), _t_2 * (256 // 2)], 
                scale = 1
            )
            
            image1 = T.functional.crop(
                image1, 
                256 // 2, 256 // 2, 256, 256
            )
            image2 = T.functional.crop(
                image2, 
                256 // 2, 256 // 2, 256, 256
            )
            image3 = T.functional.crop(
                image3, 
                256 // 2, 256 // 2, 256, 256
            )
            
            result[item] = {
                "Brightfiled": image1.squeeze(0),
                "Fluorescent": image2.squeeze(0),
                "Mask" : image3.squeeze(0),
                "theta": _angle,
                "dx": _t_1,
                "dy": _t_2,
                }
        M1 = affine_matrix(
            dxy = (-result['template']['dx'], -result['template']['dy']),
            theta = -result['template']['theta'],
            scale = 0,
            full_row = True,
            base = 1
            )
        M2 = affine_matrix(
            dxy = (result['target']['dx'], result['target']['dy']),
            theta = result['target']['theta'],
            scale = 0,
            rotation_first = True,
            full_row = True,
            base = 1
            )
        transform_matrix = torch.mm(M1, M2)
        
        # if self.image_type[0] == self.image_type[1]:
        #     data = {
        #         "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
        #         # "Template_mask": result['template']['Mask'],
        #         "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
        #         # "Target_mask": result['target']["Mask"],
        #         }
        # elif self.train:
        #     data = {
        #         "Template_image_bf": result['template']['Brightfiled'],
        #         "Template_image_fr": result['template']['Fluorescent'],
        #         "Template_mask": result['template']['Mask'],
        #         "Target_image_bf": result['target']['Brightfiled'],
        #         "Target_image_fr": result['target']['Fluorescent'],
        #         "Target_mask": result['target']["Mask"],
        #         }
        # else:
        #     data = {
        #         "Template_image_bf": result['template']['Brightfiled'],
        #         "Template_image_fr": result['template']['Fluorescent'],
        #         "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
        #         # "Template_mask": result['template']['Mask'],
        #         "Target_image_bf": result['target']['Brightfiled'],
        #         "Target_image_fr": result['target']['Fluorescent'],
        #         "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
        #         # "Target_mask": result['target']["Mask"],
        #         }
        # data.update({
        #     "Transformation_Matrix": transform_matrix,
        #     "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
        #     "Transformation_Scale": torch.tensor([1]),
        #     })
        
        data = {
            # "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
            "Template_image_bf": result['template']['Brightfiled'],
            "Template_image_fr": result['template']['Fluorescent'],
            "Template_mask": result['template']['Mask'],
            # "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
            "Target_image_bf": result['target']['Brightfiled'],
            "Target_image_fr": result['target']['Fluorescent'],
            "Target_mask": result['target']["Mask"],
            "Transformation_Matrix": transform_matrix,
            "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
            "Transformation_Scale": torch.tensor([1]),
            }
        
        L = self.grid_size#16
        
        s_mask = reduce(result['template']['Mask'], 'i (j c1) (k c2) -> j k','mean', c1=256 // L, c2=256 // L) > (0 if self.image_type == "bb" else 0.25)

        t_mask = reduce(result['target']["Mask"], 'i (j c1) (k c2) -> j k','mean', c1=256 // L, c2=256 // L) > (0 if self.image_type == "bb" else 0.25)
        
        transform_matrix_reverse = transform_matrix.inverse()
        coordinate_matrix = matrix_transform_image2coordinate(transform_matrix, L)
        coordinate_matrix_reverse = matrix_transform_image2coordinate(transform_matrix_reverse, L)
        
        s_coord = torch.where(s_mask)
        t_coord = torch.where(t_mask)
        
        s_coord = torch.stack(list(s_coord[-2:]) + [torch.ones_like(s_coord[-1])], dim = 0)
        t_coord = torch.stack(list(t_coord[-2:]) + [torch.ones_like(t_coord[-1])], dim = 0)
        
        trans_from_s = torch.mm(coordinate_matrix, s_coord.float())[:2]
        trans_from_t = torch.mm(coordinate_matrix_reverse, t_coord.float())[:2]
        
        corners_from_s = trans_from_s.unsqueeze(0).repeat(4,1,1).int() + torch.Tensor([0,0,0,1,1,0,1,1]).view(4,-1).unsqueeze(-1)
        from_s_diff = torch.pow(corners_from_s.float() - trans_from_s, 2).sum(dim=1, keepdim=True)
        corners_info_s = torch.cat([from_s_diff, corners_from_s], dim = 1)
        
        corners_from_t = trans_from_t.unsqueeze(0).repeat(4,1,1).int() + torch.Tensor([0,0,0,1,1,0,1,1]).view(4,-1).unsqueeze(-1)
        from_t_diff = torch.pow(corners_from_t.float() - trans_from_t, 2).sum(dim=1, keepdim=True)
        corners_info_t = torch.cat([from_t_diff, corners_from_t], dim = 1)
        
        s_coord_list = [(s_coord[0,l].item(), s_coord[1,l].item()) for l in range(s_coord.size(-1))]
        t_coord_list = [(t_coord[0,l].item(), t_coord[1,l].item()) for l in range(t_coord.size(-1))]
        
        matched_s_t = []
        for i in range(corners_info_s.size(-1)):
            s_points = s_coord_list[i]
            for p in corners_info_s[corners_info_s[:,0,i].argsort(), 1:, i]:
                points = (p[0].int().item(), p[1].int().item())
                if points in t_coord_list:
                    matched_s_t.append((s_points, points))
                    break
        matched_t_s = []
        for i in range(corners_info_t.size(-1)):
            t_points = t_coord_list[i]
            for p in corners_info_t[corners_info_t[:,0,i].argsort(), 1:, i]:
                points = (p[0].int().item(), p[1].int().item())
                if points in s_coord_list:
                    matched_t_s.append((points, t_points))
                    break
        all_matched = []
        for t in matched_s_t + matched_t_s:
            if t not in all_matched:
                all_matched.append(t)
                
        new_coord_s = torch.ones(3, len(all_matched))
        new_coord_t = torch.ones(3, len(all_matched))
        for i,t in enumerate(all_matched):
            new_coord_s[0,i] = t[0][0]
            new_coord_s[1,i] = t[0][1]
            new_coord_t[0,i] = t[1][0]
            new_coord_t[1,i] = t[1][1]
            
        coord_diff = (torch.mm(coordinate_matrix, new_coord_s) - new_coord_t)[:2]
        
        thr = 0.5
        
        pos_trans_s = torch.mm(coordinate_matrix, new_coord_s)[:2]
        pos_trans_s = torch.logical_and(
            torch.logical_and(
            pos_trans_s[0] > -thr, pos_trans_s[0] < L - 1 + thr 
            ),
            torch.logical_and(
            pos_trans_s[1] > -thr, pos_trans_s[1] < L - 1 + thr 
            )
        )
        
        pos_trans_t = torch.mm(coordinate_matrix_reverse, new_coord_t)[:2]
        pos_trans_t = torch.logical_and(
            torch.logical_and(
            pos_trans_t[0] > -thr, pos_trans_t[0] < L - 1 + thr 
            ),
            torch.logical_and(
            pos_trans_t[1] > -thr, pos_trans_t[1] < L - 1 + thr 
            )
        )
        
        pos_trans = torch.logical_and(pos_trans_s, pos_trans_t)
        
        matching_map = torch.cat([
                torch.zeros(1, L ** 2, L ** 2),
                torch.ones(2, L ** 2, L ** 2).mul(-1e9)
            ], 0)

        for ((x0, y0), (x1, y1)), d in zip(torch.Tensor(all_matched)[pos_trans], coord_diff.t()[pos_trans]):
            matching_map[  :, int(x0 * L + y0), int(x1 * L + y1)] = torch.cat([torch.tensor([1.0]), d], 0)       
        
        data.update({"Matching_map": matching_map})
        
        if self.image_size == 128:
            for k, v in data.items():
                if "Target_" in k or "Template_" in k:
                    data[k] = F.interpolate(v.unsqueeze(0), size = 128, mode = "nearest").squeeze(0)
        
        return data
    
    def __len__(self):
        return len(os.listdir(self.image_path))
        
    

class DualDomainMatchingImageDataset(Dataset):
    def __init__(self,
                 image_path,
                 image_size = 256,
                 image_type = "bf",
                 # train = True,
                 # grid_size = 16,
                 value_restriction = False,
                 eval_mode = True
                 ):
        self.image_path = image_path
        assert image_size in [256, 128]
        self.image_size = image_size
        assert image_type in ["bf", "fb", "bb", "ff"]
        self.image_type = image_type
        # self.train = train
        # self.grid_size = grid_size
        self.value_restriction = value_restriction
        self.eval_mode = eval_mode
    
    def _tv_affine_transform(self, img1, img2, img3, angle, translate, scale=1, shear = [0, 0]):
        if img1.dim() == 2:
            img1 = img1.unsqueeze(0)
        if img2.dim() == 2:
            img2 = img2.unsqueeze(0)
        if img3.dim() == 2:
            img3 = img3.unsqueeze(0)
        re_img1 = T.functional.affine(
            img1, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST,#BILINEAR, 
            fill = 0
        )
        re_img2 = T.functional.affine(
            img2, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        re_img3 = T.functional.affine(
            img3, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        return re_img1, re_img2, re_img3
    
    def __getitem__(self, index):
        folder = os.path.join(self.image_path, os.listdir(self.image_path)[index])
        for f in os.listdir(folder):
            if "brightfiled" in f:
                file = os.path.join(folder, f)
            if "fluorescent_1" in f:
                file_f = os.path.join(folder, f)
            if "mask" in f:
                file_m = os.path.join(folder, f)
                
        # bfimage = torch.from_numpy(np.load(file))
        # frimage = torch.from_numpy(np.load(file_f)).float().div(255)
        # frmask = torch.from_numpy(np.load(file_m)).float().sum(dim=0, keepdim=True)
        
        ind = 5#random.randint(0, bfimage.shape[0]-1)
        
        bfimage = torch.from_numpy(np.load(file)).float().div(255/2).sub(1)
        frimage = torch.from_numpy(np.load(file_f)).float().div(255/2).sub(1)
        frmask = torch.from_numpy(np.load(file_m)).float().sum(dim=0, keepdim=True)
        
        assert bfimage.shape[-1] == 512#self.image_size * 2, bfimage.shape[-1]
        assert frimage.shape[-1] == 512 #self.image_size * 2, frimage.shape[-1]
        assert frmask.shape[-1] == 512#self.image_size * 2, frmask.shape[-1]
        
        bfimage = bfimage[ind : ind + 1,...].unsqueeze(0)
        frimage = frimage.unsqueeze(0).unsqueeze(0)        
        frmask = frmask.unsqueeze(0)
        
        result = {}
        
        if self.eval_mode:
            if self.value_restriction:
                _angle_add = random.uniform(-1, 1)
                _angle_base = random.uniform(-0.125, 0.125) * (1 if self.value_restriction < 3 else 2)
                if self.value_restriction == 2:
                    _angle_base += 0.125 * (1 if _angle_base >= 0 else -1) 
        else:
            ### hpc
            if self.value_restriction:
                _angle_add = random.uniform(-1, 1)
                _large_angle = random.uniform(-1, 1)
                if abs(_large_angle) > 0.5: 
                    _angle_sign = 1 if _large_angle > 0 else -1
            
        for item in ['template', 'target']:
            
            if self.eval_mode:
                if self.value_restriction:
                    _angle = _angle_base * (1 if item == "template" else -1) + _angle_add
                else:
                    _angle = random.uniform(-1, 1)
                    
                _t_1 = random.uniform(-1, 1) / (2 if not self.value_restriction else 8) 
                _t_2 = random.uniform(-1, 1) / (2 if not self.value_restriction else 8) 

            else:
                ### hpc
                if self.value_restriction:
                    if abs(_large_angle) <= 0.5:
                        _angle = random.uniform(-1, 1) * 0.25 + _angle_add
                    else:
                        _angle = random.uniform(0.5, 1) * 0.25 * (_angle_sign if item == 'template' else - _angle_sign) + _angle_add
                    _t_1 = random.uniform(-1, 1) / 8
                    _t_2 = random.uniform(-1, 1) / 8
                else:
                    _angle = random.uniform(-1, 1)
                    _t_1 = random.uniform(-1, 1) / 2
                    _t_2 = random.uniform(-1, 1) / 2
            
            image1, image2, image3 = self._tv_affine_transform(
                bfimage,
                frimage,
                frmask,
                angle=_angle * 180,
                translate = [_t_1 * (256 // 2), _t_2 * (256 // 2)], 
                scale = 1
            )
            
            image1 = T.functional.crop(
                image1, 
                256 // 2, 256 // 2, 256, 256
            )
            image2 = T.functional.crop(
                image2, 
                256 // 2, 256 // 2, 256, 256
            )
            image3 = T.functional.crop(
                image3, 
                256 // 2, 256 // 2, 256, 256
            )
            
            result[item] = {
                "Brightfiled": image1.squeeze(0),
                "Fluorescent": image2.squeeze(0),
                "Mask" : image3.squeeze(0),
                "theta": _angle,
                "dx": _t_1,
                "dy": _t_2,
                }
        M1 = affine_matrix(
            dxy = (-result['template']['dx'], -result['template']['dy']),
            theta = -result['template']['theta'],
            scale = 0,
            full_row = True,
            base = 1
            )
        M2 = affine_matrix(
            dxy = (result['target']['dx'], result['target']['dy']),
            theta = result['target']['theta'],
            scale = 0,
            rotation_first = True,
            full_row = True,
            base = 1
            )
        transform_matrix = torch.mm(M1, M2)
        
        # if self.image_type[0] == self.image_type[1]:
        #     data = {
        #         "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
        #         # "Template_mask": result['template']['Mask'],
        #         "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
        #         # "Target_mask": result['target']["Mask"],
        #         }
        # elif self.train:
        #     data = {
        #         "Template_image_bf": result['template']['Brightfiled'],
        #         "Template_image_fr": result['template']['Fluorescent'],
        #         "Template_mask": result['template']['Mask'],
        #         "Target_image_bf": result['target']['Brightfiled'],
        #         "Target_image_fr": result['target']['Fluorescent'],
        #         "Target_mask": result['target']["Mask"],
        #         }
        # else:
        #     data = {
        #         "Template_image_bf": result['template']['Brightfiled'],
        #         "Template_image_fr": result['template']['Fluorescent'],
        #         "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
        #         # "Template_mask": result['template']['Mask'],
        #         "Target_image_bf": result['target']['Brightfiled'],
        #         "Target_image_fr": result['target']['Fluorescent'],
        #         "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
        #         # "Target_mask": result['target']["Mask"],
        #         }
        # data.update({
        #     "Transformation_Matrix": transform_matrix,
        #     "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
        #     "Transformation_Scale": torch.tensor([1]),
        #     })
        
        data = {
            # "Template_image": result['template']['Brightfiled' if self.image_type[0] == 'b' else 'Fluorescent'],
            "Template_image_bf": result['template']['Brightfiled'],
            "Template_image_fr": result['template']['Fluorescent'],
            "Template_mask": result['template']['Mask'],
            # "Target_image": result['target']['Brightfiled' if self.image_type[1] == 'b' else 'Fluorescent'],
            "Target_image_bf": result['target']['Brightfiled'],
            "Target_image_fr": result['target']['Fluorescent'],
            "Target_mask": result['target']["Mask"],
            "Transformation_Matrix": transform_matrix,
            "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
            "Transformation_Scale": torch.tensor([1]),
            }
        
        # L = self.grid_size#16
        
        for L in [8, 16]:
        
            s_mask = reduce(result['template']['Mask'], 'i (j c1) (k c2) -> j k','mean', c1=256 // L, c2=256 // L) > (0 if self.image_type == "bb" or L == 8 else 0.25)
    
            t_mask = reduce(result['target']["Mask"], 'i (j c1) (k c2) -> j k','mean', c1=256 // L, c2=256 // L) > (0 if self.image_type == "bb" or L == 8 else 0.25)
            
            transform_matrix_reverse = transform_matrix.inverse()
            coordinate_matrix = matrix_transform_image2coordinate(transform_matrix, L)
            coordinate_matrix_reverse = matrix_transform_image2coordinate(transform_matrix_reverse, L)
            
            s_coord = torch.where(s_mask)
            t_coord = torch.where(t_mask)
            
            s_coord = torch.stack(list(s_coord[-2:]) + [torch.ones_like(s_coord[-1])], dim = 0)
            t_coord = torch.stack(list(t_coord[-2:]) + [torch.ones_like(t_coord[-1])], dim = 0)
            
            trans_from_s = torch.mm(coordinate_matrix, s_coord.float())[:2]
            trans_from_t = torch.mm(coordinate_matrix_reverse, t_coord.float())[:2]
            
            s_coord_list = s_coord[:2].int()
            t_coord_list = t_coord[:2].int()
            
            corners_from_s = trans_from_s.unsqueeze(0).repeat(4,1,1).int() + torch.Tensor([0,0,0,1,1,0,1,1]).view(4,-1).unsqueeze(-1)
            from_s_diff = torch.pow(corners_from_s.float() - trans_from_s, 2).sum(dim=1, keepdim=True)
            corners_info_s = torch.cat([from_s_diff, corners_from_s], dim = 1)
            
            corners_from_t = trans_from_t.unsqueeze(0).repeat(4,1,1).int() + torch.Tensor([0,0,0,1,1,0,1,1]).view(4,-1).unsqueeze(-1)
            from_t_diff = torch.pow(corners_from_t.float() - trans_from_t, 2).sum(dim=1, keepdim=True)
            corners_info_t = torch.cat([from_t_diff, corners_from_t], dim = 1)
                
            matched_s_t = []
            for i in range(corners_info_s.size(-1)):
                s_points = s_coord_list[:,i]
            
                for p in corners_info_s[corners_info_s[:,0,i].argsort(), :, i]:
                    points = p[1:]
                    if (t_coord_list.t() == points.int()).prod(-1).sum():
                        matched_s_t.append(torch.cat([p[:1], s_points, points], dim = 0))
                        break
                
            matched_t_s = []    
            for i in range(corners_info_t.size(-1)):
                t_points = t_coord_list[:,i]
                
                for p in corners_info_t[corners_info_t[:,0,i].argsort(), :, i]:
                    points = p[1:]
                    if (s_coord_list.t() == points.int()).prod(-1).sum():
                        matched_t_s.append(torch.cat([p[:1], points, t_points], dim = 0))
                        break
                    
            if (len(matched_s_t) + len(matched_t_s)) == 0:
                all_matched = torch.tensor([]).view(-1, 5)
                import warnings
                warnings.warn("Empty matching!")
                print("No matching found for {} matching ({}, {}) at {} !".format("coarse" if L == 8 else "fine", len(matched_s_t), len(matched_t_s), folder))
            else:
                all_matched = torch.stack(matched_s_t + matched_t_s, dim = 0)
        
            for i in range(all_matched.size(0) - 1):

                for j in range(i + 1, all_matched.size(0)):
                    if L == 16:
                    # if True:
                        if torch.equal(all_matched[i,1:3], all_matched[j,1:3]) or torch.equal(all_matched[i, 3:5], all_matched[j,3:5]):
                            if all_matched[i,0] > 2:
                                break
                            elif all_matched[i,0] >= all_matched[j,0]:
                                all_matched[i,0] = 3
                                break
                            else:
                                all_matched[j,0] = 3
                    else:
                        if torch.equal(all_matched[i, 1:], all_matched[j, 1:]):
                            all_matched[i,0] = 3
                            break
                            
            all_matched = all_matched[all_matched[:,0] < 1, 1:]   
         
            new_coord_s = torch.ones(3, len(all_matched))
            new_coord_t = torch.ones(3, len(all_matched))
            for i,t in enumerate(all_matched):
                new_coord_s[0,i] = t[0]
                new_coord_s[1,i] = t[1]
                new_coord_t[0,i] = t[2]
                new_coord_t[1,i] = t[3]
                
            coord_diff = (torch.mm(coordinate_matrix, new_coord_s) - new_coord_t)[:2]
            
            # thr = 0#0.5
            
            # pos_trans_s = torch.mm(coordinate_matrix, new_coord_s)[:2]
            # pos_trans_s = torch.logical_and(
            #     torch.logical_and(
            #     pos_trans_s[0] > -thr, pos_trans_s[0] < L - 1 + thr 
            #     ),
            #     torch.logical_and(
            #     pos_trans_s[1] > -thr, pos_trans_s[1] < L - 1 + thr 
            #     )
            # )
            
            # pos_trans_t = torch.mm(coordinate_matrix_reverse, new_coord_t)[:2]
            # pos_trans_t = torch.logical_and(
            #     torch.logical_and(
            #     pos_trans_t[0] > -thr, pos_trans_t[0] < L - 1 + thr 
            #     ),
            #     torch.logical_and(
            #     pos_trans_t[1] > -thr, pos_trans_t[1] < L - 1 + thr 
            #     )
            # )
            
            # pos_trans = torch.logical_and(pos_trans_s, pos_trans_t)
            
            matching_map = torch.cat([
                    torch.zeros(1, L ** 2, L ** 2),
                    torch.ones(2, L ** 2, L ** 2).mul(-1e9)
                ], 0)
    
            # for ((x0, y0), (x1, y1)), d in zip(torch.Tensor(all_matched)[pos_trans], coord_diff.t()[pos_trans]):
            # for (x0, y0, x1, y1), d in zip(all_matched[pos_trans], coord_diff.t()[pos_trans]):
            #     matching_map[  :, int(x0 * L + y0), int(x1 * L + y1)] = torch.cat([torch.tensor([1.0]), d], 0)       
            
            # data.update({"Matching_map_{}".format(L): matching_map})
            
            for (x0, y0, x1, y1), d in zip(all_matched, coord_diff.t()):
                matching_map[  :, int(x0 * L + y0), int(x1 * L + y1)] = torch.cat([torch.tensor([1.0]), d], 0)
            data.update({"Matching_map_{}".format(L): matching_map})    
        
        if self.image_size == 128:
            for k, v in data.items():
                if "Target_" in k or "Template_" in k:
                    data[k] = F.interpolate(v.unsqueeze(0), size = 128, mode = "nearest").squeeze(0)
        
        return data
    
    def __len__(self):
        return len(os.listdir(self.image_path))
        