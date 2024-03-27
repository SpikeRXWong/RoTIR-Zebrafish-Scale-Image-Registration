# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:27:37 2023

@author: rw17789
"""

import os
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch.nn.functional as F

from .utils import affine_matrix, matrix_transform_image2coordinate
import random
from einops.einops import reduce#, rearrange, repeat
from math import pi


class ArtificialMatchingImageDataset(Dataset):
    def __init__(self, 
                 image_size, 
                 image_mask_path, 
                 background_path, 
                 kernal_size = 8, 
                 threshold = 0.25, 
                 base_scale = 1.5, 
                 add_noise = True, 
                 eval_data = False, 
                 same_scale = False
                 ):  
        self.background_path = background_path
        # self.background = torch.load(background_path, map_location="cpu")
        self.image_mask_path = image_mask_path
        
        # self.circle = self._make_mask(512, 215.0/512.0).unsqueeze(0)
        self.image_size = image_size
        
        assert base_scale >= 1
        self.base_scale = base_scale
        self.kernal_size = kernal_size
        self.threshold = threshold
        self.add_noise = add_noise
        self.eval = eval_data
        self.same_scale = same_scale
        
        # for seize the random seed
        self.epoch = None
        
    def _make_mask(self, l, r = 225.0 / 512.0):
        if not isinstance(l, tuple):
            l = (l, l)
        if r < 1:
            r = r * l[0]
        h, w = l
        mask = torch.zeros(l).bool()
        c_h = (h - 1)/2.0
        c_w = (w - 1)/2.0
        for i in range(h):
            for j in range(w):
                dsqure = (c_h - i)**2 + (c_w - j)**2
                if dsqure < r**2:
                    mask[i,j] = True
        return mask

    def _tv_affine_transform(self, img, mask, angle, translate, scale, shear = [0, 0]):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        re_img = T.functional.affine(
            img, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST,#BILINEAR, 
            fill = 0
        )
        re_mask = T.functional.affine(
            mask, 
            angle = angle, 
            translate=translate, 
            scale=scale, 
            shear = shear, 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        return re_img, (re_mask > 0).float()
    
    def __call__(self, batch_size, epoch, device = "cpu"):
        self.epoch = epoch
        data_list = []
        
        torch.random.manual_seed(self.epoch)
        for index in torch.randperm(len(os.listdir(self.image_mask_path))).tolist()[: batch_size]:
            data_list.append(self.__getitem__(index))
        data_dict = {}
        for k in data_list[0].keys():
            data_dict[k] = torch.stack([d[k] for d in data_list], dim = 0).to(device)
        self.epoch = None
        return data_dict
        
    def __getitem__(self, index):
        if self.epoch is not None:
            random.seed(self.epoch + 1e5)
            
        circle = self._make_mask(512, 215.0/512.0).unsqueeze(0)
        
        image_mask_dict = torch.load(os.path.join(self.image_mask_path, os.listdir(self.image_mask_path)[index]), map_location="cpu")

        h, w = image_mask_dict['image'].size()[-2:]

        if random.random() < 0.5 and not self.eval:
            for k, v in image_mask_dict.items():
                image_mask_dict[k] = torch.flip(v, dims=[-1])
                
        _shear = [0, 32 * random.uniform(-1, 1) if random.random() < 0.5 and not self.eval else 0]

        result = {}
        for item in ['template', 'target']:
            
            _angle = 0 if item == 'target' and self.eval else random.uniform(-1, 1)
            _t_1 = 0 if item == 'target' and self.eval else random.uniform(-1, 1)
            _t_2 = 0 if item == 'target' and self.eval else random.uniform(-1, 1)
            _scale = 0 if item == 'target' and self.eval else random.uniform(-1, 1)
            if self.same_scale and item == 'target':
                _scale = result['template']['scale']
                
            while(True):
                image, mask = self._tv_affine_transform(
                    image_mask_dict['image'], 
                    image_mask_dict['mask'],
                    angle=_angle * 180, 
                    translate = [_t_1 * (h // 2), _t_2 * (w // 2)], 
                    scale =  self.base_scale ** _scale,
                    shear = _shear
                )
                if torch.equal(circle * mask, mask):
                    break
                else:
                    _t_1 = _t_1 * 0.85
                    _t_2 = _t_2 * 0.85

            background = torch.load(self.background_path, map_location="cpu")
            final_image = background * (1 - mask) + image * mask.mul(1 if item == 'target' and self.eval else random.uniform(0.5, 2.0)) # here add random random.uniform(1.0, 3.0)

            if self.image_size != 512:
                final_image = F.interpolate(final_image.unsqueeze(0), size = self.image_size, mode='bilinear').squeeze(0)
                mask = F.interpolate(mask.unsqueeze(0), size = self.image_size, mode='nearest').squeeze(0)
                
            if self.add_noise:
                if self.epoch is not None:
                    torch.random.manual_seed(self.epoch + 2e5)
                gaussin_map = torch.normal(mean = torch.zeros(1,1,20,20).add(0.05), std = torch.ones(1,1,20,20).mul(0.02 * 1))#max(1, self.image_size/128)))
                gaussin_map = F.interpolate(gaussin_map, size = self.image_size, mode = "nearest").squeeze(0) * \
                    F.interpolate(circle.unsqueeze(0).float(), size = self.image_size, mode = "nearest").squeeze(0)
                final_image = (final_image + gaussin_map).clamp(-1,1)

            result[item] = {
                "Image": final_image,
                "Mask": mask,
                "theta": _angle,
                "dx": _t_1,
                "dy": _t_2,
                "scale": _scale,
                }
        M1 = affine_matrix(
            dxy = (-result['template']['dx'], -result['template']['dy']),
            theta = -result['template']['theta'],
            scale = -result['template']['scale'],
            full_row = True,
            base = self.base_scale
            )
        M2 = affine_matrix(
            dxy = (result['target']['dx'], result['target']['dy']),
            theta = result['target']['theta'],
            scale = result['target']['scale'],
            rotation_first = True,
            full_row = True,
            base = self.base_scale
            )
        transform_matrix = torch.mm(M1, M2)
        data = {
            "Template_image": result['template']['Image'],
            # "Template_mask": result['template']['Mask'],
            "Target_image": result['target']['Image'],
            # "Target_mask": result['target']["Mask"],
            "Transformation_Matrix": transform_matrix,
            "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
            "Transformation_Scale": torch.Tensor([self.base_scale ** (result['target']['scale'] - result['template']['scale'])]),
            }

        L = self.image_size // self.kernal_size

        # coordinate_matrix = matrix_transform_image2coordinate(transform_matrix, L)

        s_mask = reduce(result['template']['Mask'], 'i (j c1) (k c2) -> j k','mean', c1=self.kernal_size, c2=self.kernal_size)
        mask0 = (s_mask > 0)
        s_mask = (s_mask > (0.25))

        t_mask = reduce(result['target']["Mask"], 'i (j c1) (k c2) -> j k','mean', c1=self.kernal_size, c2=self.kernal_size)
        mask1 = (t_mask > 0)
        t_mask = (t_mask > (0.25))

        m0h_min, m0h_max = max(torch.where(mask0)[0].min() - self._randedge(), 0), min(torch.where(mask0)[0].max() + 1 + self._randedge(), L)
        m0w_min, m0w_max = max(torch.where(mask0)[1].min() - self._randedge(), 0), min(torch.where(mask0)[1].max() + 1 + self._randedge(), L)
        m1h_min, m1h_max = max(torch.where(mask1)[0].min() - self._randedge(), 0), min(torch.where(mask1)[0].max() + 1 + self._randedge(), L)
        m1w_min, m1w_max = max(torch.where(mask1)[1].min() - self._randedge(), 0), min(torch.where(mask1)[1].max() + 1 + self._randedge(), L)
        
        mask0, mask1 = torch.zeros_like(mask0), torch.zeros_like(mask1)

        mask0[m0h_min: m0h_max, m0w_min: m0w_max] = True
        mask1[m1h_min: m1h_max, m1w_min: m1w_max] = True
        
        s_mask = torch.logical_and(s_mask, mask0)
        t_mask = torch.logical_and(t_mask, mask1)
        
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
            
        matching_map = torch.cat([
                torch.zeros(1, L ** 2, L ** 2),
                torch.ones(2, L ** 2, L ** 2).mul(-1e9)
            ], 0)
       
        for ((x0, y0), (x1, y1)), d in zip(all_matched, coord_diff.t()):
            matching_map[  0, int(x0 * L + y0), int(x1 * L + y1)] = 1.0
            matching_map[-2:, int(x0 * L + y0), int(x1 * L + y1)] = d
            
        data.update(
            {
                "Matching_map": matching_map,
                "Template_square_mask": mask0,
                "Target_square_mask": mask1,
                }
            )
        if self.epoch is not None:
            self.epoch += int(1e6)
        return data
    
    def _randedge(self):
        if self.epoch is not None:
            random.seed(self.epoch + 3e5)
        return random.choices([-1,0,1,2], weights=(0.1, 0.6, 0.2, 0.1), k=1)[0]
                
    def __len__(self):
        return len(os.listdir(self.image_mask_path))
    
class RealMatchingImageDataset(Dataset):
    def __init__(self, 
                 image_size, 
                 image_mask_path, 
                 kernal_size = 8, 
                 threshold = 0.25, 
                 eval_data = False
                 ):        
        self.image_mask_path = image_mask_path
        
        self.image_size = image_size
        
        self.kernal_size = kernal_size
        
        self.threshold = threshold
        
        self.eval = eval_data
        
        self.epoch = None

    def _tv_affine_transform(self, img, mask, angle):
        if img.dim() == 2:
            img = img.unsqueeze(0)
        if mask.dim() == 2:
            mask = mask.unsqueeze(0)
        re_img = T.functional.affine(
            img, 
            angle = angle, 
            translate=[0,0], 
            scale=1, 
            shear = [0,0], 
            interpolation = T.InterpolationMode.NEAREST,#BILINEAR, 
            fill = 0
        )
        re_mask = T.functional.affine(
            mask, 
            angle = angle, 
            translate=[0,0], 
            scale=1, 
            shear = [0,0], 
            interpolation = T.InterpolationMode.NEAREST, 
            fill = 0
        )
        return re_img, (re_mask > 0).float()
    
    def __call__(self, batch_size, epoch, device = "cpu"):
        self.epoch = epoch
        data_list = []
        
        torch.random.manual_seed(self.epoch)
        for index in torch.randperm(len(os.listdir(self.image_mask_path))).tolist()[: batch_size]:
            data_list.append(self.__getitem__(index))
        data_dict = {}
        for k in data_list[0].keys():
            data_dict[k] = torch.stack([d[k] for d in data_list], dim = 0).to(device)
        self.epoch = None
        return data_dict
        
    def __getitem__(self, index):
        if self.epoch is not None:
            random.seed(self.epoch + 1e5)
        
        image_mask_dict = torch.load(os.path.join(self.image_mask_path, os.listdir(self.image_mask_path)[index]), map_location="cpu")

        h, w = image_mask_dict['image'].size()[-2:]

        result = {}
        for item in ['template', 'target']:
            
            _angle = 0 if item == 'target' and self.eval else random.uniform(-1, 1)
            
            image, mask = self._tv_affine_transform(
                image_mask_dict['image'], 
                image_mask_dict['mask'],
                angle=_angle * 180, 
            )

            if self.image_size != 512:
                image = F.interpolate(image.unsqueeze(0), size = self.image_size, mode='bilinear').squeeze(0)
                mask = F.interpolate(mask.unsqueeze(0), size = self.image_size, mode='nearest').squeeze(0)

            result[item] = {
                "Image": image,
                "Mask": mask,
                "theta": _angle,
                }
        M1 = affine_matrix(
            theta = -result['template']['theta'],
            )
        M2 = affine_matrix(
            theta = result['target']['theta'],
            )
        transform_matrix = torch.mm(M1, M2)
        data = {
            "Template_image": result['template']['Image'],
            # "Template_mask": result['template']['Mask'],
            "Target_image": result['target']['Image'],
            # "Target_mask": result['target']["Mask"],
            "Transformation_Matrix": transform_matrix,
            "Transformation_Rotation_Angle": torch.Tensor([(result['target']['theta'] - result['template']['theta']) * pi]),
            "Transformation_Scale": torch.Tensor([1])
            }

        L = self.image_size // self.kernal_size

        coordinate_matrix = matrix_transform_image2coordinate(transform_matrix, L)

        s_mask = reduce(result['template']['Mask'], 'i (j c1) (k c2) -> j k','mean', c1=self.kernal_size, c2=self.kernal_size)
        mask0 = (s_mask > 0)
        s_mask = (s_mask > (0.25))

        t_mask = reduce(result['target']["Mask"], 'i (j c1) (k c2) -> j k','mean', c1=self.kernal_size, c2=self.kernal_size)
        mask1 = (t_mask > 0)
        t_mask = (t_mask > (0.25))

        m0h_min, m0h_max = max(torch.where(mask0)[0].min() - self._randedge(), 0), min(torch.where(mask0)[0].max() + 1 + self._randedge(), L)
        m0w_min, m0w_max = max(torch.where(mask0)[1].min() - self._randedge(), 0), min(torch.where(mask0)[1].max() + 1 + self._randedge(), L)
        m1h_min, m1h_max = max(torch.where(mask1)[0].min() - self._randedge(), 0), min(torch.where(mask1)[0].max() + 1 + self._randedge(), L)
        m1w_min, m1w_max = max(torch.where(mask1)[1].min() - self._randedge(), 0), min(torch.where(mask1)[1].max() + 1 + self._randedge(), L)
        
        mask0, mask1 = torch.zeros_like(mask0), torch.zeros_like(mask1)

        mask0[m0h_min: m0h_max, m0w_min: m0w_max] = True
        mask1[m1h_min: m1h_max, m1w_min: m1w_max] = True
        
        s_mask = torch.logical_and(s_mask, mask0)
        t_mask = torch.logical_and(t_mask, mask1)
        
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
            
        matching_map = torch.cat([
                torch.zeros(1, L ** 2, L ** 2),
                torch.ones(2, L ** 2, L ** 2).mul(-1e9)
            ], 0)
       
        for ((x0, y0), (x1, y1)), d in zip(all_matched, coord_diff.t()):
            matching_map[  0, int(x0 * L + y0), int(x1 * L + y1)] = 1.0
            matching_map[-2:, int(x0 * L + y0), int(x1 * L + y1)] = d
            
        data.update(
            {
                "Matching_map": matching_map,
                "Template_square_mask": mask0,
                "Target_square_mask": mask1,
                }
            )
        if self.epoch is not None:
            self.epoch += int(1e6)
        return data
    
    def _randedge(self):
        if self.epoch is not None:
            random.seed(self.epoch + 2e5)
        return random.choices([-1,0,1,2], weights=(0.1, 0.6, 0.2, 0.1), k=1)[0]
                
    def __len__(self):
        return len(os.listdir(self.image_mask_path))