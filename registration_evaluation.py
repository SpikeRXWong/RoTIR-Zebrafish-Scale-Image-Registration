# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 21:50:57 2023

@author: rw17789
"""

import argparse
import torch

from model import ImageRegistration
import os

from torchvision.transforms.functional import crop, rotate

from dataset import matrix_calculation_function, affine_transform

from ssim import SSIM
import torchvision.transforms as T

def parse_args():
    
    parser = argparse.ArgumentParser(description="Fish Scale Image Matching Evaluation")
    
    parser.add_argument("-f", "--folder", type = str, default = "/mnt/storage/scratch/rw17789/Registration_Result")
    
    parser.add_argument("-l", "--list", type = str, nargs='+')
    
    parser.add_argument("-i", "--index", type = int, nargs='+')

    parser.add_argument("-d", "--device", type = str, default = "cpu", choices = ["cpu", "cuda"])
    
    parser.add_argument("-n", "--name", type = str, default = "N", choices = ["N", "M"])

    parser.add_argument("-m", "--obligatory_mask", type = str, default = "none", choices = ["none", "true", "false"])
    
    parser.add_argument("-r", "--rotation", default = False, action = 'store_true')

    parser.add_argument("-c", "--compare_refine", default = False, action = 'store_true')
    
    return parser.parse_args()

def main(args):

    result_dict = {}
    
    assert args.list is not None
    if args.index is None:
        args.index = [None] * len(args.list)
    elif len(args.index) < len(args.list):
        args.index = args.index + args.index[-1:] * (len(args.list) - len(args.index))
    file_path_list = []
    for f, i in zip(args.list, args.index):
        tar = "Fishscale_registration_{}_{}".format(f, args.name)
        
        if "Fishscale_registration_{}_{}.pth".format(f, i) in os.listdir(os.path.join(args.folder, tar)):
            ind = i                                                                                 
        else:
            tar_file_num = [int(tt[-9:-4]) for tt in os.listdir(os.path.join(args.folder, tar))]
            ind = max(tar_file_num)
        file_path_list.append(os.path.join(args.folder, tar, "Fishscale_registration_{}_{:0>5d}.pth".format(f, ind)))    
    
    for file_name, file_path in zip(args.list, file_path_list):

        print(file_name, ":", file_path)
    
        model_file = torch.load(file_path, map_location="cpu")
        
        test_dict_whole = torch.load(os.path.join(args.folder,"test_datadict_for_compare.pth"), map_location="cpu")
        
        if (args.obligatory_mask == "none" and model_file['Apply_mask'] == 'none') or args.obligatory_mask == "false":
            print("Registraion without guidance mask!")

            test_dict_whole.pop('Template_square_mask')
            test_dict_whole.pop('Target_square_mask')                
        
        use_scale = model_file['Parameter']['loss_weight']['Scale'] != 0
        use_trans = model_file['Parameter']['loss_weight']['Translation'] != 0
    
        model = ImageRegistration(model_file['Parameter']['model'])
    
        model.load_state_dict(model_file["Model_state"])
        
        model = model.to(args.device)
        
        ##
        for rotation_index in range(4):
            if args.rotation:
                for dict_key, dict_v in test_dict_whole.items():
                    if 'Template' in dict_key:
                        test_dict_whole[dict_key] = rotate(dict_v, 90)
            elif rotation_index != 0:
                break
    
            model.eval()
            with torch.no_grad():
                output = model(test_dict_whole)#(test_dict_cuda if args.device == "cuda" else test_dict_whole)
        
            score_thr = []
            for op in output['score_map'][:,:-1,:-1]:
                t = torch.minimum(op.flatten().sort()[0][-3], torch.tensor(0.4))
                score_thr.append(t)
            score_thr = torch.Tensor(score_thr).view(-1,1,1)
            
            _use_trans_list = [use_trans]
            
            if use_trans and args.compare_refine:
                _use_trans_list.append(False)
                
            for trans_index, _use_trans in enumerate(_use_trans_list):
                key = file_name 
                if args.rotation:
                    key = key + "_{:0>3d}".format((rotation_index + 1) * 90) 
                if trans_index != 0:
                    key = key + "c"
                print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>Result of {}".format(key))
                result_dict[key] = {
                    "value" : [],
                    "image" : [],
                    "config" : "{}{}{}".format(
                        str(model_file['Parameter']['model']['Backbone']['up_progress'])[0], 
                        str(model_file['Parameter']['model']['Apply_scale'])[0], 
                        "N" if (args.obligatory_mask == "none" and model_file['Apply_mask'] == 'none') or args.obligatory_mask == "false" else "A"# model_file['Apply_mask'][0].capitalize() 
                    )
                }
        
                affine_matrix = matrix_calculation_function(output, score_thr, not use_scale, _use_trans).to("cpu")
                result_dict[key]['matrix'] = affine_matrix
        
                transformed_mask = affine_transform(test_dict_whole['Template_mask'], affine_matrix)
                transformed_image = affine_transform(test_dict_whole['Template_image'], affine_matrix) * transformed_mask
        
                target_mask = test_dict_whole['Target_mask']
                target_image = test_dict_whole['Target_image'] * test_dict_whole['Target_mask']
        
                for i in range(test_dict_whole['Target_image'].shape[0]):
                    whole_mask = torch.logical_or(transformed_mask[i:i+1].bool(), target_mask[i:i+1].bool())
                    h, w = torch.where(whole_mask)[-2:]
                    height = (h.max() - h.min() + 1)
                    width = (w.max() - w.min() + 1)
                    top = h.min()
                    left = w.min()
        
                    transformed_crop_image = crop(
                        transformed_image[i:i+1], top, left, height, width
                    )
                    target_crop_image = crop(
                        target_image[i:i+1], top, left, height, width
                    )
        
                    dice = torch.logical_and(transformed_mask[i:i+1].bool(),target_mask[i:i+1].bool()).sum() * 2 / \
                        (transformed_mask[i:i+1].bool().sum() + target_mask[i:i+1].bool().sum())
            
                    cw_ssim_score = pytorch_cw_ssim(
                        transformed_crop_image,
                        target_crop_image
                    )
                    result_dict[key]["value"].append(torch.tensor([dice, cw_ssim_score]))
                    if trans_index == 0:
                        result_dict[key]["image"].append(torch.cat([transformed_crop_image, target_crop_image], dim = 1))     
                    
                result_dict[key]["value"] = torch.stack(result_dict[key]["value"],dim=0)
                print("{} is saved, shape of record is {}".format(key, result_dict[key]["value"].shape))
                print("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")

    torch.save(result_dict, os.path.join(args.folder, "45_pairs_reverse_rotated_image_registraion_evaluation_{}.pth".format(os.environ['SLURM_JOBID'])))

def pytorch_cw_ssim(im1, im2, int_range = (-1,1)):
    assert isinstance(im1, torch.Tensor) and isinstance(im2, torch.Tensor)
    assert im1.shape == im2.shape
    if im1.dim() > 2:
        for s in im1.shape[:-2]:
            assert s == 1
        d = im1.dim() - 3
        for i in range(d):
            im1 = im1.squeeze(0)
            im2 = im2.squeeze(0)
    else:
        assert im1.dim() == 2
        im1 = im1.unsqueeze(0)
        im2 = im2.unsqueeze(0)
    if int_range:
        im1 = im1.sub(int_range[0]).div(int_range[1]-int_range[0])
        im2 = im2.sub(int_range[0]).div(int_range[1]-int_range[0])
    im1_pil = T.ToPILImage()(im1)
    im2_pil = T.ToPILImage()(im2)
#     return im1_pil, im2_pil
    cw_ssim_score = SSIM(im1_pil).cw_ssim_value(im2_pil)
    return cw_ssim_score 

    
if __name__ == "__main__":
    args = parse_args()
    main(args)