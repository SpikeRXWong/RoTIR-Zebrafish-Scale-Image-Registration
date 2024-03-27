# -*- coding: utf-8 -*-
"""
Created on Thu Mar  2 15:41:34 2023

@author: rw17789
"""

import argparse
import os
import time

import torch

from model_scale import ImageRegistration, Matching_Loss
from dataset_scale import ArtificialMatchingImageDataset, RealMatchingImageDataset
# from math import log

def parse_args():

    parser = argparse.ArgumentParser(description="Fish Scale Image Matching Trainer")
    
    parser.add_argument("--device", default = "cuda", type = str, choices = ["cpu", "cuda"])
    
    parser.add_argument("--in_channel", default = 1, type = int)
    
    parser.add_argument("--hidden_channel", default = 64, type = int)
    
    parser.add_argument("--num_rotation", default = 4, type = int)
    
    parser.add_argument("--layer_iteration", default = 4, type = int)
    
    parser.add_argument("--attention_type", default = 'linear', type = str, choices = ['linear', 'full'])
    
    parser.add_argument("--matching_algorithm", type = str, default = 'sinkhorn')
    
    parser.add_argument("--sinkhorn_alpha", default = 1, type = int)
    
    parser.add_argument("--sinkhorn_iters", default = 3, type = int)
    
    parser.add_argument("--apply_mask", default = 'all', choices = ['all', 'none', 'random'])       
    
    parser.add_argument("-s", "--image_size", default = 128, type = int, choices = [128, 256, 512])
    
    parser.add_argument("--image_link", type = str, default ='/mnt/storage/scratch/rw17789/real_target_data/image_mask_dict_5.pth')
    
    parser.add_argument("--real_image_link", type = str)
    
    parser.add_argument("--background_link", type = str, default = '/mnt/storage/scratch/rw17789/real_target_data/basic_blank_background.pth')
    
    parser.add_argument("-b", "--batch_size", default = 4, type = int)
    
    parser.add_argument("-l", "--learning_rate", type = float)
    
    parser.add_argument("--optim_type", type = str, default = 'adamw', choices = ['adamw', 'adam'])
    
    parser.add_argument("--scheduler_type", type = str, default = 'MultiStepLR', choices = ['MultiStepLR', 'CosineAnnealing', 'ExponentialLR'])
    
    parser.add_argument("--milestones", type = int, default = [5000, 10000, 15000, 20000], nargs='+')
    
    parser.add_argument("--save_destination", type = str)
    
    parser.add_argument("-n", "--num_epoches", type = int)
    
    parser.add_argument("--loss_weight", type = float, default = [ 1.0, 1.0, 1.0, 1.0], nargs='+')
    
    parser.add_argument("--score_weight", type = float, default = [ 20.0, 10.0 , 1.0], nargs='+')
    
    parser.add_argument("--ckpt_interval", type = int, default = 250)
    
    parser.add_argument("-c", "--checkpoint_link", type = str)
    
    parser.add_argument("--RESUME", default = False, action = 'store_true')
    
    parser.add_argument("--renew_weight", default = False, action = 'store_true')
    
    parser.add_argument("--eval_interval", default = 20, type = int)
    
    parser.add_argument("--base_scale", default = 1.5, type = float)
    
    parser.add_argument("--RESET_LR", default = False, action = 'store_true')
    
    parser.add_argument("--app_scale", default = False, action = 'store_true')
    
    parser.add_argument("--fake_real_ratio", default = 20, type = int)
    
    parser.add_argument("--noup_progress", default = False, action = 'store_true')
    
    parser.add_argument("--notrans_correction", default = False, action = 'store_true')
    
    parser.add_argument("--random_dataset", default = True, action = 'store_false')

    parser.add_argument("--force_no_mask", default = False, action = 'store_true')

    return parser.parse_args()
    
def main(args):
    
    start_time = time.time()
    start_time_point = time.localtime(start_time)
    
    print("Fish scale image registration")
    print("Job ID: {}".format(os.environ['SLURM_JOBID']))
    print("Start time: {}/{:0>2d}/{:0>2d}\n".format(
        start_time_point.tm_year, start_time_point.tm_mon, start_time_point.tm_mday
        ))
    
    if not torch.cuda.is_available():
        args.device = "cpu"
        
    if args.RESUME:
        assert args.checkpoint_link is not None
        checkpoint = torch.load(args.checkpoint_link, args.device)
        model_name = checkpoint["Model_name"]
        
        config = checkpoint["Model_parameter"]
        
        if 'up_progress' not in config['Backbone']:
            config['Backbone'].update({'up_progress': True})
        if 'Trans_correction' not in config:
            config.update({'Trans_correction':True})
        
        
        if config["Backbone"]["bone_kernel"][0]:
            args.image_size = 512
        elif config["Backbone"]["bone_kernel"][1]:
            args.image_size = 256
        else:
            args.image_size = 128
        
        if args.renew_weight:
            loss_weight = {
                "Score_map": args.loss_weight[0],
                "Angle": args.loss_weight[1],
                "Scale": args.loss_weight[2] if config['Apply_scale'] else 0,
                "Translation": args.loss_weight[3] if config['Trans_correction'] else 0,
                }
        
            score_weight = {
                "Positive_loss": args.score_weight[0],
                "Negative_mask_loss": args.score_weight[1],
                "Negative_back_loss": args.score_weight[2],
                }
        else:
            loss_weight = checkpoint["Loss_weight"] 
            
            score_weight = checkpoint["Score_map_weight"]
        
        args.optim_type = checkpoint["Optimizer_type"] if "Optimizer_type" in checkpoint else args.optim_type
        
        args.scheduler_type = checkpoint["Scheduler_type"] if "Scheduler_type" in checkpoint else args.scheduler_type
        
        args.apply_mask = checkpoint["Apply_mask"] if not args.force_no_mask else "none"
        
        start_iter = checkpoint["Epoch"]
        
        if args.num_epoches <= start_iter + 1:
            args.num_epoches += start_iter + 1
    else:
        model_name = "Fishscale_registration_{}".format(os.environ['SLURM_JOBID'])
        start_iter = -1 # start from -1 or last epoch
        
        if args.image_size == 128:
            bone_kernel = [False, False]
        elif args.image_size == 256:
            bone_kernel = [False, True]
        else:
            bone_kernel = [True, True]
        
        config = {
            "Backbone":{
                "in_channel": args.in_channel,
                "hidden_channel": args.hidden_channel,
                "n_rotation": args.num_rotation,
                "bone_kernel": bone_kernel,
                "up_progress": not args.noup_progress,
                },
            "Pos_encoding":{
                "max_shape": (args.image_size // 8 + 1,) * 2,
                },
            "Transformer":{
                "nhead": 24 if not args.noup_progress else 8,
                "layer_names": ['self', 'cross'] * args.layer_iteration,
                "attention_type": args.attention_type,
                },
            "Matching_algorithm": {
                "Type": args.matching_algorithm, 
                "alpha": args.sinkhorn_alpha, 
                "iters": args.sinkhorn_iters,
                },
            "Apply_scale": args.app_scale,
            "Trans_correction": not args.notrans_correction, 
            }
        
        loss_weight = {
            "Score_map": args.loss_weight[0],
            "Angle": args.loss_weight[1],
            "Scale": args.loss_weight[2] if config['Apply_scale'] else 0,
            "Translation": args.loss_weight[3] if config['Trans_correction'] else 0,
            }
        
        score_weight = {
            "Positive_loss": args.score_weight[0],
            "Negative_mask_loss": args.score_weight[1],
            "Negative_back_loss": args.score_weight[2],
            }
    
    model = ImageRegistration(config)
        
    loss_func = Matching_Loss(loss_weight, 
                              score_weight, 
                              config["Matching_algorithm"]["Type"], 
                              None
                              ).to(args.device)
        
    if args.optim_type == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate, weight_decay = 0)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr = args.learning_rate, weight_decay = 0.1)
        
    if args.scheduler_type == 'MultiStepLR':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones = args.milestones, gamma=0.5)
    elif args.scheduler_type == 'CosineAnnealing':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 30)
    else:
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.999992)
        
    if args.RESUME:       
        model.load_state_dict(checkpoint["Model_state"])
        if not args.RESET_LR:
            optimizer.load_state_dict(checkpoint["Optimizer"])
            scheduler.load_state_dict(checkpoint["Scheduler"])
        
        del checkpoint
        torch.cuda.empty_cache()
        
    model = model.to(args.device)
    
    if args.RESUME and not args.RESET_LR:
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor) and v.device != args.device:
                    state[k] = v.to(args.device)
    
    training_dataset = ArtificialMatchingImageDataset(
        image_size = args.image_size, 
        image_mask_path = args.image_link, 
        background_path = args.background_link,
        kernal_size = args.image_size // 16,
        base_scale = args.base_scale,
        add_noise = True, 
        same_scale = not config['Apply_scale'],
        )
    
    real_dataset = RealMatchingImageDataset(
        image_size = args.image_size, 
        image_mask_path = args.real_image_link, 
        kernal_size = args.image_size // 16,
        )
    
    if args.random_dataset:
        dataloader = torch.utils.data.DataLoader(
            training_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            drop_last=True
        )
        
        realloader = torch.utils.data.DataLoader(
            real_dataset,
            batch_size = args.batch_size,
            shuffle = True,
            drop_last=True
        )
        
    Loss_list = []
    Eval_list = []
    
    folder_loc = os.path.join(args.save_destination, model_name)
    if not os.path.exists(folder_loc):
        os.makedirs(folder_loc)
        
    print("************* Start Training *************")
    
    for epoch in range(start_iter + 1, args.num_epoches):
        t0 = time.time()

        torch.cuda.empty_cache()
        
        model.train()
        
        if args.random_dataset:
            data = next(iter(dataloader)) if epoch % args.fake_real_ratio != 0 else next(iter(realloader))
            for k, v in data.items():
                data[k] = v.to(args.device)
        else:
            data = training_dataset(args.batch_size, epoch, args.device) if epoch % args.fake_real_ratio != 0 else real_dataset(args.batch_size, epoch, args.device)
        
        if args.apply_mask == "none":
            data.pop("Template_square_mask")
            data.pop("Target_square_mask")
        elif args.apply_mask == "random":
            if not args.random_dataset:
                torch.random.manual_seed(epoch)
            if torch.rand(1) < 0.5:
                data.pop("Template_square_mask")
                data.pop("Target_square_mask")
        
        # for k, v in data.items():
        #     data[k] = v.to(args.device)
        
        output = model(data)
        
        losses, loss_dict = loss_func(**output, data_dict = data)
        
        loss_dict.update({"Total_loss": losses.item()})
        
        Loss_list.append(loss_dict)
        
        print("Train result at Epoch {:0>5d}:".format(epoch))
        for k,v in loss_dict.items():
            print("Loss of {} : {:.4f}".format(k, v))
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        if (epoch + 1) % args.eval_interval == 0 or epoch == args.num_epoches - 1:
            torch.cuda.empty_cache()
            model.eval()
            
            if args.random_dataset:
                eval_data = next(iter(dataloader))
                for k, v in eval_data.items():
                    eval_data[k] = v.to(args.device)
            else:
                eval_data = training_dataset(args.batch_size, epoch + args.num_epoches, args.device)
                
            if args.apply_mask == "none":
                eval_data.pop("Template_square_mask")
                eval_data.pop("Target_square_mask")
            elif args.apply_mask == "random":
                if not args.random_dataset:
                    torch.random.manual_seed(epoch)
                if torch.rand(1) < 0.5:
                    eval_data.pop("Template_square_mask")
                    eval_data.pop("Target_square_mask")            
            
            with torch.no_grad():
                eval_output = model(eval_data)
                eval_losses, eval_loss_dict = loss_func(**eval_output, data_dict = eval_data)
            eval_loss_dict.update({"Total_loss": eval_losses.item()})
            Eval_list.append(eval_loss_dict)
            
            print("------------------------------------")
            print("Evaluation result at Epoch {:0>5d}:".format(epoch))
            for k,v in eval_loss_dict.items():
                print("Eval_loss of {} : {:.4f}".format(k, v))
            print("------------------------------------")
        
        if (epoch + 1) % args.ckpt_interval == 0 or epoch == args.num_epoches - 1:
            
            checkpoint = {
                "Model_name": model_name,
                "Epoch": epoch,
                "Apply_mask": args.apply_mask, 
                
                "Model_state": model.state_dict(),
                "Optimizer": optimizer.state_dict(),
                "Scheduler": scheduler.state_dict(),
                
                "Model_parameter": config,
                "Loss_weight": loss_weight,
                "Score_map_weight": score_weight,
                
                "Optimizer_type": args.optim_type,
                "Scheduler_type": args.scheduler_type,
                }
            chkp_path = os.path.join(folder_loc, "Checkpoint_{}_{:0>5d}.pth".format(model_name, epoch))
            torch.save(checkpoint, chkp_path)
            del checkpoint, chkp_path
        
        scheduler.step()
                
        t1 = time.time()
        print("Time consumption for Epoch {:0>5d} is {}:{:0>2d}:{:0>2d}\n".format(epoch, int(t1-t0)//3600, (int(t1-t0)%3600)//60, int(t1-t0)%60))
        print("====================================")
        
                        
    # save_model
    saved_loss = {}
    if len(Loss_list) > 0:
        for key in Loss_list[0]:
            saved_loss[key] = [l[key] for l in Loss_list]
            
    saved_eval_loss = {}
    if len(Eval_list) > 0:
        for key in Eval_list[0]:
            saved_eval_loss[key] = [l[key] for l in Eval_list]
            
    model = model.to("cpu")
    model_save = {
        "Model_name": model_name,
        
        "Model_state": model.state_dict(),
        
        "Apply_mask": args.apply_mask, 
        
        "Loss": saved_loss,
        "Eval_Loss": saved_eval_loss,
        
        "Parameter": {
            "model": config,
            "loss_weight": loss_weight,
            "score_map_weight": score_weight,
            }
        }
    save_path = os.path.join(folder_loc, "{}_{:0>5d}.pth".format(model_name, args.num_epoches))
    torch.save(model_save, save_path)
    print("================ Model Saved ================")
    
    end_time = time.time()
    
    print("\nTime consumption from Epoch {} to Epoch {} is {}:{:0>2d}:{:0>2d}".format(start_iter+1, args.num_epoches - 1, int(end_time - start_time)//3600, (int(end_time - start_time)%3600)//60, int(end_time - start_time)%60))
        
    
if __name__ == "__main__":
    args = parse_args()
    main(args)
    