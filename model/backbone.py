# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 15:08:13 2023

@author: rw17789
"""

import torch
import torch.nn.functional as F

from e2cnn import nn
from e2cnn import gspaces

from einops.einops import rearrange

class Block(torch.nn.Module):
    def __init__(self, in_type, out_type, down = False, double = False):
        super().__init__()
        self.frame= nn.SequentialModule(
            nn.R2Conv(in_type, out_type, kernel_size= 4 if down else 3, stride = 2 if down else 1, padding = 1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace = True),
            nn.R2Conv(out_type, out_type, kernel_size= 4 if down and double else 3, stride = 2 if down and double else 1, padding = 1),
            nn.InnerBatchNorm(out_type),
            nn.ReLU(out_type, inplace = True)
            )
        if in_type.size != out_type.size:
            self.link = nn.SequentialModule(
                nn.R2Conv(in_type, out_type, kernel_size = 1),
                nn.InnerBatchNorm(out_type)
            )   
        else:
            self.link = None
        if down:
            self.down = nn.PointwiseAvgPool(out_type, 4 if double else 2, 4 if  double else 2)
        else:
            self.down = None
        
    def forward(self, x):
        x1 = self.frame(x)
        x0 = self.link(x) if self.link is not None else x
        if self.down is not None:
            x0 = self.down(x0)
        return x1 + x0
        
class Feature_Extraction(torch.nn.Module):
    
    def __init__(self, in_channel, hidden_channel=64, n_rotation = 4, bone_kernel = [False, False], num_layers = 3, up_progress = True):
#     def __init__(self, in_channel, out_channel = (8,1), hidden_channel=64, n_rotation = 4, num_layers = 3):
        super().__init__()
        
        self.r2_act = gspaces.Rot2dOnR2(N = n_rotation)        
        self.in_type = nn.FieldType(self.r2_act, in_channel * [self.r2_act.trivial_repr])
        hidden_channel = hidden_channel // n_rotation
        self.hd = hidden_channel
        out_type = nn.FieldType(self.r2_act, hidden_channel * [self.r2_act.regular_repr])
        
#         self.layer0 = nn.SequentialModule(
#             nn.R2Conv(self.in_type, out_type, kernel_size=4, stride=2, padding = 1),
#             nn.InnerBatchNorm(out_type),
#             nn.ReLU(out_type, inplace=True)
#         )
        # self.layer0 = Block(self.in_type, out_type,True)
        
        self.layer0 = Block(self.in_type, out_type,True,bone_kernel[0])
        
        self.down_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            in_type = out_type
            out_type = nn.FieldType(self.r2_act, hidden_channel * (2**i) * [self.r2_act.regular_repr])
            self.down_layers.append(Block(in_type, out_type, True if i != 0 else bone_kernel[1]))
        
        self.bottleneck = nn.R2Conv(out_type, out_type, 1)
        
        d3_vector_out_type = nn.FieldType(self.r2_act, hidden_channel * (2 ** (num_layers - 1)) * [self.r2_act.irrep(1)])
        self.layer3_out = nn.R2Conv(out_type, d3_vector_out_type, kernel_size=1)        
        
        self.up_progress = up_progress
        if self.up_progress:
            self.up_layers = torch.nn.ModuleList()
            for i in range(2): # num_layers - 1 or 2
                in_type = out_type
                out_type = nn.FieldType(self.r2_act, hidden_channel * (2 ** (num_layers - 2 - i)) * [self.r2_act.regular_repr])
                self.up_layers.append(
                    nn.SequentialModule(
                        nn.R2Upsampling(in_type, 2),
                        nn.R2Conv(in_type, out_type, 1)
                    )
                )            
                self.up_layers.append(Block(out_type, out_type))
            
            d1_regular_out_type = nn.FieldType(self.r2_act, hidden_channel * [self.r2_act.regular_repr])
            self.layer1_out = nn.R2Conv(out_type, d1_regular_out_type, 1)
            self.gpool = nn.GroupPooling(d1_regular_out_type) 
    
    def forward(self, input):
        x = nn.GeometricTensor(input, self.in_type)
        
        x = self.layer0(x)
        
        down_list = []
        for _layer in self.down_layers:
            x = _layer(x)
            down_list.append(x)
            # print(x.shape)
            
        x = self.bottleneck(x)
        
        d3_out = self.layer3_out(x)
        
        feature_out3 = rearrange(d3_out.tensor, 'b (c d) h w -> b c d h w', d = self.hd)
        
        if self.up_progress:
            for _up, _layer, _x in zip(self.up_layers[0::2], self.up_layers[1::2], down_list[-2::-1]):
                x = _up(x)
                x = x + _x
                x = _layer(x)
                # print(x.shape)
            
            d1_out = self.gpool(self.layer1_out(x))
        
            feature_out1 = rearrange(d1_out.tensor, 'b c (d1 h) (d2 w) -> b (d1 d2) c h w', d1 = 4, d2 = 4) # 2 * (nume_layers -1) or 4
        
        feature = torch.cat([feature_out3, feature_out1], dim = 1) if self.up_progress else feature_out3

        
        return feature