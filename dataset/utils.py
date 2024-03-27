# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 14:30:08 2023

@author: rw17789
"""

import torch
import torch.nn.functional as F

from math import pi, cos, sin

def affine_matrix(
    dxy = (0,0),
    theta = 0,
    scale = 0,
    rotation_first = False,
    full_row = True,
    size = 2,
    base = 1.5,
    drever = False
):
    dx = dxy[0] * size/2 if not drever else dxy[-1] * size/2
    dy = dxy[-1] * size/2 if not drever else dxy[0] * size/2
    
    angle = theta * pi
    
    scale = base ** (- scale)
    
    t_matrix = torch.Tensor([[1, 0, -dx],
                             [0, 1, -dy],
                             [0, 0,   1]])
    a_matrix = torch.Tensor([[  cos(angle) * scale, sin(angle) * scale, 0],
                             [ -sin(angle) * scale, cos(angle) * scale, 0],
                             [                   0,                  0, 1]])
    if rotation_first:
        matrix = torch.matmul(a_matrix, t_matrix)
    else:
        matrix = torch.matmul(t_matrix, a_matrix)
    
    if full_row:
        return matrix
    else:
        return matrix[:2,...]  
    
def affine_transform(img, matrix, mode = 'nearest', padding_mode="zeros"):
    assert isinstance(img, torch.Tensor)
    dim = img.dim()
    if dim ==2:
        img = img[None, None, ...]
    elif dim == 3:
        img = img[None,...]
    
    N = img.shape[0]
    if matrix.dim() == 2:
        matrix = matrix.unsqueeze(0).repeat(N,1,1,)
    if matrix.shape[-2] == 3:
        matrix = matrix[:,:2,:]
        
    grid = F.affine_grid(matrix, img.size(),align_corners = False)
    affine_image = F.grid_sample(img, grid,mode=mode, padding_mode=padding_mode, align_corners = False)
    if dim == 2:
        affine_image = affine_image.squeeze(0).squeeze(0)
    elif dim == 3:
        affine_image = affine_image.squeeze(0)
    return affine_image

def matrix_transform_image2coordinate(matrix, L):
    """
    Parameters
    ----------
    matrix : Tensor, size 3 X 3 or 2 X 3
        Image affine transformation matrix.
        matrix = [[ 1/scale * cos(theta), 1/scale * sin(theta), x],
                  [-1/scale * sin(theta), 1/scale * cos(theta), y],
                  [                    0,                    0, 1]]
    L : float
        The size of the image to be transformed.

    Returns
    -------
    coord_matrix : Tensor, size 3 X 3
        Coordinate affine transformation matrix.
        coord_matrix = [[ scale * cos(theta), scale * sin(theta), M],
                        [-scale * sin(theta), scale * cos(theta), N],
                        [                  0,                  0, 1]]

    """
    ps = 1 / (matrix[0,0] ** 2 + matrix[0,1] ** 2) ** 0.5
    cos, sin = matrix[0,0] * ps, matrix[0,1] * ps
    m, n  = matrix_parameter_forward(matrix[0, -1], matrix[1, -1], L, cos, sin)
    M, N  = ps * m + ((L-1)/2) * (1 - ps), ps * n + ((L-1)/2) * (1 - ps)
    coord_matrix = torch.Tensor([[ ps * cos, ps * sin, M],
                                 [-ps * sin, ps * cos, N],
                                 [        0,        0, 1]])
    return coord_matrix

def matrix_transform_coordinate2image(matrix, L):
    """
    Parameters
    ----------
    matrix : Tensor, size 3 X 3 or 2 X 3
        Coordinate affine transformation matirx.
        coord_matrix = [[ scale * cos(theta), scale * sin(theta), M],
                        [-scale * sin(theta), scale * cos(theta), N],
                        [                  0,                  0, 1]]
    L : float
        The size of the image to be transformed.

    Returns
    -------
    image_matrix : Tensor, size 3 X 3
        Image affine transformation matrix.
        matrix = [[ 1/scale * cos(theta), 1/scale * sin(theta), x],
                  [-1/scale * sin(theta), 1/scale * cos(theta), y],
                  [                    0,                    0, 1]]

    """
    ps = (matrix[0,0] ** 2 + matrix[0,1] ** 2) ** 0.5
    cos, sin = matrix[0,0] / ps, matrix[0,1] / ps
    m = (matrix[0,-1] - ((L-1)/2) * (1-ps)) / ps
    n = (matrix[1,-1] - ((L-1)/2) * (1-ps)) / ps
    x, y = matrix_parameter_backward(m, n, L, cos, sin)
    image_matrix = torch.Tensor([[ cos/ps, sin/ps, x],
                                 [-sin/ps, cos/ps, y],
                                 [      0,      0, 1]])
    return image_matrix

def matrix_parameter_forward(x, y, L, cos, sin):
    """
    Parameters
    ----------
    x : FLOAT
        image affine transformation matrix [0, -1].
    y : FLOAT
        image affine transformation matrix [1, -1].
    L : FLOAT
        feature map size.
    cos : FLOAT
        cos(theta).
    sin : FLOAT
        sin(theta).

    Returns
    -------
    M : FLOAT
        coordiante matirx [0, -1].
    N : FLOAT
        coordinate matirx [1, -1].

    """
    M = -(L-1)/2 * (cos + sin) - L * (y * cos + x * sin) / 2 + (L-1)/2
    N = -(L-1)/2 * (cos - sin) + L * (y * sin - x * cos) / 2 + (L-1)/2
    return M, N    

def matrix_parameter_backward(M, N, L, cos, sin):
    """
    Parameters
    ----------
    M : FLOAT
        coordiante matirx [0, -1].
    N : FLOAT
        coordinate matirx [1,-1].
    L : FLOAT
        feature map size.
    cos : FLOAT
        cos(theta).
    sin : FLOAT
        sin(theta).

    Returns
    -------
    x : FLOAT
        image affine transformation matrix [0, -1].
    y : FLOAT
        image affine transformation matrix [1, -1].

    """
    x = (L-1)/L * (sin + cos -1) - 2 * (M * sin + N * cos)/L
    y = (L-1)/L * (cos - sin -1) - 2 * (M * cos - N * sin)/L
    return x, y

def select_alpha(b, s=None, t = 20, base = 0.75):
    if s is None:
        s = torch.ones_like(b)
    mean = 0
    num = 0
    for j, ind in enumerate(b.abs().sort()[1]):
        new_mean = (mean * num + b[ind] * s[ind]) / (num + s[ind])
        if (new_mean - mean).abs() < t * (base ** (j - 1)):
            num += s[ind]
            mean = new_mean
        else:
            break
    return mean

def matrix_calculation_function(output, angle = "Auto", threshold = 0.3, freeze_scale = False, add_trans = True, show_print = False, coordinate = False):
    batch_size = output['score_map'].size(0)
    device = output['score_map'].device
    
    if isinstance(threshold, (float, int)):
        threshold = torch.tensor([threshold]).repeat(batch_size).float().to(device)#.view(-1, 1, 1)
    assert len(threshold) == batch_size, "threshold length {} not match to output size {}".format(len(threshold), batch_size)
    
    bottom = torch.Tensor([[[0, 0, 1.0]]]).repeat(batch_size, 1, 1).to(device)
    
    reshape_map = output['score_map'].size(1) == output['angle_map'].size(1) + 1
    
    conf = (output['score_map'][:,:-1, :-1] if reshape_map else output['score_map']) >= threshold.view(-1, 1, 1)
    L = int(conf.size(1) ** 0.5)
    longest = conf.sum(dim=[1,2]).max()
    angle_scale_list = []
    coordinate_1_list = []
    coordinate_2_list = []
    
    for i in range(batch_size):
        # add score
        score = (output['score_map'][i, :-1, :-1] if reshape_map else output['score_map'][i])[conf[i]]
        
        if angle == "Auto":
            cos = output['angle_map'][i][conf[i]][:, 0]
            sin = output['angle_map'][i][conf[i]][:, 1]
            alpha = (torch.sign(sin) * torch.acos(cos))#.median()

            max_order = score.sort()[1][-1]

            torch_pi = torch.acos(torch.tensor(-1)) 

            if alpha[max_order] < -0.75 * torch_pi and alpha.max() > 0.75 * torch_pi:
                alpha = alpha - (alpha.max() > 0.75 * torch_pi).float() * (2 * torch_pi)
            elif alpha[max_order] > 0.75 * torch_pi and alpha.min() < -0.75 * torch_pi:
                alpha = alpha + (alpha.min() < -0.75 * torch_pi).float() * (2 * torch_pi)

            alpha = select_alpha(
                alpha - alpha[max_order], 
                s = score, 
                t = torch_pi/10, 
                base = 0.9) + alpha[max_order]

            scale = output['scale_map'][i][conf[i]].mean().exp() if not freeze_scale else torch.tensor(1).to(device)
            
            angle_scale = torch.Tensor([
                [ torch.cos(alpha), torch.sin(alpha)],
                [-torch.sin(alpha), torch.cos(alpha)]
            ]).to(device) * scale
            angle_scale_list.append(angle_scale)
        elif isinstance(angle, torch.Tensor):
            
            angle = angle.view(-1)
            assert angle.size(0) == batch_size
            
            angle_scale = torch.tensor([
                [ torch.cos(angle[i]), torch.sin(angle[i])],
                [-torch.sin(angle[i]), torch.cos(angle[i])]
            ]).to(device)
            angle_scale_list.append(angle_scale)

        h,w = torch.where(conf[i])
        hx, hy, wx, wy = h // L, h % L, w // L, w % L
        coordinate_1 = torch.stack([hx, hy, torch.ones_like(hx)], dim = 0).repeat(1, torch.ceil(longest / len(hx)).int().item())[:, :longest].float()
        coordinate_2 = torch.cat([
            torch.stack([wx, wy], dim = 0).float() + (output['trans_map'][i][conf[i]].permute(1,0) if add_trans else 0), 
            torch.ones_like(wx).unsqueeze(0).float()
        ], dim = 0)
        coordinate_2 = coordinate_2.repeat(1, torch.ceil(longest / len(hx)).int().item())[:, :longest]
        coordinate_1_list.append(coordinate_1)
        coordinate_2_list.append(coordinate_2)

    coordinate_1 = torch.stack(coordinate_1_list, dim =0)
    coordinate_2 = torch.stack(coordinate_2_list, dim =0)
    
    if angle is None:
        translation = torch.zeros(batch_size, 3, 1).to(device)
    else:
        angle_scale_matrix = torch.stack(angle_scale_list, dim = 0)
        
        translation = torch.randn(batch_size,2,1).to(device)

    translation.requires_grad = True

    optimizer = torch.optim.Adam([translation], lr = 0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 400, gamma = 0.75)

    for i in range(3000):
        if angle is None:
            angle_scale_matrix = torch.cat([
                torch.cat([ torch.cos(translation[:,:1,:]), torch.sin(translation[:,:1,:]), translation[:,1:2,:]], dim = -1),
                torch.cat([-torch.sin(translation[:,:1,:]), torch.cos(translation[:,:1,:]), translation[:,2: ,:]], dim = -1)
            ], dim = 1)
    
            matrix = torch.cat([angle_scale_matrix, bottom], dim = 1)
        else:
            matrix = torch.cat([torch.cat([angle_scale_matrix, translation], dim=2), bottom], dim = 1) 
        
        generate = torch.einsum("bij, bjk -> bik", matrix, coordinate_1)
        loss = torch.nn.functional.smooth_l1_loss(generate, coordinate_2)

        if show_print and (i + 1) % 100 == 0:
            print(i, ":", loss.item())

        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
    
    affine_matrix = matrix.detach().clone()
    affine_matrix_list = []
    for i in range(batch_size):
        affine_matrix_list.append(matrix_transform_coordinate2image(affine_matrix[i], L))

    affine_matrix = torch.stack(affine_matrix_list, dim=0)
    if coordinate:
        matches = torch.cat([coordinate_2[:,:2,:], coordinate_1[:,:2,:]], dim = 1).permute(0,2,1)[:,:,torch.arange(3,-1,-1).long()]
        return affine_matrix, matches, conf.sum(dim=[1,2])
    else:
        return affine_matrix
        

def third_party_matrix_calculation_function(input_dict, L = 256, show_print = False):
    
    positive = ~torch.isnan(input_dict['coordinate_1'][:,0,0])
    
    coordinate_1 = input_dict['coordinate_1'][positive]
    coordinate_2 = input_dict['coordinate_2'][positive]
    
    assert coordinate_1.shape == coordinate_2.shape
    batch_size = coordinate_1.size(0)
    device = coordinate_1.device
    
    bottom = torch.Tensor([[[0, 0, 1.0]]]).repeat(batch_size, 1, 1).to(device)
    
    translation = torch.zeros(batch_size, 3, 1).to(device)

    translation.requires_grad = True

    optimizer = torch.optim.Adam([translation], lr = 0.02)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 400, gamma = 0.75)

    for i in range(3000):
        angle_scale_matrix = torch.cat([
            torch.cat([ torch.cos(translation[:,:1,:]), torch.sin(translation[:,:1,:]), translation[:,1:2,:]], dim = -1),
            torch.cat([-torch.sin(translation[:,:1,:]), torch.cos(translation[:,:1,:]), translation[:,2: ,:]], dim = -1)
        ], dim = 1)

        matrix = torch.cat([angle_scale_matrix, bottom], dim = 1)
        
        generate = torch.einsum("bij, bjk -> bik", matrix, coordinate_1)
        loss = torch.nn.functional.smooth_l1_loss(generate, coordinate_2)

        if show_print and (i + 1) % 100 == 0:
            print(i, ":", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    affine_matrix = matrix.detach().clone()
    affine_matrix_list = []
    
    i = 0
    for p in positive: # range(batch_size):
        if p:
            affine_matrix_list.append(matrix_transform_coordinate2image(affine_matrix[i], L))
            i += 1
        else:
            affine_matrix_list.append(torch.Tensor([float('nan')]).view(1,1).repeat(3,3))

    affine_matrix = torch.stack(affine_matrix_list, dim=0)
    return affine_matrix
