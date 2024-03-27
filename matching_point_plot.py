# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 19:28:58 2023

@author: rw17789
"""

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

def plot_matches(im1, im2, matches, inliers=None, Npts=None, lines=True,
                 unnormalize=True, radius=5, dpi=150, sav_fig=False,
                 colors=None):
    # Read images and resize        
    if isinstance(im1, torch.Tensor):            

        im1 = im1.squeeze().add(1).mul(255/2).cpu().data.numpy().astype(np.uint8)
        im2 = im2.squeeze().add(1).mul(255/2).cpu().data.numpy().astype(np.uint8)

        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)
        
    elif isinstance(im1, np.ndarray):
        I1 = Image.fromarray(im1)
        I2 = Image.fromarray(im2)    
    elif isinstance(im1, str):
        I1 = Image.open(im1)
        I2 = Image.open(im2)
    else:
        I1 = im1
        I2 = im2
        
    w1, h1 = I1.size
    w2, h2 = I2.size 

    if h1 <= h2:
        scale1 = 1;
        scale2 = h1/h2
        w2 = int(scale2 * w2)
        I2 = I2.resize((w2, h1))
    else:
        scale1 = h2/h1
        scale2 = 1
        w1 = int(scale1 * w1)
        I1 = I1.resize((w1, h2))
    catI = np.concatenate([np.array(I1), np.array(I2)], axis=1)

    # Load all matches
    match_num = matches.shape[0]
    if inliers is None:
        if Npts is not None:
            Npts = Npts if Npts < match_num else match_num
        else:
            Npts = matches.shape[0]
        inliers = range(Npts) # Everthing as an inlier
    else:
        if Npts is not None and Npts < len(inliers):
            inliers = inliers[:Npts]
    print('Plotting inliers: ', len(inliers))

    x1 = scale1*matches[inliers, 0]
    y1 = scale1*matches[inliers, 1]
    x2 = scale2*matches[inliers, 2] + w1
    y2 = scale2*matches[inliers, 3]
    c = np.random.rand(len(inliers), 3) 
    
    if colors is not None:
        c = colors
    
    # Plot images and matches
    fig = plt.figure(figsize=(30, 20))
    axis = plt.gca()#fig.add_subplot(1, 1, 1)
    axis.imshow(catI, "gray")
    axis.axis('off')
    
    #plt.imshow(catI)
    #ax = plt.gca()
    for i, inid in enumerate(inliers):
        # Plot
        axis.add_artist(plt.Circle((x1[i], y1[i]), radius=radius, color=c[i,:]))
        axis.add_artist(plt.Circle((x2[i], y2[i]), radius=radius, color=c[i,:]))
        if lines:
            axis.plot([x1[i], x2[i]], [y1[i], y2[i]], c=c[i,:], linestyle='-', linewidth=radius)
    if sav_fig:        
        fig.savefig(sav_fig, dpi=dpi,  bbox_inches='tight')      
    plt.show()    