# RoTIR: Rotation-Equivariant Network and Transformers for Zebraish Scale Image Registration

This is a PyTorch implementation of the RoTIR model.

## Author
* Ruixiong Wang, Visual Information Laboratory, University of Brsitol
* Alin Achim, Visual Information Laboratory, University of Brsitol
* Stephen Cross, Wolfson Bioimaging Facility, University of Brsitol
 
## Requirements:
* Python == 3.7.10
* torch==1.8.1
* torchvision==0.9.1
* pillow==10.0.1
* numpy==1.26.0
* matplotlib==3.8.0
* e2cnn==0.2.2
* einops==0.6.0
* [pyssim](https://github.com/jterrace/pyssim.git)

## Workflow and Network Architecture

<img src="_readme/fig_1.PNG" width=600> 

## Dataset Synthesis

<img src="_readme/fig_2.PNG" width=600> 

## Results
1. Result table
<img src="_readme/tab_1.PNG" width=400> 
2. Transformed results
<img src="_readme/fig_3.PNG" width=600> 
3. Feature matching
<img src="_readme/fig_4.PNG" width=600> 

## Test

Zebrafish scale image registration test is present in [Fish scale registration whole process.ipynb](https://github.com/SpikeRXWong/RoTIR-Zebrafish-Scale-Image-Resigtration/blob/main/Fish%20scale%20registration%20whole%20process.ipynb)
