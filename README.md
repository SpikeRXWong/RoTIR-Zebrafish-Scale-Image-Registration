# RoTIR: Rotation-Equivariant Network and Transformers for Zebraish Scale Image Registration

This is a PyTorch implementation of the RoTIR model.

DOI: [10.1007/978-3-031-66955-2_20](https://link.springer.com/chapter/10.1007/978-3-031-66955-2_20)

Presentation Video:

[![RoTIR Presentation at MIUA 2024](_readme/MIUA_RoTIR_presentation_(Title).png)](https://www.youtube.com/watch?v=wc4t63IJFiE&list=PLFm8siRFfuk1N__sqp0EkgIzrecF-YW7i)

## Author
* Ruixiong Wang, Visual Information Laboratory, University of Bristol
* Alin Achim, Visual Information Laboratory, University of Bristol
* Renata Raele-Rolfe, School of Biochemistry, University of Bristol
* Qiao Tong, School of Physiology, Pharmacology and Neuroscience, University of Bristol
* Dylan Bergen, School of Physiology, Pharmacology and Neuroscience, University of Bristol
* Chrissy Hammond, School of Physiology, Pharmacology and Neuroscience, University of Bristol
* Stephen Cross, Wolfson Bioimaging Facility, University of Bristol
 
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

## Train

-Code for training RoTIR model
```bash
python train.py -l 1e-3 -n 25000 -s 512 -loss_weight 1 10 20 20 --score_weight 1 500 1
```

## Test

Test of zebrafish scale image registration on 7 unprocessed [image pairs](https://github.com/SpikeRXWong/RoTIR-Zebrafish-Scale-Image-Resigtration/tree/main/test_image) is present in [Fish scale registration whole process.ipynb](https://github.com/SpikeRXWong/RoTIR-Zebrafish-Scale-Image-Resigtration/blob/main/Fish%20scale%20registration%20whole%20process.ipynb), using [pre-trained model](https://github.com/SpikeRXWong/RoTIR-Zebrafish-Scale-Image-Resigtration/tree/main/pre_trained)

## Citation 

```bash
@InProceedings{10.1007/978-3-031-66955-2_20,
  author="Wang, Ruixiong
  and Achim, Alin
  and Raele-Rolfe, Renata
  and Tong, Qiao
  and Bergen, Dylan
  and Hammond, Chrissy
  and Cross, Stephen",
  editor="Yap, Moi Hoon
  and Kendrick, Connah
  and Behera, Ardhendu
  and Cootes, Timothy
  and Zwiggelaar, Reyer",
  title="RoTIR: Rotation-Equivariant Network and Transformers for Zebrafish Scale Image Registration",
  booktitle="Medical Image Understanding and Analysis",
  year="2024",
  publisher="Springer Nature Switzerland",
  address="Cham",
  pages="285--299",
  abstract="Image registration is essential for aligning features of interest from multiple images. With the recent development of deep learning techniques, image registration approaches have advanced to a new level. In this work, we present Rotation-Equivariant network and Transformers for Image Registration (RoTIR), a deep-learning-based method for aligning zebrafish scale images captured by light microscopy. This approach overcomes the challenge of arbitrary rotation, translation detection, and the absence of ground truth data. We employ feature-matching approaches based on Transformers and general E(2)-equivariant steerable CNNs for model creation. Besides, an artificial training dataset is employed for semi-supervised learning. Results show that RoTIR successfully achieves the goal of zebrafish scale image registration.",
  isbn="978-3-031-66955-2"
}
```
