# CFNet: Automatic Multi-Modal Brain Tumor Segmentation through Hierarchical Coarse-to-Fine Fusion and Feature Communication
## Overview
![image](https://github.com/CYlala/CFNet/assets/110222769/c6e01214-6da6-4959-aee2-8c5e1270037a)
Figure 1: Our network project, named CFNet, is a deep learning endeavor focused on segmenting brain tumors in MRI images. It utilizes a convolutional neural network (CNN) to merge multiple MRI sequences and automatically identify and segment tumor regions. We pair these modalities and input them into two independent but structurally identical encoders. Additionally, we introduce a Modality-Cross Attention Fusion Module to integrate features from multiple modalities. Furthermore, we propose a Multi-scale Context Perception Module to enhance feature semantic perception.
## Dataset
downloading BraTS 2019 dataset which can be found from [Here](https://www.med.upenn.edu/cbica/brats2019/data.html "https://www.med.upenn.edu/cbica/brats2019/data.html") .
downloading BraTS 2020 dataset which can be found from [Here](https://www.med.upenn.edu/cbica/brats2020/data.html "https://www.med.upenn.edu/cbica/brats2020/data.html") .
## Structure of the repo
*`utils`：helper functions for io, loggings, training, visualizations.
  *`utils/dataset.py`:dataset preprocessing file
  *`utils/util.py`:helper functions for io, loggings, training, visualizations, validating
*`list`：txt file directory of the data
*`dataset`：loading and setup input datasets. 
*`model`：the model of the framework is proposed
*`lossfunction.py`:loss function
*`metirc.py`:model evaluation index
*`visual.py`:segmentation result visualization file
## Preparation
Build the environment with `python3.6.5` and `torch1.8.0`.
## Train
In the training process, you may need to adjust the parameters to adapt to your device, such as the batch size and the learning rate.
```
cd CFNet
python train.py
```
## Test
```
cd CFNet
python test.py
```
## Pre-trained models
| Model | Weights | 
| --- | --- |
| BraTs2019 |[Baidu](https://pan.baidu.com/s/1DLHHuENBpzKjS0l5eFcLRw "https://pan.baidu.com/s/1DLHHuENBpzKjS0l5eFcLRw ") (password:1234) |
| BraTS2020 | [Baidu](https://pan.baidu.com/s/1DLHHuENBpzKjS0l5eFcLRw "https://pan.baidu.com/s/1DLHHuENBpzKjS0l5eFcLRw ") (password:1234) |
## Experimental results
![image](https://github.com/CYlala/CFNet/assets/110222769/a775b949-0138-495e-a2fe-8976a088d943)
<p align="center">  Figure 2:Visual comparison results on the BraTS2019 benchmark. </p>

![image](https://github.com/YaruC/CFNet/assets/160707518/6be8ae5b-859f-4431-8d85-5e74707018b6)
<p align="center">  Figure 3:Visual comparison results on the BraTS2020 benchmark. </p>

![image](https://github.com/YaruC/CFNet/assets/160707518/635530f3-bac8-45d4-8a16-0a4d4986eec6)

![image](https://github.com/YaruC/CFNet/assets/160707518/9f58deff-8ba5-451d-9f23-5e81819024cb)



