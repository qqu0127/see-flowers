# see-flowers
# Flower Classification with Small Sample Data Using Deep Convolutional Neural Networks

## Overview
This repo provides my work on flower image classification using deep convnet. I trained serveral models on [VGG 102 flower dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/) and better the previous state-of-art result, 0.868.  
The best top-1 accuracy achieved so far: 0.904.  
Note here that I follow the official protocol for dataset split.  
* training: 1020  
* validation: 1020  
* test: 6149  

## Objectives
* Study the transfer learning on mulitclass, fine grained image classification task.
* Study the visualization tools and techniques for deep convnets.
* Try to explain the convnet-based classifier using visualization.

## Usage
* Preparation
```python
python init.py # download the dataset and organize

```
* Training
```python
python train.py --model=[model_name]

```

## Environment
python 2.7  
keras 2  
(to be completed...)
## Models
  
### Baseline model
A simple 2-layer baseline convnet.  
acc: 0.312  
<img src="./img/learning%20curves/baseline_acc.png" width=400>
<img src="./img/learning%20curves/baseline_loss.png" width=400>
### VGG16
Fine tune VGG16 with weights pretrained on imagenet.  
acc: 0.757  
<img src="./img/learning%20curves/vgg16_acc.png" width=400>
<img src="./img/learning%20curves/vgg16_loss.png" width=400>

### VGG19
Fine tune VGG19 with weights pretrained on imagenet.  
acc: 0.781  
<img src="./img/learning%20curves/vgg19_acc.png" width=400>
<img src="./img/learning%20curves/vgg19_loss.png" width=400>

### Inception-v3
Fine tune inception-v3 with weights pretrained on imagenet.  
acc: 0.904  
<img src="./img/learning%20curves/inception-v3_acc.png" width=400>
<img src="./img/learning%20curves/inception-v3_loss.png" width=400>

## Visualization
### original
<img src="./img/vis/0_original.png" width=200 /><img src="./img/vis/4_ori.png" width=200 /><img src="./img/vis/8_ori.png" width=200 />

### saliency
<img src="./img/vis/0_saliency.png" width=200 /><img src="./img/vis/4_saliency.png" width=200 /><img src="./img/vis/8_saliency.png" width=200 />
### heatmap
<img src="./img/vis/0_heatmap.png" width=200 /><img src="./img/vis/4_heatmap.png" width=200 /><img src="./img/vis/8_heatmap.png" width=200 />
