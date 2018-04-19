import config
import models
import util
import keras
import os
import sys
import models
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from vis.visualization import visualize_saliency, overlay, visualize_cam
from vis.utils import utils
from keras import activations
import numpy as np
import matplotlib.cm as cm


model = models.getVGG16()
model.load_weights('./trained/vgg16/vgg16_best_top12.hdf5')
input_shape = (224, 224)
batch_size = 32

test_gen = models.getTestData(target_size = input_shape)
(x_, y_) = test_gen.next()

img = x_[0,:,:,:]

modifier = 'guided'
layer_idx = utils.find_layer_idx(model, 'predictions')

grads = visualize_saliency(model, layer_idx, filter_indices=20, 
	seed_input=img, backprop_modifier=modifier)

mpimg.imsave("./trained/temp/img.png", img)
mpimg.imsave("./trained/temp/grads.png", grads)

heatmap = visualize_cam(model, layer_idx, filter_indices=20, 
	seed_input=img, backprop_modifier=modifier)

heatmap = overlay(heatmap / 255.0, img)

mpimg.imsave("./trained/temp/heatmap.png", heatmap)


