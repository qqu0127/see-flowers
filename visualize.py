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

def get_act_max(model, save_path,
	class_ind = range(len(config.classes))):
	'''
		Using keras-vis to generate activation maximization for each class.
		It generates an input image that maximize the probability of being classified to certain class.
		For more information, please see https://raghakot.github.io/keras-vis/visualizations/activation_maximization/

		@input: 
			model -- keras model with loaded weights, compiled
			save_path -- the directory to save the generated images
			class_ind -- (optional) an array, the specific class indexes to generate the act_max image

		output: None

	'''
	layer_ind = vutils.find_layer_idx(model, 'predictions')
	model.layer[layer_ind].activation = activations.linear
	model = vutils.apply_modifications(model)
	for ind in class_ind:
		print 'Generating for class {}'.format(ind)
		img = visualize_activation(model, layer_idx, filter_indices=ind)
		mpimg.imsave(save_path + str(ind) + ".png", img)
	print('Done')

def get_attention_map(model, img, modifier = 'guided', SAVE_FLAG = False, path = None):
	'''
		Given an image and a model, using keras-vis, generate the saliency map and heatmap of that image.
		This is for the study of the decision making of classifier, which parts of the image have large 
		influence on the class category and loss function.
		For more information, please see https://raghakot.github.io/keras-vis/

		@input:
			model -- keras model with loaded weights, compiled
			img -- an image with the shape as required by the model
		
		@output:
			res -- an dictionary {'saliency': [[sal]], 'heatmap': [[heatmap]]}
					that stores the saliency map and heatmap images

	'''
	if(np.mean(img) > 1):
		img = img / 255.0

	layer_idx = utils.find_layer_idx(model, 'predictions')
	grads = visualize_saliency(model, layer_idx, filter_indices = None, seed_input=img, backprop_modifier=modifier)
	heatmap = visualize_cam(model, layer_idx, filter_indices = None, seed_input=img, backprop_modifier=modifier)
	heatmap = overlay(heatmap / 255.0, img)

	if SAVE_FLAG:
		mpimg.imsave(path + "original.png", img)
		mpimg.imsave(path + "saliency.png", grads)
		mpimg.imsave(path + "heatmap.png", heatmap)
	return {'saliency' : grads, 'heatmap' : heatmap}



if __name__ == '__main__':
	model = models.getVGG16()
	model.load_weights('trained/vgg16/vgg16_best_top12.hdf5')
	path = "trained/vis/"
	test_gen = models.getTestData(target_size = (224, 224))
	(x_, y_) = test_gen.next()
	img = x_[0, :,:,:]

	get_attention_map(model, img, 'guided', True, path)

