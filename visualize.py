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
	layer_ind = utils.find_layer_idx(model, 'predictions')
	model.layer[layer_ind].activation = activations.linear
	model = utils.apply_modifications(model)
	for ind in class_ind:
		print 'Generating for class {}'.format(ind)
		img = visualize_activation(model, layer_idx, filter_indices=ind)
		mpimg.imsave(save_path + str(ind) + ".png", img)
	print('Done')

def get_attention_map(model, img, ind = None, modifier = 'guided', SAVE_FLAG = False, path = None):
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
	if(np.mean(img) < 1):
		img = img * 255.0

	layer_idx = utils.find_layer_idx(model, 'predictions')
	#model.layers[layer_idx].activation = activations.linear
	#model = utils.apply_modifications(model)
	grads = visualize_saliency(model, layer_idx, filter_indices = ind, seed_input=img, backprop_modifier=modifier)
	heatmap = visualize_cam(model, layer_idx, filter_indices = ind, seed_input=img, backprop_modifier=modifier)
	heatmap = overlay(heatmap, img)

	if SAVE_FLAG:
		mpimg.imsave(path + "original.png", img)
		mpimg.imsave(path + "saliency.png", grads)
		mpimg.imsave(path + "heatmap.png", heatmap)
	return {'saliency' : grads, 'heatmap' : heatmap}



if __name__ == '__main__':
	vgg16 = models.getVGG16()
	vgg16.load_weights('trained/inception-v3/inception-v3_best_412.hdf5')
	path = "trained/vis/attention_map/inception-v3/"

	test_gen = models.getTestData(target_size = (299, 299))
	(x_, y_) = test_gen.next()
	for i in range(x_.shape[0]):
		print("Generating " + str(i))
		img = x_[i, :, :, ]
		ind = np.argmax(y_[i, :])
		res = get_attention_map(vgg16, img, None)
		mpimg.imsave(path + str(i) + "_ori.png", img)
		mpimg.imsave(path + str(i) + "_saliency.png", res['saliency'])
		mpimg.imsave(path + str(i) + "_heatmap.png", res['heatmap'])

