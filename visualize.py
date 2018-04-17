import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
from vis.utils import utils as vutils
from keras import activations
from vis.visualization import visualize_activation

import config
import models
import keras
import os
import sys
import argparse
import numpy as np
import pickle
from keras import backend as K

from keras.layers import *
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


model = models.getVGG16()
model.load_weights('trained/vgg16/vgg16_best.hdf5')
layer_idx = vutils.find_layer_idx(model, 'predictions')
model.layers[layer_idx].activation = activations.linear
model = vutils.apply_modifications(model)

for ind in range(len(config.classes)):
	plt.rcParams['figure.figsize'] = (18, 6)
	img = visualize_activation(model, layer_idx, filter_indices=ind)
	mpimg.imsave("trained/vis/vgg16/" + str(ind) + ".png", img)
