import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input
from keras import activations
from keras.models import load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras import initializers
from keras.models import Sequential, Model

from keras.applications import imagenet_utils
import numpy as np

import h5py as h5py
from vis.visualization import visualize_activation,visualize_saliency,overlay,visualize_cam
from vis.utils import utils



path = "trained/vis/"
model_origin = InceptionV3(weights='imagenet', include_top=True, input_tensor=Input(shape=(299, 299, 3)))

layer_idx = utils.find_layer_idx(model_origin, 'predictions')
print("Remove Activation from Last Layer")
# Swap softmax with linear
model_origin.layers[layer_idx].activation = activations.linear
print("Done. Now Applying changes to the model ...")
model_origin = utils.apply_modifications(model_origin)

im_file="trained/vis/husky.jpg"
img1 = image.load_img(im_file,target_size=(299,299))
img1 = image.img_to_array(img1)
img1 = np.expand_dims(img1, axis=0)
img1 = preprocess_input(img1)
print(img1.shape)
heatmap = visualize_cam(model_origin, layer_idx, filter_indices=248, seed_input=img1[0,:,:,:])
img_init=utils.load_img(im_file,target_size=(299,299))
heatmap = overlay(heatmap, img_init)
mpimg.imsave(path + "original.png", img_init)
mpimg.imsave(path + "heatmap.png", heatmap)




