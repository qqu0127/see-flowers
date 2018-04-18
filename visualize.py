import config
import models
import util
import keras
import os
import sys


model = models.getVGG16()
model.load_weights('trained/vgg19/vgg19_best_top12.hdf5')
path = "trained/temp/vgg19"

util.vis_max(model, path)