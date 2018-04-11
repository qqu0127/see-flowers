import config
import models
import keras
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

from keras.layers import *
from keras.models import Model, Sequential
import numpy as np
from keras.utils import plot_model
import config
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import History
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'
input_shape = (224, 224)
batch_size = 32
train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)

model = VGG16(weights='imagenet', include_top = False, input_tensor=Input(shape=(224, 224, 3)))
#model.compile(loss=keras.losses.categorical_crossentropy,	optimizer=Adam(lr=1e-4),	metrics=['accuracy'])

#bottleneck_features_train = model.predict_generator(train_gen, train_gen.n)
bottleneck_features_val = model.predict_generator(val_gen, val_gen.n)
np.save(open('bottleneck_features_val.npy', 'w'), bottleneck_features_val)
#np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)