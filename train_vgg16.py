import config
import models
import keras
import os
import numpy as np
import pickle

from keras.applications.vgg16 import VGG16
from keras.layers import *
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam


def get_bottleneck(model, train_gen, val_gen):
	print("compute training data...")
	bottleneck_features_train = model.predict_generator(train_gen, train_gen.n)
	np.save(open('bottleneck_features_train.npy', 'w'), bottleneck_features_train)
	print("compute validation data...")
	bottleneck_features_val = model.predict_generator(val_gen, val_gen.n)
	np.save(open('bottleneck_features_val.npy', 'w'), bottleneck_features_val)

os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

input_shape = (224, 224)
batch_size = 32
train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)
test_gen = models.getTestData(target_size = input_shape)

model = models.getVGG16()
model.compile(
	loss=keras.losses.categorical_crossentropy,
	optimizer=Adam(lr=1e-5),
	metrics=['accuracy'])

filename = model.name + "_best.hdf5"
checkpoint = ModelCheckpoint(config.trained_dir + filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = History()
early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.002, patience=30, verbose=0, mode='auto')

callbacks_list = [checkpoint, history, early_stopping]

model.fit_generator(
	train_gen,
	steps_per_epoch=train_gen.n // batch_size,
	epochs=500, 
	validation_steps=val_gen.n // batch_size,
	callbacks=callbacks_list,
	validation_data=val_gen,
	)


filename = model.name + "_history"
with open(config.trained_dir + filename, 'wb') as file_pi:
	pickle.dump(history.history, file_pi)

print("Complete training.\n")

print("Metrics: \n" + model.metrics_names)
model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)

