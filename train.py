import config
import models
import keras
import os
import sys
import argparse
import numpy as np
import pickle

from keras.layers import *
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam



def get_callbacks(filename):
	checkpoint = ModelCheckpoint(config.trained_dir + filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	history = History()
	early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.002, patience=30, verbose=0, mode='auto')
	callbacks_list = [checkpoint, history, early_stopping]

	return callbacks_list

def train_vgg16():
	input_shape = (224, 224)
	batch_size = 32
	train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
	val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)
	test_gen = models.getTestData(target_size = input_shape)

	model = models.getVGG16()
	model.compile(
		loss=keras.losses.categorical_crossentropy,
		optimizer=Adam(lr=1e-4),
		metrics=['accuracy'])
	filename = model.name + "_best.hdf5"
	callbacks_list = get_callbacks(filename)

	model.fit_generator(
		train_gen,
		steps_per_epoch=train_gen.n // batch_size,
		epochs=200, 
		validation_steps=val_gen.n // batch_size,
		callbacks=callbacks_list,
		validation_data=val_gen,
		)
	filename = model.name + "_history"
	with open(config.trained_dir + filename, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)
	print("Complete training.")

	print("Metrics: ")
	print(model.metrics_names)
	met = model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)
	print(met)
	return met

def train_vgg19():
	input_shape = (224, 224)
	batch_size = 32
	train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
	val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)
	test_gen = models.getTestData(target_size = input_shape)

	model = models.getVGG19()
		
	model.compile(
		loss=keras.losses.categorical_crossentropy,
		optimizer=Adam(lr=1e-4),
		metrics=['accuracy'])

	filename = model.name + "_best.hdf5"
	callbacks_list = get_callbacks(filename)

	model.fit_generator(
		train_gen,
		steps_per_epoch=train_gen.n // batch_size,
		epochs=200, 
		validation_steps=val_gen.n // batch_size,
		callbacks=callbacks_list,
		validation_data=val_gen,
		)

	filename = model.name + "_history"
	with open(config.trained_dir + filename, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	print("Complete training.")

	print("Metrics: ")
	print(model.metrics_names)
	met = model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)
	print(met)
	return met

def train_inception_v3():
	input_shape = (299, 299)
	batch_size = 32
	train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
	val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)
	test_gen = models.getTestData(target_size = input_shape)

	model = models.getInceptionV3()
	model.compile(
		loss=keras.losses.categorical_crossentropy,
		optimizer=Adam(lr=1e-5),
		metrics=['accuracy'])

	filename = model.name + "_best.hdf5"
	callbacks_list = get_callbacks()

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

	print("Metrics: ")
	print(model.metrics_names)
	met = model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)
	print(met)
	return met
	
def train_baseline_model():
	input_shape = (32, 32)
	batch_size = 32
	train_gen = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
	val_gen = models.getValData(batch_size, data_aug=True, target_size=input_shape)
	test_gen = models.getTestData(target_size = input_shape)

	model = models.baseline_model()
	model.compile(loss=keras.losses.categorical_crossentropy,
		optimizer='rmsprop',
		metrics=['accuracy'])

	filename = model.name + "_best.hdf5"
	checkpoint = ModelCheckpoint(config.trained_dir + filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
	history = History()
	early_stopping = EarlyStopping(monitor='val_acc', min_delta=0.002, patience=10, verbose=0, mode='auto')

	callbacks_list = [checkpoint, history, early_stopping]

	model.fit_generator(
		train_gen,
		steps_per_epoch=train_gen.n // batch_size,
		epochs=100,
		validation_steps=val_gen.n // batch_size,
		callbacks=callbacks_list,
		validation_data=val_gen,
		)
	filename = model.name + "_history"
	with open(config.trained_dir + filename, 'wb') as file_pi:
		pickle.dump(history.history, file_pi)

	print("Complete training.\n")

	print("Metrics: ")
	print(model.metrics_names)
	met = model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)
	print(met)
	return met

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=str, required=True, help='Base model architecture', choices=config.MODEL_LIST)
	args = parser.parse_args()
	{
		config.MODEL_VGG16: train_vgg16(),
		config.MODEL_VGG19: train_vgg19(),
		config.MODEL_INCEPTION_V3: train_inception_v3(),
		config.MODEL_BASELINE: train_baseline_model(),
		config.MODEL_RESNET50: train_resnet50()
	}[args.model]


