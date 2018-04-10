import config
import models
import keras
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import History
from keras.optimizers import Adam


os.environ['MKL_NUM_THREADS'] = '4'
os.environ['GOTO_NUM_THREADS'] = '4'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['openmp'] = 'True'
input_shape = (224, 224)

train_data = models.getTrainData(data_aug=True, target_size=input_shape)
val_data = models.getValData(data_aug=True, target_size=input_shape)

model = models.getVGG16()
model.compile(loss=keras.losses.categorical_crossentropy,
	optimizer=Adam(lr=1e-4),
	metrics=['accuracy'])

filename = model.name + "_best.hdf5"
checkpoint = ModelCheckpoint(config.trained_dir + filename, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
history = History()

callbacks_list = [checkpoint, history]

model.fit_generator(
	train_data,
	steps_per_epoch=64,
	epochs=500,
	validation_steps=64,
	callbacks=callbacks_list,
	validation_data=val_data,
	)

import pickle
filename = model.name + "_history"
with open(config.trained_dir + filename, 'wb') as file_pi:
	pickle.dump(history.history, file_pi)