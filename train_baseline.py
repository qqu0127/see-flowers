import config
import models
import keras
from keras.callbacks import ModelCheckpoint
import os
from keras.callbacks import History, EarlyStopping
from keras.optimizers import Adam
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

input_shape = (32, 32)
batch_size = 32
train_data = models.getTrainData(batch_size, data_aug=True, target_size=input_shape)
val_data = models.getValData(batch_size, data_aug=True, target_size=input_shape)

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
	train_data,
	steps_per_epoch=train_data.n // batch_size,
	epochs=100,
	validation_steps=val_data.n // batch_size,
	callbacks=callbacks_list,
	validation_data=val_data,
	)

import pickle
filename = model.name + "_history"
with open(config.trained_dir + filename, 'wb') as file_pi:
	pickle.dump(history.history, file_pi)