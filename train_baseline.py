import config
import models
import keras
import os
import pickle
from keras.callbacks import History, EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam
os.environ['MKL_NUM_THREADS'] = '16'
os.environ['GOTO_NUM_THREADS'] = '16'
os.environ['OMP_NUM_THREADS'] = '16'
os.environ['openmp'] = 'True'

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
model.evaluate_generator(generator=test_gen, use_multiprocessing=True, workers=6)

