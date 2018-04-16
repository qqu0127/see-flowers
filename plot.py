import config
import models
import util
from keras.callbacks import ModelCheckpoint
import os

import pickle
import matplotlib.pyplot as plt

model_name = 'VGG16'
load_path = config.trained_dir + '/vgg16/vgg16_history'
save_path = config.trained_dir + '/vgg16/'
with open(load_path, 'rb') as file_pi:
    h = pickle.load(file_pi)



plt.plot(h['acc'])
plt.plot(h['val_acc'])
plt.title(model_name + ' model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(save_path + '/' + model_name + '_acc.png')
plt.close()

plt.plot(h['loss'])
plt.plot(h['val_loss'])
plt.title(model_name + ' model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.savefig(save_path + '/' + model_name + '_loss.png')
plt.close()
