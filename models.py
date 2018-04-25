import keras
from keras.applications.vgg16 import VGG16
from keras.applications.vgg19 import VGG19
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50

from keras.layers import *
from keras.models import Model, Sequential
import numpy as np
from keras.utils import plot_model
import config
from keras.preprocessing.image import ImageDataGenerator

def getVGG16(new_layer_size=4096, new_layer_name='fc2'):
	base = VGG16(weights='imagenet', include_top = False, input_tensor=Input(shape=(224, 224, 3)))
	for layer in base.layers:
		layer.trainable = False
	x = base.output
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu')(x)
	x = Dropout(0.6)(x)
	x = Dense(new_layer_size, activation='relu', name=new_layer_name)(x)
	x = Dropout(0.6)(x)
	x = Dense(len(config.classes), activation='softmax', name='predictions')(x)

	model = Model(inputs=base.input, outputs=x, name='vgg16')
	return model

def getVGG19(new_layer_size=4096, new_layer_name='fc2'):
	base = VGG19(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
	for layer in base.layers:
		layer.trainable = False
	x = base.output
	x = Flatten(name='flatten')(x)
	x = Dense(4096, activation='relu', name='fc1')(x)
	x = Dropout(0.6)(x)
	x = Dense(4096, activation='relu', name='fc2')(x)
	x = Dropout(0.6)(x)
	x = Dense(len(config.classes), activation='softmax', name='predictions')(x)

	model = Model(input=base.input, outputs=x, name='vgg19')
	return model


def getInceptionV3(new_layer_size=4096, new_layer_name='fc1'):
	base = InceptionV3(weights='imagenet', include_top = False, input_tensor=Input(shape=(299, 299, 3)))
	for layer in base.layers:
		layer.trainable = False
	x = base.output
	x = GlobalAveragePooling2D(name='avg_pool')(x)
	#x = Dense(new_layer_size, activation='elu', name=new_layer_name)(x)
	#x = Dropout(0.7)(x)
	x = Dense(len(config.classes), activation='softmax', name='predictions')(x)

	model = Model(inputs=base.input, outputs=x, name='inception-v3')
	return model

def getResNet50(new_layer_size=4096, new_layer_name='fc1'):
	base = ResNet50(weights='imagenet', include_top=False, input_tensor=Input(shape=(224, 224, 3)))
	for layer in base.layers:
		layer.trainable = False
	x = base.output
	x = Flatten(name='flatten')(x)
	x = Dense(len(config.classes), activation='softmax', name='predictions')(x)

	model = Model(inputs=base.input, outputs=x, name='ResNet50')
	return model

def getBaseline():
	img_input = Input(shape=(32, 32, 3))
	x = Conv2D(16, (3, 3), activation = 'relu', padding='same', name='conv1_1')(img_input)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling_1')(x)
	x = Dropout(0.25)(x)


	x = Conv2D(32, (3, 3), activation = 'relu', padding='same', name='conv2_1')(x)
	x = MaxPooling2D((2, 2), strides=(2, 2), name='maxpooling_2')(x)
	x = Dropout(0.25)(x)

	x = Flatten(name='flatten')(x)
	x = Dense(1024, activation='relu', name='fc1')(x)
	x = Dropout(0.25)(x)
	x = Dense(102, activation='softmax', name='predictions')(x)

	model = Model(inputs=img_input, outputs=x, name='baseline_model')
	return model


def getDataGen(data_aug = True):
	if data_aug:
		idg = ImageDataGenerator(
			rescale=1./255, 
			rotation_range=30., 
			shear_range=0.2, 
			zoom_range=0.2,
			horizontal_flip=True,
			width_shift_range=0.2,
			height_shift_range=0.2,
			zca_whitening=True,
			vertical_flip=True
			)
	else:
		idg = ImageDataGenerator(rescale=1./255)
	#idg.mean = np.array([103.939, 116.779, 123.68], dtype=np.float32).reshape((3, 1, 1))
	return idg

def getTrainData(batch_size = 32, data_aug = True, target_size = (224, 224)):
	idg = getDataGen(data_aug = data_aug)
	return idg.flow_from_directory(config.test_dir, batch_size=batch_size, target_size = target_size)
	
def getValData(batch_size = 32, data_aug = True, target_size = (224, 224)):
	idg = getDataGen(data_aug = data_aug)
	return idg.flow_from_directory(config.val_dir, batch_size=batch_size, target_size = target_size)

def getTestData(target_size = (224, 224)):
	idg = getDataGen(data_aug = False)
	return idg.flow_from_directory(config.test_dir, target_size = target_size, shuffle = True, seed = 66)

if __name__ == '__main__':
	train_data = getTrainData(batch_size = 32, data_aug=True, target_size=(32, 32))
	val_data = getValData(batch_size = 32, data_aug=True, target_size=(32, 32))
	model = baseline_model()
	model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.SGD(lr=0.01),
              metrics=['accuracy'])

	model.fit_generator(
        train_data,
        steps_per_epoch=2000,
        epochs=50,
        validation_data=val_data,
        validation_steps=800)

