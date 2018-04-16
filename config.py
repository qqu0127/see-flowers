from os.path import join as join_path
import os
import labels
import numpy as np
import pandas

data_dir = './data/sorted/'
trained_dir = None

train_dir = None
val_dir = None
test_dir = None

MODEL_VGG16 = "vgg16"
MODEL_VGG19 = "vgg19"
MODEL_INCEPTION_V3 = "inception_v3"
MODEL_RESNET50 = "resnet50"
MODEL_RESNET152 = "resnet152"
MODEL_BASELINE = "baseline"

MODEL_LIST = [MODEL_VGG16, MODEL_VGG19, MODEL_INCEPTION_V3, MODEL_RESNET50, MODEL_RESNET152]

classes = labels.labels


def set_paths():
    global train_dir, val_dir, test_dir, trained_dir
    train_dir = join_path(data_dir, 'train/')
    val_dir = join_path(data_dir, 'valid/')
    test_dir = join_path(data_dir, 'test/')
    trained_dir = './trained/'


set_paths()