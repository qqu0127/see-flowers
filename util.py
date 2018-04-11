import os
import numpy as np
import config
from sklearn.metrics import confusion_matrix
#import matplotlib.pyplot as plt
from shutil import copyfile, rmtree
from urllib import urlretrieve
import matplotlib.image as mpimg
from skimage.transform import resize, rescale

def download_file(url, dest=None, data_path = 'data'):
	if not dest:
		dest = os.path.join(data_path, url.split('/')[-1])
	urlretrieve(url, dest)
def move_files(dir_name, labels):
	cwd = os.path.dirname(os.path.realpath(__file__))
	cur_dir_path = os.path.join(config.data_dir, dir_name)
	if not os.path.exists(cur_dir_path):
		os.mkdir(cur_dir_path)
	for i in range(0, 102):
		class_dir = os.path.join(config.data_dir, dir_name, str(i))
		os.mkdir(class_dir)
	for label in labels:
		src = str(label[0])
		dst = os.path.join(cwd, config.data_dir, dir_name, label[1], src.split(os.sep)[-1])
		copyfile(src, dst)

def load_process_data(path, img_size = (224, 224), RESCALE = True):
    X = []
    y = []
    
    for n in os.listdir(path):
        p = os.path.join(path, n)
        for im in os.listdir(p):
            img = mpimg.imread(os.path.join(p, im))
            img_re = resize(img, img_size)
            if RESCALE:
                img_re = 1 - img_re / 255.0
            X.append(img_re)
            y.append(int(n))
    return X, y
'''
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    '''