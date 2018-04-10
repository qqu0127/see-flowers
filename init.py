import os
import numpy as np
import sys
import tarfile
import util
import glob
import config
from shutil import copyfile, rmtree
from scipy.io import loadmat


data_path = 'data'
trained_path = './trained'
base_url = 'http://www.robots.ox.ac.uk/~vgg/data/flowers/102/'

if not os.path.exists(data_path):
	os.mkdir(data_path)

if not os.path.isdir(trained_path):
	os.mkdir(trained_path)

flowers_archive_path = os.path.join(data_path, '102flowers.tgz')
img_label_path = os.path.join(data_path, 'imagelabels.mat')
setid_path = os.path.join(data_path, 'setid.mat')

if not os.path.isfile(flowers_archive_path):
	print ('Downloading images...')
	util.download_file(base_url + '102flowers.tgz')
else:
	print("Images data already existed\n")

if not os.path.isdir('./data/jpg'):
	print("Unzip the images files...")
	tarfile.open(flowers_archive_path).extractall(path=data_path)

if not os.path.isfile(img_label_path):
    print("Downloading image labels...")
    util.download_file(base_url + 'imagelabels.mat')
else:
	print("Image labels already existed\n")
if not os.path.isfile(setid_path):
	print("Downloading train/test/valid splits...")
	util.download_file(base_url + 'setid.mat')
else:
	print("Set split already existed\n")

setid = loadmat(setid_path)

# Read .mat file containing image labels.
img_labels = loadmat(img_label_path)['labels'][0]
img_labels -= 1


files = sorted(glob.glob(os.path.join(data_path, 'jpg', '*.jpg')))
labels = np.array([i for i in zip(files, img_labels)])

if os.path.exists(config.data_dir):
    rmtree(config.data_dir, ignore_errors=True)
os.mkdir(config.data_dir)


idx_train = setid['trnid'][0] - 1
idx_test = setid['tstid'][0] - 1
idx_valid = setid['valid'][0] - 1

util.move_files('train', labels[idx_test, :])
util.move_files('test', labels[idx_train, :])
util.move_files('valid', labels[idx_valid, :])