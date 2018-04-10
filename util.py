import os
import numpy as np
import config
from shutil import copyfile, rmtree
from urllib import urlretrieve

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