import os
import numpy as np
import pandas as pd
import feather
import torch
from torchvision import transforms
from PIL import Image


def request_labels(wanted_list, df):
	"""
	get labels for images in the wanted list
	:param wanted_list: list of str, the images' names
	:param df: dataframe, dataframe containing lables
	:return: 1-d tensor, the labels
	"""
	labels = []
	for id in wanted_list:
		labels.append(df.loc[df['id'] == id, 'label'].to_numpy(dtype=np.int)[0])
	return torch.as_tensor(labels)


def read_images(dir):
	"""
	read and normalize images
	:param dir: str, directory path
	:return: 4-d tensor, the images
	"""
	imagenet_mean = np.asarray([0.485, 0.456, 0.406])
	imagenet_std = np.asarray([0.229, 0.224, 0.225])
	
	list_imgs = []
	for filename in os.listdir(dir):
		if filename.endswith('.png'):
			img = Image.open(os.path.join(dir, filename))
			rgbimg = img.convert('RGB')
			rgbimg = np.asarray(rgbimg) / 255.0
			normalized_rgbimg = rgbimg - imagenet_mean / imagenet_std
			list_imgs.append(normalized_rgbimg)
	return torch.stack(list_imgs)
			

def main():
	dir_path = os.path.dirname(os.path.realpath(__file__))
	# get labels
	labels_train_path = os.path.join(dir_path, 'data', 'labels_train.feather')
	labels_test_path = os.path.join(dir_path, 'data', 'labels_test.feather')
	train_df = feather.read_dataframe(labels_train_path)
	test_df = feather.read_dataframe(labels_test_path)
	# get images dir
	train_images_dir = os.path.join(dir_path, 'data', 'mnist_sample', 'train')
	test_images_dir = os.path.join(dir_path, 'data', 'mnist_sample', 'test')
	train_images = read_images(train_images_dir)
	test_images = read_images(test_images_dir)
	print(train_images.shape)
	
	# TODO: First step
	
	# TODO: Second step
	
	# TODO: Last step
	

if __name__ == '__main__':
	main()