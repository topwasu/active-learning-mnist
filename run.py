import os
import time
import numpy as np
import pandas as pd
import feather
import torch
from torchvision import models
from PIL import Image


def request_labels(wanted_list, df):
	"""
	get labels for images in the wanted list
	please make sure that all images in the wanted list still do not have labels
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
	read, normalize, and upsample images
	:param dir: str, directory path
	:return: 4-d tensor and list of str, the images and their names
	"""
	imagenet_mean = np.asarray([0.485, 0.456, 0.406])
	imagenet_std = np.asarray([0.229, 0.224, 0.225])
	
	list_imgs = []
	list_imgs_names = []
	for filename in os.listdir(dir):
		if filename.endswith('.png'):
			list_imgs_names.append(filename)
			
			img = Image.open(os.path.join(dir, filename))
			rgbimg = img.convert('RGB')
			rgbimg = np.asarray(rgbimg) / 255.0
			normalized_rgbimg = rgbimg - imagenet_mean / imagenet_std
			list_imgs.append(torch.from_numpy(normalized_rgbimg))
	list_imgs = torch.stack(list_imgs)
	return torch.nn.functional.interpolate(list_imgs, (224, 224)), list_imgs_names


def calc_accuracy(logits, labels):
	"""
	calculate accuracy of the predicted labels
	:param logits: tensor
	:param labels: tensor
	:return: float, accuracy
	"""
	logits = logits.cpu().detach().numpy()
	labels = labels.cpu().detach().numpy()
	predicted_labels = np.argmax(logits, axis=1)
	return np.mean(np.equal(predicted_labels, labels))


def train(images, labels, model, num_epochs, lr=1e-3, batch_size=64, verbose=False):
	"""
	train the model
	:param images: 4-d tensor, input images
	:param labels: 1-d tensor, input labels
	:param model: model
	:param num_epochs: int, number of training epochs
	:param lr: float, learning rate
	:param batch_size: int, size of batch
	:param verbose: boolean
	"""
	model = model.train()
	optimizer = torch.optim.Adam(model.parameters(), lr=lr)
	classification_criterion = torch.nn.CrossEntropyLoss()
	num_len = labels.shape[0]
	st = time.time()
	for ep in range(num_epochs):
		ct = 0
		total_acc = 0
		for i in range(num_len // batch_size):
			indices = np.random.choice(num_len, batch_size, replace=False)
			batch_images = images[indices]
			batch_labels = labels[indices]
			
			if torch.cuda.is_available():
				batch_images = batch_images.cuda()
				batch_labels = batch_labels.cuda()
			
			optimizer.zero_grad()
			
			logits = model(batch_images)
			loss = classification_criterion(logits, batch_labels)
			
			loss.backward()
			optimizer.step()
			
			if verbose:
				total_acc += calc_accuracy(logits, batch_labels) * batch_labels.shape[0]
			
			ct += batch_size
		
		if verbose:
			print(f'Train Epoch {ep} at time {(time.time() - st) / 60.0} mins - average accuracy = {total_acc / num_len}')


def test(images, labels, model, batch_size=64):
	"""
	test the model
	:param images: 4-d tensor, input images
	:param labels: 1-d tensor, input labels
	:param model: model
	:param batch_size: int, size of batch
	:return: float, performance of the model on the test set
	"""
	num_len = labels.shape[0]
	ct = 0
	total_acc = 0
	while ct < num_len:
		batch_images = images[ct:min(ct + batch_size, num_len)]
		batch_labels = labels[ct:min(ct + batch_size, num_len)]
		
		if torch.cuda.is_available():
			batch_images = batch_images.cuda()
			batch_labels = batch_labels.cuda()
		
		logits = model(batch_images)
		total_acc += calc_accuracy(logits, batch_labels) * batch_labels.shape[0]
		
		ct += batch_size
	return total_acc / num_len


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
	train_images, train_images_names = read_images(train_images_dir)
	test_images, test_images_names = read_images(test_images_dir)
	test_labels = request_labels(test_images_names, test_df)
	# get pre-trained model
	resnet = models.resnet18(pretrained=True)
	resnet.fc = torch.nn.Linear(512, 10, bias=True)
	
	# First step----------------------------------------------------------------
	np.random.seed(0)
	indices = np.random.choice(len(train_images_names), 1000, replace=False)
	labeled_images = train_images[indices]
	images_names = train_images_names[indices]
	labels = request_labels(images_names, train_df)
	
	train(labeled_images, labels, resnet, 100, batch_size=64, verbose=True)
	
	print(f'First checkpoint accuracy: {test(test_images, test_labels, resnet)}')
	# --------------------------------------------------------------------------
	
	# TODO: Second step
	rest_train_images = train_images[np.logical_not(np.isin(train_images_names, images_names))]
	
	# TODO: Last step
	

if __name__ == '__main__':
	main()