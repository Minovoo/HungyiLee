import os, sys
import numpy as np 
from random import shuffle
import argparse
from math import log, floor
import pandas as pd 
from sklearn import preprocessing

train_data_path = 'data/train.csv'
train_label_path = 'data/train.csv'
test_data_path = 'data/test.csv'
output_dir = 'data'
save_dir = 'data'



# IO file
def load_data(train_data_path, train_label_path, test_data_path):
# def load_data():
	X_train = pd.read_csv(train_data_path, sep = ',', header = 0)
	Y_train = pd.read_csv(train_label_path, sep = ',', header = 0)
	X_test = pd.read_csv(test_data_path, sep = ',', header = 0)

	# drop the income column in train data
	X_train.drop('income', axis = 1, inplace = True)

	# concatenate X_train and X_test to discrete 
	train = pd.concat([X_train, X_test], ignore_index = True)
	train = _discrete(train)
	X_train = train[:len(X_train)]
	X_test = train[len(X_train):]

	# to be array
	Y_train = np.array(Y_train.values)
	Y_train = Y_train[:, -1]
	X_train = np.array(X_train.values)
	X_test = np.array(X_test.values)

	for i in range(len(Y_train)):
		if ('>' in Y_train[i]):
			Y_train[i] = int(1)
		else:
			Y_train[i] = int(0)
	
	return (X_train, Y_train, X_test)




# preprocessing discrete
def _discrete(data):
	for i in range(len(data.columns)):
		if (isinstance(data[data.columns[i]][0], str)):
			feat = preprocessing.LabelEncoder().fit_transform(data[data.columns[i]])
			data[data.columns[i]] = feat
	return data



# define shuffle
def _shuffle(X, Y):
	randomize = np.arange(len(X))
	np.random.shuffle(randomize)
	return (X[randomize], Y[randomize])


# define normalize
def normalize(X_all, X_test):
	# Feature normalization with train and test X
	# normal = (data - mean of each feature) / (standard deviation)
	X_train_test = np.concatenate((X_all, X_test))
	mu = (sum(X_train_test) / X_train_test.shape[0])
	sigma = np.std(X_train_test, axis = 0)
	mu = np.tile(mu, (X_train_test.shape[0], 1))
	sigma = np.tile(sigma, (X_train_test.shape[0], 1))
	X_train_test_normed = (X_train_test - mu) / sigma

	# Split to train, test again
	X_all = X_train_test_normed[0:X_all.shape[0]]
	X_test = X_train_test_normed[X_all.shape[0]:]
	return X_all, X_test


# define split valid
def split_valid_set(X_all, Y_all, percentage):
	all_data_size = len(X_all)
	valid_data_size = int(floor(all_data_size * percentage))

	X_all, Y_all = _shuffle(X_all, Y_all)

	X_train, Y_train = X_all[0:valid_data_size], Y_all[0:valid_data_size]
	X_valid, Y_valid = X_all[valid_data_size:], Y_all[valid_data_size:]

	return X_train, Y_train, X_valid, Y_valid


# define sigmoid 
def sigmoid(z):
	# must have this step.
	z = z.astype(float)
	res = 1 / (1.0 + np.exp(-z))
	return np.clip(res, 1e-8, 1 - (1e-8))


# get valid score
def valid(w, b, X_valid, Y_valid):
	valid_data_size = len(X_valid)
	z = (np.dot(X_valid, np.transpose(w)) + b)
	y = sigmoid(z)
	y_ = np.around(y)
	result = (np.squeeze(Y_valid) == y_)
	print('Validation acc = %f' % (float(result.sum()) / valid_data_size))

	return


# train model
def train(X_all, Y_all, save_dir):
	# Split a 10%-validation set from the training set
	valid_set_percentage = 0.1
	X_train, Y_train, X_valid, Y_valid = split_valid_set(X_all, Y_all, valid_set_percentage)

	# Initiallize parameter, hyperparameter
	w = np.zeros((14, ))
	b = np.zeros((1, ))
	l_rate = 0.1
	batch_size = 32
	train_data_size = len(X_train)
	step_num = int(floor(train_data_size / batch_size))
	epoch_num = 1000
	save_param_iter = 50

	# Start training
	total_loss = 0.0
	for epoch in range(1, epoch_num):
		# Do validation and parameter saving
		if (epoch) % save_param_iter == 0:
			print('=======Saving Param at epoch %d======='% epoch)
			if not os.path.exists(save_dir):
				os.mkdir(save_dir)
			np.savetxt(os.path.join(save_dir, 'w'), w)
			np.savetxt(os.path.join(save_dir, 'b'), [b, ])
			print ('epoch avg loss = %f' % (total_loss / (float(save_param_iter) * train_data_size)))
			total_loss = 0.0
			valid(w, b, X_valid, Y_valid)

			# Random shuffle
			X_train, Y_train = _shuffle(X_train, Y_train)

			# Train with batch
			for idx in range(step_num):
				X = X_train[idx * batch_size: (idx + 1) * batch_size]
				Y = Y_train[idx * batch_size: (idx + 1) * batch_size]

				z = np.dot(X, np.transpose(w)) + b
				y = sigmoid(z)

				cross_entropy = -1 * (np.dot(np.squeeze(Y), np.log(y)) + np.dot((1 - np.squeeze(Y)), np.log(1 - y)))
				total_loss += cross_entropy

				w_grad = np.mean(-1 * X * (np.squeeze(Y) - y).reshape((batch_size, 1)), axis = 0)
				b_grad = np.mean(-1 * (np.squeeze(Y) - y))

				# SGD updating parameter
				w = w - l_rate * w_grad
				b = b - l_rate * b_grad
	return


# infer & output ans.csv
def infer(X_test, save_dir, output_dir):
	test_data_size = len(X_test)

	#Load parameters
	print('=====Loading Param from %s=====' % save_dir)
	w = np.loadtxt(os.path.join(save_dir, 'w'))
	b = np.loadtxt(os.path.join(save_dir, 'b'))

	# predict
	z = (np.dot(X_test, np.transpose(w)) + b)
	y = sigmoid(z)
	y_ = np.around(y)

	print('=====Writing output to %s=====' % output_dir)
	if not os.path.exists(output_dir):
		os.mkdir(output_dir)
	output_path = os.path.join(output_dir, 'log_prediction.csv')
	with open(output_path, 'w') as f:
		f.write('id, label\n')
		for i, v in enumerate(y_):
			f.write('%d, %d\n' % (i + 1, v))


	return 


# main block
def main():
	# Load feature and label
	X_all, Y_all, X_test = load_data(train_data_path, train_label_path, test_data_path)
	# X_all, Y_all, X_test = load_data()
	# Normalization
	X_all, X_test = normalize(X_all, X_test)

	# To train or to infer
	train(X_all, Y_all, save_dir)
	infer(X_test, save_dir, output_dir)
	print("Error: Argument --train or --infer not found")
	return 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Logistic Regression with Gradient Descent Method')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--train', action='store_true', default=False, dest='train', help='Input --train to Train')
    group.add_argument('--infer', action='store_true', default=False, dest='infer', help='Input --infer to Infer')
    parser.add_argument('--train_data_path', type=str, default='feature/X_train', dest='train_data_path', help='Path to training data')
    parser.add_argument('--train_label_path', type=str, default='feature/Y_train', dest='train_label_path', help='Path to training data\'s label')
    parser.add_argument('--test_data_path', type=str, default='feature/X_test', dest='test_data_path', help='Path to testing data')
    parser.add_argument('--save_dir', type=str, default='logistic_params/', dest='save_dir', help='Path to save the model parameters')
    parser.add_argument('--output_dir', type=str, default='logistic_output/', dest='output_dir', help='Path to save the model parameters')
    opts = parser.parse_args()
    main(opts)






























