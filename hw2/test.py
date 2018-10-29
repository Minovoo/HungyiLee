import os, sys
import numpy as np 
from random import shuffle
import argparse
from math import log, floor
import pandas as pd 
from sklearn import preprocessing





# preprocessing discrete
def _discrete(data):
	for i in range(len(data.columns)):
		if (isinstance(data[data.columns[i]][0], str)):
			feat = preprocessing.LabelEncoder().fit_transform(data[data.columns[i]])
			data[data.columns[i]] = feat
	return data


# IO file
def load_data(train_data_path, train_label_path, test_data_path):
# def load_data():
	X_train = pd.read_csv(train_data_path, sep = ',', header = 0)
	Y_train = pd.read_csv(train_label_path, sep = ',', header = 0)
	X_test = pd.read_csv(test_data_path, sep = ',', header = 0)

	X_train.drop('income', axis = 1, inplace = True)
	train = pd.concat([X_train, X_test], ignore_index = True)

	train = _discrete(train)

	X_train = train[:len(X_train)]
	X_test = train[len(X_train):]

	Y_train = np.array(Y_train.values)
	Y_train = Y_train[:, -1]

	X_train = np.array(X_train.values)

	for i in range(len(Y_train)):
		if ('>' in Y_train[i]):
			Y_train[i] = int(1)
		else:
			Y_train[i] = int(0)

	

	
	X_test = np.array(X_test.values)



	return (X_train, Y_train, X_test)








(X_train, Y_train, X_test, train) = load_data('data/train.csv','data/train.csv', 'data/test.csv')


print(X_train)
print(Y_train)
print(X_test)
