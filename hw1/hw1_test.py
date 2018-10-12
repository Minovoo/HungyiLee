import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import io


# every feature is a dimension
data = []
for i in range(18):
	data.append([])

n_row = 0
text = io.open('data/train.csv', 'r', encoding='utf-8')
row = csv.reader(text, delimiter = ',')
for r in row:
	# the first column has no data
	if n_row != 0:
		for i in range(3, 27):
			if r[i] != 'NR':
				data[(n_row - 1) % 18].append(float(r[i]))
			else:
				data[(n_row - 1) % 18].append(float(0))
	n_row = n_row + 1
text.close()


def getPM():
	return data[9][:20]




x_data = []
y_data = []

for mon in range(12):
	for dat in range(471):
		x_data.append([])
		for single in range(9):
			x_data[471 * mon + dat].append(data[9][480 * mon + dat + single])
		# for i in range(9):
			# x[471 * mon + dat].append(data[9][480 * mon + dat + i])
		y_data.append(data[9][480 * mon + dat + 9]) 
x_data = np.array(x_data)
y_data = np.array(y_data)


def getX_Y():
	return x_data, y_data

x_data = np.concatenate((np.ones((x_data.shape[0], 1)), x_data), axis = 1)

w = np.zeros(len(x_data[0]))
lr = 1
iterator = 2500
lr_seg = np.zeros(len(x_data[0]))

for i in range(iterator):
	w_grad = np.zeros(len(x_data[0]))

	for n in range(len(x_data)):
		w_grad[0] = w_grad[0] - 2.0 * (y_data[n] - w[0] - w[1] * x_data[n][1] - w[2] * x_data[n][2] - w[3] * x_data[n][3] - w[4] * x_data[n][4] - w[5] * x_data[n][5] - w[6] * x_data[n][6] - w[7] * x_data[n][7] - w[8] * x_data[n][8] - w[9] * x_data[n][9]) * 1.0
		for m in range(1, 10):
			w_grad[m] = w_grad[m] - 2.0 * (y_data[n] - w[0] - w[1] * x_data[n][1] - w[2] * x_data[n][2] - w[3] * x_data[n][3] - w[4] * x_data[n][4] - w[5] * x_data[n][5] - w[6] * x_data[n][6] - w[7] * x_data[n][7] - w[8] * x_data[n][8] - w[9] * x_data[n][9]) * x_data[n][m]

	for j in range(len(lr_seg)):
		lr_seg[j] = lr_seg[j] + w_grad[j] ** 2


	# update parameters
	for j in range(len(w)):
		w[j] = w[j] - lr / np.sqrt(lr_seg[j]) * w_grad[j]

	lms = 0.0
	for n in range(len(x_data)):
		lms = lms + (y_data[n] - w[0] - w[1] * x_data[n][1] - w[2] * x_data[n][2] - w[3] * x_data[n][3] - w[4] * x_data[n][4] - w[5] * x_data[n][5] - w[6] * x_data[n][6] - w[7] * x_data[n][7] - w[8] * x_data[n][8] - w[9] * x_data[n][9]) ** 2
	lms = lms / len(x_data)
	lms = math.sqrt(lms)

	print ('iterator: %d | Cost: %f') % (i, (y_data[n] - w[0] - w[1] * x_data[n][1] - w[2] * x_data[n][2] - w[3] * x_data[n][3] - w[4] * x_data[n][4] - w[5] * x_data[n][5] - w[6] * x_data[n][6] - w[7] * x_data[n][7] - w[8] * x_data[n][8] - w[9] * x_data[n][9]))



fileObject = open('result/hw1_test_w2.txt', 'w')  
for ip in w:  
    fileObject.write(str(ip))  
    fileObject.write('\n')  
fileObject.close()  






























