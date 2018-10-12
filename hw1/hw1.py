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


x = []
y = []
# every 12 months
for mon in range(12):
	# there are 471 data available in every month
	for dat in range(471):
		x.append([])
		# 18 kind of harmful substances
		for kind in range(18):
			# nine hours in succession to predict the tenth hour 
			for single in range(9):
				x[471 * mon + dat].append(data[kind][480 * mon + dat + single])
		# only store the data of PM2.5, this data + 9 is its label
		y.append(data[9][480 * mon + dat + 9])
x = np.array(x)
y = np.array(y)


# add bias
x = np.concatenate((np.ones((x.shape[0], 1)), x), axis = 1)



def getX():
	return x, x.transpose()

# initial weight, learning rate, iterator
w = np.zeros(len(x[0]))
l_rate = 10
iterator = 10000


# train
x_t = x.transpose()
s_gra = np.zeros(len(x[0]))

for i in range(iterator):
	hypo = np.dot(x, w)
	loss = hypo - y
	cost = np.sum(loss ** 2) / len(x)
	cost_a = math.sqrt(cost)
	gra = np.dot(x_t, loss)
	s_gra += gra**2
	ada = np.sqrt(s_gra)
	w = w - l_rate * gra / ada
	print ('iterator: %d | Cost: %f') % (i, cost_a)




# save model
np.save('model', w)


# read model
w = np.load('model.npy')


#read testing data
test_x = []
n_row = 0
text = open('data/test.csv', 'r')
row = csv.reader(text, delimiter = ',')


for r in row:
	if n_row % 18 == 0:
		test_x.append([])
		for i in range(2, 11):
			test_x[n_row // 18].append(float(r[i]))
	else:
		for i in range(2, 11):
			if r[i] != 'NR':
				test_x[n_row // 18].append(float(r[i]))
			else:
				test_x[n_row // 18].append(0)
	n_row = n_row + 1
text.close()
test_x = np.array(test_x)


# add bias 
test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis = 1)



# get answer 
ans = []
for i in range(len(test_x)):
	ans.append(['id' + str(i)])
	a = np.dot(w, test_x[i])
	ans[i].append(a)

filename = "result/predict.csv"
text = open(filename, 'w+')
s = csv.writer(text, delimiter = ',', lineterminator = '\n')
s.writerow(['id', 'value'])
for i in range(len(ans)):
	s.writerow(ans[i])
text.close()


fileObject = open('result/hw1_w.txt', 'w')  
for ip in w:  
    fileObject.write(str(ip))  
    fileObject.write('\n')  
fileObject.close()  





def getW():
	return w


def getX():
	return test_x













