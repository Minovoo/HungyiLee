import numpy as np
import csv



# fileObject = open('result/hw1_test_w.txt', 'w')  
file_name = 'result/hw1_test_w.txt'
w = []
for line in open(file_name):
    line = line.split()
    w.append(float(line[0]))


def getW():
	return w

w = np.array(w)

test_x = []
n_row = 0
text = open('data/test.csv', 'r')
row = csv.reader(text, delimiter = ',')
lab = 0
num = 0

for r in row:
	if n_row % 18 == 0:
		lab = 0
	if lab == 9:
		test_x.append([])
		for i in range(2, 11):
			test_x[num].append(float(r[i]))
		num = num + 1
	n_row = n_row + 1
	lab = lab + 1

test_x = np.array(test_x)


def getTest():
	return test_x

test_x = np.concatenate((np.ones((test_x.shape[0], 1)), test_x), axis = 1)

# get answer
ans = []
for i in range(len(test_x)):
	ans.append(['id' + str(i)])
	value = np.sum(w * test_x[i])
	ans[i].append(value)


filename = "result/predict_test.csv"
text = open(filename, 'w+')
s = csv.writer(text, delimiter = ',', lineterminator = '\n')
s.writerow(['id', 'value'])
for i in range(len(ans)):
	s.writerow(ans[i])
text.close()