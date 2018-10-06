import numpy as np
import matplotlib.pyplot as plt


# y_data = b + w1 * x_data + x2 * xdata^2
x_data = [338., 333., 328., 207., 226., 25., 179., 60., 208.,  606.]
y_data = [640., 633., 619., 393., 428., 27., 193., 66., 226., 1591.]


x = np.arange(-200, -100, 1) # bias
y = np.arange(-5, 5, 0.1) # weight 1
z = np.arange(-5, 5, 0.1) # weight 2
Q = np.zeros((len(x), len(y), len(z)))
X, Y, Z = np.meshgrid(x, y, z)
for i in range(len(x)):
	for j in range(len(y)):
		for k in range(len(z)):
			b = x[i]
			w1 = y[j]
			w2 = z[k]
			Q[k][j][i] = 0
			for n in range((len(x_data))):
				Q[k][j][i] = Q[k][j][i] + (y_data[n] - b - w1 * x_data[n] - w2 * x_data[n] ** 2)
			Q[k][j][i] = Q[k][j][i] / len(x_data)


# initial b, w1, w2
b = 100
w1 = 1
w2 = 1
lr = 1
iteration = 100000


b_his = [b]
w1_his = [w1]
w2_his = [w2]

lr_b = 0
lr_w1 = 0
lr_w2 = 0

for i in range(iteration):
	b_grad = 0.
	w1_grad = 0.
	w2_grad = 0.

	for n in range(len(x_data)):
		b_grad = b_grad - 2.0 * (y_data[n] - b - w1 * x_data[n] - w2 * x_data[n] ** 2)
		w1_grad = w1_grad - 2.0 * (y_data[n] - b - w1 * x_data[n] - w2 * x_data[n] ** 2) * x_data[n]
		w2_grad = w2_grad - 2.0 * (y_data[n] - b - w1 * x_data[n] - w2 * x_data[n] ** 2) * x_data[n] ** 2

	lr_b = lr_b + b_grad ** 2
	lr_w1 = lr_w1 + w1_grad ** 2
	lr_w2 = lr_w2 + w2_grad ** 2

	b = b - lr / np.sqrt(lr_b) * b_grad
	w1 = w1 - lr / np.sqrt(lr_w1) * w1_grad
	w2 = w2 - lr / np.sqrt(lr_w2) * w2_grad

	b_his.append(b)
	w1_his.append(w1)
	w2_his.append(w2)


print "b: %f, w1: %f, w2: %f" % (b, w1, w2)
loss = 0.0
for i in range(len(x_data)):
	loss = loss + (y_data[n] - b - w1 * x_data[n] - w2 * x_data[n] ** 2) ** 2
print "Loss: %f" % loss ** 0.5
# loss = 53.715627