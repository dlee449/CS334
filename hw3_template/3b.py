import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sgdLR
from lr import LinearRegression, file_to_numpy

xTrain = file_to_numpy('new_xTrain.csv')
yTrain = file_to_numpy('eng_yTrain.csv')


subset_index = np.random.randint(len(xTrain), size=int(len(xTrain)*0.4))
xtrain = xTrain[subset_index, :]
ytrain = yTrain[subset_index]
xtest = np.delete(xTrain, subset_index, axis=0)
ytest = np.delete(yTrain, subset_index)

epoch_range = range(1, 30)
batch_size = 1

lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]

result = [[] for i in range(len(lr))]

for i, val in enumerate(lr):
		model = sgdLR.SgdLR(val, batch_size, max(epoch_range))
		trainStats = model.train_predict(xtrain, ytrain, xtest, ytest)
		for epoch in epoch_range:
			result[i].append(trainStats[len(xtrain)*epoch-1]['train-mse'])
		print(val)
fig = plt.figure()

plt.plot(epoch_range, result[0], label='lr=0.1')
plt.plot(epoch_range, result[1], label='lr=0.01')
plt.plot(epoch_range, result[2], label='lr=0.001')
plt.plot(epoch_range, result[3], label='lr=0.0001')
plt.plot(epoch_range, result[4], label='lr=0.00001')

plt.legend(loc='upper right')
plt.title('Mean Squared Error as a Function of the Epoch in Various Learning Rates')
plt.xlabel('epoch')
plt.ylabel('mean squared error')

fig.show()

fig.savefig('3b.png', dpi = 500)
