import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sgdLR
from lr import LinearRegression, file_to_numpy

xTrain = file_to_numpy('new_xTrain.csv')
yTrain = file_to_numpy('eng_yTrain.csv')
xTest = file_to_numpy('new_xTest.csv')
yTest = file_to_numpy('eng_yTest.csv')

# training size = 16770
batch_size = [43, 258, 1290, 5590, 16770]
lr = [0.1, 0.01, 0.001, 0.0001, 0.00001]
epoch_range = range(1, 30)
for i, batch_val in enumerate(batch_size):
    result = [[] for i in range(len(lr))]
    for j, lr_val in enumerate(lr):
        model = sgdLR.SgdLR(lr_val, batch_val, max(epoch_range))
        trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
        for epoch in epoch_range:
            result[j].append(trainStats[len(xTrain) / batch_val * epoch - 1]['train-mse'])
    
    fig = plt.figure()

    plt.plot(epoch_range, result[0], label='lr=0.1')
    plt.plot(epoch_range, result[1], label='lr=0.01')
    plt.plot(epoch_range, result[2], label='lr=0.001')
    plt.plot(epoch_range, result[3], label='lr=0.0001')
    plt.plot(epoch_range, result[4], label='lr=0.00001')

    plt.legend(loc='upper right')
    plt.title(f'Batch size = {batch_val}')
    plt.xlabel('epoch')
    plt.ylabel('mean squared error')
    fig.savefig(f'q4_size_{batch_val}.png', dpi = 500)
    print(f'Batch size: {batch_val} done')
