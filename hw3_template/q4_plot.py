import sgdLR
from lr import LinearRegression, file_to_numpy
import standardLR
import sgdLR
import matplotlib.pyplot as plt

xTrain = file_to_numpy('new_xTrain.csv')
yTrain = file_to_numpy('eng_yTrain.csv')
xTest =  file_to_numpy('new_xTest.csv')
yTest =  file_to_numpy('eng_yTest.csv')

best_lr = [0.0001, 0.001, 0.01, 0.1, 0.1, 0.1]
batchsizes = [1, 43, 258, 1290, 5590, 16770]

# closed form solution
model_standard = standardLR.StandardLR()
trainStats_standard = model_standard.train_predict(xTrain, yTrain, xTest, yTest)
time_standard = trainStats_standard[0]['time']
mse_train_standard = trainStats_standard[0]['train-mse']
mse_test_standard = trainStats_standard[0]['test-mse']

fig = plt.figure(figsize=(15, 8))
axes_train = fig.add_subplot(1,2,1)
axes_train.scatter(time_standard, mse_train_standard, label='Closed Form Solution', s = 3)

axes_test = fig.add_subplot(1,2,2)
axes_test.scatter(time_standard, mse_test_standard, label='Closed Form Solution', s = 3)

# sgd solution
for i, batch_val in enumerate(batchsizes):
    model = sgdLR.SgdLR(best_lr[i], batch_val, 10)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    time = []
    mse_train = []
    mse_test = []
    for j in trainStats:
        time.append(trainStats[j]['time'])
        mse_train.append(trainStats[j]['train-mse'])
        mse_test.append(trainStats[j]['test-mse'])
    axes_train.scatter(time, mse_train, label=f'Batch Size = {batch_val}', s = 3)
    axes_test.scatter(time, mse_test, label=f'Batch Size = {batch_val}', s = 3)
    print(f'Batch size: {batch_val} done')


axes_test.legend(loc='upper right', fontsize='small')
axes_train.set_xlabel('Time(sec)')
axes_train.set_ylabel('mean squared error')
axes_test.set_xlabel('Time(sec)')
axes_train.set_title('Different Batch Sizes on Training set')
axes_test.set_title('Different Batch Sizes on Test set')
axes_train.set_xlim([0, 3])
axes_test.set_xlim([0, 3])
fig.savefig('q4.png', dpi = 500)

