import sgdLR
from lr import LinearRegression, file_to_numpy
import matplotlib.pyplot as plt



xTrain = file_to_numpy('new_xTrain.csv')
yTrain = file_to_numpy('eng_yTrain.csv')
xTest = file_to_numpy('new_xTest.csv')
yTest = file_to_numpy('eng_yTest.csv')

model = sgdLR.SgdLR(0.0001, 1, 30)
trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)

train_dat = []
test_dat = []

epoch_range = range(1, 30)

for i in epoch_range:
    print(i)
    train_mse = trainStats[len(xTrain)*i-1]['train-mse']
    train_dat.append(train_mse)

    test_mse = trainStats[len(xTrain)*i-1]['test-mse']
    test_dat.append(test_mse)
    

fig = plt.figure()
plt.plot(epoch_range, train_dat, label='Training MSE')
plt.plot(epoch_range, test_dat, label='Test MSE')

plt.legend(loc='upper right')
plt.xlabel('epoch')
plt.ylabel('mean squared error')
plt.title('Training and Test with Learning Rate 0.0001')

fig.savefig('3c.png', dpi = 500)