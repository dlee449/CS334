import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import knn

# 3(d)

xTest = pd.read_csv("q3xTest.csv")
xTrain = pd.read_csv("q3xTrain.csv")
yTest = pd.read_csv("q3yTest.csv")
yTrain = pd.read_csv("q3yTrain.csv")
# Create lists of accarcy percentage for both Train and Test sets
accuracy_Train = []
accuracy_Test = []

# k ranges from 1 to 25
for i in range(1,25):
    model = knn.Knn(i)
    model.train(xTrain, yTrain['label'])
    # predict the test dataset
    yHatTrain = model.predict(xTrain)
    yHatTest = model.predict(xTest)
    accuracy_Train.append(knn.accuracy(yHatTrain, yTest['label']))
    accuracy_Test.append(knn.accuracy(yHatTest, yTest['label']))

# k ranges from 1 to 25
x = np.arange(1, 25)
y1 = accuracy_Train
y2 = accuracy_Test

plt.plot(x, y1, label = "train")
plt.plot(x, y2, label = "test")
plt.legend()
plt.savefig("q3(d)_train&test_accuracy_vs_k")


# 3(e)
"""
The computational complexity of the predict function would be O(nd + nlogn).

As the code is looping through a point to calculate distance for each of the features, it would take O(d) and because
the loop is done for all the samples, the computational complexity would be O(nd).
Also, as the argsort() function uses quicksort to osrt the distances to find k nearest neighbors, 
the computational complexity of O(nlogn) would be added.

"""
