import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from q4 import *

xTest = pd.read_csv("q4xTest.csv")
xTrain = pd.read_csv("q4xTrain.csv")
yTest = pd.read_csv("q4yTest.csv")
yTrain = pd.read_csv("q4yTrain.csv")


xTrainStd, xTestStd = standard_scale(xTrain, xTest)
xTrainMM, xTestMM = minmax_range(xTrain, xTest)
xTrainIrr, yTrainIrr = add_irr_feature(xTrain, xTest)

# lists of accuracy percentage for each pre-processing
accuracy_no = []
accuracy_standard = []
accuracy_minmax = []
accuracy_irrel = []

for i in range(1,25):
    acc1 = knn_train_test(i, xTrain, yTrain, xTest, yTest)
    acc2 = knn_train_test(i, xTrainStd, yTrain, xTestStd, yTest)
    acc3 = knn_train_test(i, xTrainMM, yTrain, xTestMM, yTest)
    acc4 = knn_train_test(i, xTrainIrr, yTrain, yTrainIrr, yTest)
    accuracy_no.append(acc1)
    accuracy_standard.append(acc2)
    accuracy_minmax.append(acc3)
    accuracy_irrel.append(acc4)

x = np.arange(1, 25)
y1 = accuracy_no
y2 = accuracy_standard
y3 = accuracy_minmax
y4 = accuracy_irrel

plt.plot(x, y1, label = "no-preprocessing")
plt.plot(x, y2, label = "standard scale")
plt.plot(x, y3, label = "min max scale")
plt.plot(x, y4, label = "with irrelevant feature")
plt.legend()
plt.savefig("q4(d)_accuracy_differentpreprocessing")


"""
Using standard scale and minmax scale increase the accuracy in general.
Adding irrelevant features lower the accuracy in general.

"""