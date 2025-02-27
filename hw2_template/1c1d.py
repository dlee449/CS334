import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dt import dt_train_test, DecisionTree


#1(c) creating plot
# load the train and test data
xTrain = pd.read_csv("q4xTrain.csv")
yTrain = pd.read_csv("q4yTrain.csv")
xTest = pd.read_csv("q4xTest.csv")
yTest = pd.read_csv("q4yTest.csv")

#default value for maxDepth and minLeafSample
default_val = 5

#x-axis of plot
maxDepth = list(np.arange(1, 9))
minLeafSample = list(np.arange(1, 9))

#y-axis of plot
depth_gini = np.zeros(shape = (8, 2))
leaf_gini = np.zeros(shape = (8, 2))
depth_entropy = np.zeros(shape = (8, 2))
leaf_entropy = np.zeros(shape = (8, 2))

#iterate decision tree to get training accuracy and test accuracy

#Using gini, default value for minLeafSample
for i in range(len(maxDepth)):
	dt = DecisionTree('gini', i + 1, default_val)
	train_acc, test_acc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
	depth_gini[i, 0] = train_acc
	depth_gini[i, 1] = test_acc

#Using gini, default value for maxDepth
for j in range(len(minLeafSample)):
	dt = DecisionTree('gini', default_val, j + 1)
	train_acc, test_acc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
	leaf_gini[j, 0] = train_acc
	leaf_gini[j, 1] = test_acc

#Using entropy, default value for minLeafSample
for i in range(len(maxDepth)):
	dt = DecisionTree('entropy', i + 1, default_val)
	train_acc, test_acc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
	depth_entropy[i, 0] = train_acc
	depth_entropy[i, 1] = test_acc

#Using entropy, default value for maxDepth
for j in range(len(minLeafSample)):
	dt = DecisionTree('entropy', default_val, j + 1)
	train_acc, test_acc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
	leaf_entropy[j, 0] = train_acc
	leaf_entropy[j, 1] = test_acc


#subplot for gini maximum depth
fig, axs = plt.subplots(2, 2, figsize=(15, 10))
axs[0, 0].plot(maxDepth, depth_gini[: ,0], label='Train')
axs[0, 0].plot(maxDepth, depth_gini[: ,1], label='Test')
axs[0, 0].set_title('Gini Maximum Depth')

#subplot for gini minimum leaf samples
axs[0, 1].plot(minLeafSample, leaf_gini[: ,0], label='Train')
axs[0, 1].plot(minLeafSample, leaf_gini[: ,1], label='Test')
axs[0, 1].set_title('Gini Minimum Leaf Samples')

#subplot for entropy maximum depth
axs[1, 0].plot(maxDepth, depth_entropy[: ,0], label='Train')
axs[1, 0].plot(maxDepth, depth_entropy[: ,1], label='Test')
axs[1, 0].set_title('Entropy Maximum Depth')

#subplot for entropy minimum leaf samples
axs[1, 1].plot(minLeafSample, leaf_entropy[: ,0], label='Train')
axs[1, 1].plot(minLeafSample, leaf_entropy[: ,1], label='Test')
axs[1, 1].set_title('Entropy Minimum Leaf Samples')

legends = ('Train', 'Test')
fig.legend(legends, loc='upper right')
fig.suptitle('2D plots of accuracy')
plt.savefig('1(c).png')


#1(d)

"""

The computational complexity of the train function is O(pdnlogn).
For each feature, all the possible splits are calculated -> dn
and for each of this process, the rows are sorted. -> dnlogn.
If you add this this becomes O(dnlogn).
And this process is done for each node in the tree which makes O(p(dnlogn)) which is just 
O(pdnlogn).

The computational complexity of the predict function is O(p).
THe predict function is only traversing the tree which is just the
maximum depth.

"""