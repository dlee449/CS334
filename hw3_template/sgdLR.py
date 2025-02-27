import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class SgdLR(LinearRegression):
    lr = 1  # learning rate
    bs = 1  # batch size
    mEpoch = 1000 # maximum epoch size

    def __init__(self, lr, bs, epoch):
        self.lr = lr
        self.bs = bs
        self.mEpoch = epoch

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        
        trainStats = {}
        

        # Get the training sample size and the mini-batch size
        N = xTrain.shape[0]
        B = int(N/self.bs)

        # Add intercept column to both training and test data
        xTrain = np.c_[np.ones(xTrain.shape[0]), xTrain]
        xTest = np.c_[np.ones(xTest.shape[0]), xTest]


        # Initialize beta to zeros with intercept added
        self.beta = np.ones((xTrain.shape[1], 1))
        
        start = time.time()

        # For each epoch
        for epoch in range(self.mEpoch):
            # shuffle
            idx = np.random.permutation(N)
            

            # for each mini-batch
            for i in range(B):

                # Get the current mini-batch using shuffled indices
                bIdx = idx[i * self.bs : (i + 1) * self.bs]
                X_batch = xTrain[bIdx]
                y_batch = yTrain[bIdx]

                # negative gradient = x^T(y-x*beta)
                temp1 = X_batch.T
                temp2 = y_batch-np.dot(X_batch, self.beta)
                grad = np.dot(temp1, temp2) / self.bs

                # beta = beta + lr * gradient
                self.beta = self.beta + self.lr * grad

                # get MSE for training data and test data and time taken
                train_mse = self.mse(xTrain, yTrain)
                test_mse = self.mse(xTest, yTest)
                end = time.time()

                # result in dictionary key with current iteration number
                trainStats[epoch * B + i] = {
                    "train-mse": train_mse,
                    "test-mse": test_mse,
                    "time": end - start
                }

        return trainStats


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("xTrain",
                        help="filename for features of the training data")
    parser.add_argument("yTrain",
                        help="filename for labels associated with training data")
    parser.add_argument("xTest",
                        help="filename for features of the test data")
    parser.add_argument("yTest",
                        help="filename for labels associated with the test data")
    parser.add_argument("lr", type=float, help="learning rate")
    parser.add_argument("bs", type=int, help="batch size")
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)
    # setting the seed for deterministic behavior
    np.random.seed(args.seed)   
    model = SgdLR(args.lr, args.bs, args.epoch)
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()

