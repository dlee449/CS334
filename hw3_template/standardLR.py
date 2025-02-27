import argparse
import numpy as np
import pandas as pd
import time

from lr import LinearRegression, file_to_numpy


class StandardLR(LinearRegression):

    def train_predict(self, xTrain, yTrain, xTest, yTest):
        """
        See definition in LinearRegression class
        """
        trainStats = {}
        start = time.time()
        # Adding the intercept column of 1 to each row
        xTrain = np.c_[np.ones(xTrain.shape[0]), xTrain]
        xTest = np.c_[np.ones(xTest.shape[0]), xTest]

        # beta = (x^T x)^-1 x^T y
        xTrain_transpose = xTrain.T
        xTx = np.dot(xTrain_transpose, xTrain)
        xTx_inv = np.linalg.inv(xTx)
        xTy = np.dot(xTrain_transpose, yTrain)
        self.beta = np.dot(xTx_inv, xTy)

        # Calculate MSE for training data and test data
        train_mse = self.mse(xTrain, yTrain)
        test_mse = self.mse(xTest, yTest)
        end = time.time()
        trainStats[0] = {
            "time": end - start,
            "train-mse": train_mse,
            "test-mse": test_mse
            
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

    args = parser.parse_args()
    # load the train and test data
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    model = StandardLR()
    trainStats = model.train_predict(xTrain, yTrain, xTest, yTest)
    print(trainStats)


if __name__ == "__main__":
    main()
