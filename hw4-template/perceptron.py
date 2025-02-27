import argparse
import numpy as np
import pandas as pd
import time

class Perceptron(object):
    mEpoch = 1000  # maximum epoch size
    w = None       # weights of the perceptron

    def __init__(self, epoch):
        self.mEpoch = epoch

    def train(self, xFeat, y):

        stats = {}
        # TODO implement this

        # initialize the weights
        self.w = np.zeros(xFeat.shape[1]+1)

        # for each epoch
        for i in range(self.mEpoch):
            # initialize the number of mistakes as 0
            mistakes = 0
            # for each email
            for i in range(xFeat.shape[0]):
                text_vector = xFeat[i]
                label = y[i][0]
                # calculate w * xi
                prediction = self.w[0] + np.dot(text_vector, self.w[1:])
                # if w*xi is >= 0
                if prediction >= 0:
                    # predict as +1 class
                    y_pred = 1
                    # if mistake on negative
                    if label != y_pred:
                        # update weight
                        self.w[1:] -= text_vector
                        self.w[0] -= 1
                        # add 1 to number of mistakes
                        mistakes += 1
                # if w*xi is < 0
                else:
                    # predict as 0 class
                    y_pred = 0
                    # if mistake on positive
                    if label != y_pred:
                        # update weight
                        self.w[1:] += text_vector
                        self.w[0] += 1
                        # add 1 to number of mistakes
                        mistakes += 1

            # if there were no mistakes
            if mistakes == 0:
                # update stats as 0 mistakes and break
                stats[i] = {'mistakes': 0}
                return stats
            # if there were mistakes
            else:
                # update stats by the number of mistakes and continue
                stats[i] = {'mistakes': mistakes}

        return stats

    def predict(self, xFeat):
        """
        Given the feature set xFeat, predict 
        what class the values will have.

        Parameters
        ----------
        xFeat : nd-array with shape m x d
            The data to predict.  

        Returns
        -------
        yHat : 1d array or list with shape m
            Predicted response per sample
        """
        yHat = []

        # calculate w * x
        predictions = self.w[0] + np.dot(xFeat, self.w[1:])
        
        # for each prediction
        for i in range(len(predictions)):
            # if the prediction is greater than or equal to 0
            if predictions[i] >= 0:
                # predict label as +1
                yHat.append(1)
            # if the prediction is less than 0
            else:
                # predict label as 0
                yHat.append(0)

        return yHat


def calc_mistakes(yHat, yTrue):
    """
    Calculate the number of mistakes
    that the algorithm makes based on the prediction.

    Parameters
    ----------
    yHat : 1-d array or list with shape n
        The predicted label.
    yTrue : 1-d array or list with shape n
        The true label.      

    Returns
    -------
    err : int
        The number of mistakes that are made
    """

    # initialize the number of mistakes
    mistakes = 0

    # for each label
    for i in range(len(yHat)):
        # if the predicted label is not equal to the true label
        if yHat[i] != yTrue[i][0]:
            # add 1 to mistakes
            mistakes += 1
    
    return mistakes


def file_to_numpy(filename):
    """
    Read an input file and convert it to numpy
    """
    df = pd.read_csv(filename)
    return df.to_numpy()


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
    parser.add_argument("epoch", type=int, help="max number of epochs")
    parser.add_argument("--seed", default=334, 
                        type=int, help="default seed number")
    
    args = parser.parse_args()
    # load the train and test data assumes you'll use numpy
    xTrain = file_to_numpy(args.xTrain)
    yTrain = file_to_numpy(args.yTrain)
    xTest = file_to_numpy(args.xTest)
    yTest = file_to_numpy(args.yTest)

    np.random.seed(args.seed)   
    model = Perceptron(args.epoch)
    trainStats = model.train(xTrain, yTrain)
    print(trainStats)
    yHat = model.predict(xTest)
    # print out the number of mistakes
    print("Number of mistakes on the test dataset")
    print(calc_mistakes(yHat, yTest))


if __name__ == "__main__":
    main()