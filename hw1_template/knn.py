import argparse
import numpy as np
import pandas as pd


class Knn(object):
    k = 0    # number of neighbors to use
    X_train = None
    y_train = None

    def __init__(self, k):
        """
        Knn constructor

        Parameters
        ----------
        k : int 
            Number of neighbors to use.
        """
        self.k = k


    def train(self, xFeat, y):
        """
        Train the k-nn model.

        Parameters
        ----------
        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of labels associated with training data.

        Returns
        -------
        self : object
        """
        # TODO do whatever you need

        # change to numpy array if xFeat is a pandas dataframe
        if isinstance(xFeat, pd.DataFrame):
            xFeat = xFeat.to_numpy()
        self.X_train = xFeat
        self.y_train = y
        return self


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
            Predicted class label per sample
        """
        yHat = [] # variable to store the estimated class label

        # change to numpy array if xFeat is a pandas dataframe
        if isinstance(xFeat, pd.DataFrame):
            xFeat = xFeat.to_numpy()
        
        # for each row
        for row in range(xFeat.shape[0]):
            # euclidean distance
            sum_square = np.sum((self.X_train - xFeat[row, :])**2, axis=1)
            dist = np.sqrt(sum_square)
            # argsort to save index of distances in order to find closest neighbors
            index_dist = np.argsort(dist)
            # get the labels for k nearest neighbors
            k_neighbors = self.y_train.iloc[index_dist[0:self.k]]
            # append the most voted label as the predicted label
            yHat.append(k_neighbors.mode()[0])
        return yHat



def accuracy(yHat, yTrue):
    """
    Calculate the accuracy of the prediction

    Parameters
    ----------
    yHat : 1d-array with shape n
        Predicted class label for n samples
    yTrue : 1d-array with shape n
        True labels associated with the n samples

    Returns
    -------
    acc : float between [0,1]
        The accuracy of the model
    """
    # TODO calculate the accuracy
    acc = 0
    count = 0
    # for all the predicted label
    for i in range(len(yHat)):
        # if the predicted label is correct, count += 1
        if yHat[i] == yTrue[i]:
            count += 1
    # the accuracy percentage 
    acc = float(count/len(yHat)) * 100
    return acc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("k",
                        type=int,
                        help="the number of neighbors")
    parser.add_argument("--xTrain",
                        default="q3xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q3yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q3xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q3yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the model
    knn = Knn(args.k)
    knn.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = knn.predict(xTrain)
    trainAcc = accuracy(yHatTrain, yTrain['label'])
    # predict the test dataset
    yHatTest = knn.predict(xTest)
    testAcc = accuracy(yHatTest, yTest['label'])
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)
    

if __name__ == "__main__":
    main()
