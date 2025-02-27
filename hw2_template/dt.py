import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from scipy import stats

#Helper function for Attribute selection measure
def attribute_selection_measure(y, criterion):

    #The total number of entries in the given case
    total = len(y)

    #Get the unique values of the labels
    labels = np.unique(y)

    #List of probabily of each unique values
    prob_labels = []
    
    #Calculate the probability of each label ocurring in the case
    for i in range(len(labels)):
        prob_labels.append(float(np.sum(y == labels[i]))/total)

    #if using gini as the measure
    if criterion == "gini":
        #Initialize gini index as 1
        gini = 1
        #Subtract the square of the probability of each labels to get gini index
        for i in range(len(prob_labels)):
            gini -= prob_labels[i]**2

        return gini
    
    #if using entropy as the measure
    elif criterion == "entropy":
        #Initialize entropy as 0
        entropy = 0
        #Subtract the p log p of each labels to get the entropy
        for i in range(len(prob_labels)):
            entropy -= prob_labels[i] * np.log2(prob_labels[i])

        return entropy

#Helper function for finding the best feature and the best value to split in the decision tree
def find_best_split(xFeat, y, criterion, minLeafSample):

    #initialize current best feature to split as the first feature
    best_feature = xFeat.columns[0]

    #initialize current best value to split as the first value
    best_val = xFeat.iloc[0, 0]

    #initialize best gain as 0
    best_gain = 0

    #iterate all the features
    for curr_feature in xFeat.columns:
        #sort the xFeat in order of the value in the current feature
        sorted_index = np.argsort(xFeat[curr_feature])
        sorted_xFeat = xFeat.iloc[sorted_index]
        #sort the y correspondingly
        sorted_y = y[sorted_index]

        #get the original gini/entropy
        measure_df = attribute_selection_measure(y, criterion)

        
        for i in range(minLeafSample, len(xFeat)-minLeafSample):
            curr_val = sorted_xFeat.iloc[i][curr_feature]
            less_than = sorted_xFeat[curr_feature] <= curr_val
            greater_than = sorted_xFeat[curr_feature] > curr_val
            y_less = sorted_y[less_than]
            y_greater = sorted_y[greater_than]
            #if gini, gini impurity of y_less
            #if entropy, entropy of y_less
            measure_less = attribute_selection_measure(y_less, criterion)
            #if gini, gini impurity of y_greater
            #if entropy, entropy of y_greater
            measure_greater = attribute_selection_measure(y_greater, criterion)
            #if gini, impurity of total
            #if entropy, entropy of total
            measure_total = float(len(y_less)/len(y)) * measure_less + float(len(y_greater)/len(y)) * measure_greater
            #if gini, gini gain
            #if entropy, information gain
            information_gain = measure_df - measure_total
            if information_gain > best_gain:
                best_feature = curr_feature
                best_val = curr_val
                best_gain = information_gain
    return best_feature, best_val

#Node class for creating decision tree
class Node():
    def __init__(self):
        #splitting feature
        self.feature = None
        #splitting value 
        self.value = None
        #left node
        self.left = None
        #right node
        self.right = None
        #depth of current node
        self.depth = 0
        #prediction 
        self.predict = None

class DecisionTree(object):
    maxDepth = 0       # maximum depth of the decision tree
    minLeafSample = 0  # minimum number of samples in a leaf
    criterion = None   # splitting criterion

    def __init__(self, criterion, maxDepth, minLeafSample):
        """
        Decision tree constructor

        Parameters
        ----------
        criterion : String
            The function to measure the quality of a split.
            Supported criteria are "gini" for the Gini impurity
            and "entropy" for the information gain.
        maxDepth : int 
            Maximum depth of the decision tree
        minLeafSample : int 
            Minimum number of samples in the decision tree
        """
        self.criterion = criterion
        self.maxDepth = maxDepth
        self.minLeafSample = minLeafSample
        #initialize head node
        self.head = Node()
    
    #stopping criteria
    def is_stopping_criteria(self, left_y, right_y, depth):
        #if the left or right node of the current node has less than the minimum samples in a leaf
        if len(left_y) < self.minLeafSample or len(right_y) < self.minLeafSample:
            return True
        #if the current node has depth that is greater than the maxDepth
        elif depth >= self.maxDepth:
            return True
        #not a stopping criteria
        else:
            return False
        

    def decision_tree(self, xFeat, y, curr_node):
        
    
        #find the best feature and splitting value
        best_feature, best_val = find_best_split(xFeat, y, self.criterion, self.minLeafSample)

        #save the splitted datas of x and y
        left_index = xFeat[best_feature] <= best_val
        right_index = xFeat[best_feature] > best_val
        left_x = xFeat.loc[left_index]
        right_x = xFeat.loc[right_index]
        left_y = y[left_index]
        right_y = y[right_index]

        #check for stopping criteria first
        if self.is_stopping_criteria(left_y, right_y, curr_node.depth):
            #if at stopping criteria, add the mode of y in the node's value
            #prediction
            curr_node.predict = stats.mode(y, keepdims = True)[0]
        
        else:
            #Save feature, value to the current node
            curr_node.feature = best_feature
            curr_node.value = best_val
            #Create the left and right node with depth increased by 1
            curr_node.left = Node()
            curr_node.right = Node()
            curr_node.left.depth = curr_node.depth + 1
            curr_node.right.depth = curr_node.depth + 1
            #Recursion to create whole decision tree
            self.decision_tree(left_x, left_y, curr_node.left)
            self.decision_tree(right_x, right_y, curr_node.right)


    def predict_sample(self, node, x):
        # If there is a child node
        if node.left != None:
            # If the sample's value is greater than the split value
            if x[node.feature] > node.value:
                # Move to the right node and recursively call the function
                newNode = node.right
                return self.predict_sample(newNode, x)
            # If the sample's value is less than or equal to the split value
            else:
                # Move to the left node and recursively call the function
                newNode = node.left
                return self.predict_sample(newNode, x)
        # If the current node is the prediction node, return the predicted value
        else:
            return node.predict



    def train(self, xFeat, y):
        """
        Train the decision tree model.

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
        self.decision_tree(xFeat, y.to_numpy(), self.head)
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
        yHat = [] # feature to store the estimated class label
        # Iterate over all the rows of the provided xFeat
        for i in range(xFeat.shape[0]):
            # Append the predicted label to yHat
            yHat.append(self.predict_sample(self.head, xFeat.iloc[i, :]))
        return yHat


def dt_train_test(dt, xTrain, yTrain, xTest, yTest):
    """
    Given a decision tree model, train the model and predict
    the labels of the test data. Returns the accuracy of
    the resulting model.

    Parameters
    ----------
    dt : DecisionTree
        The decision tree with the model parameters
    xTrain : nd-array with shape n x d
        Training data 
    yTrain : 1d array with shape n
        Array of labels associated with training data.
    xTest : nd-array with shape m x d
        Test data 
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    acc : float
        The accuracy of the trained knn model on the test data
    """
    # train the model
    dt.train(xTrain, yTrain['label'])
    # predict the training dataset
    yHatTrain = dt.predict(xTrain)
    trainAcc = accuracy_score(yTrain['label'], yHatTrain)
    # predict the test dataset
    yHatTest = dt.predict(xTest)
    testAcc = accuracy_score(yTest['label'], yHatTest)
    return trainAcc, testAcc


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("md",
                        type=int,
                        help="maximum depth")
    parser.add_argument("mls",
                        type=int,
                        help="minimum leaf samples")
    parser.add_argument("--xTrain",                        
                        default="q4xTrain.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="q4yTrain.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="q4xTest.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="q4yTest.csv",
                        help="filename for labels associated with the test data")

    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain)
    yTrain = pd.read_csv(args.yTrain)
    xTest = pd.read_csv(args.xTest)
    yTest = pd.read_csv(args.yTest)
    # create an instance of the decision tree using gini
    dt1 = DecisionTree('gini', args.md, args.mls)
    trainAcc1, testAcc1 = dt_train_test(dt1, xTrain, yTrain, xTest, yTest)
    print("GINI Criterion ---------------")
    print("Training Acc:", trainAcc1)
    print("Test Acc:", testAcc1)
    dt = DecisionTree('entropy', args.md, args.mls)
    trainAcc, testAcc = dt_train_test(dt, xTrain, yTrain, xTest, yTest)
    print("Entropy Criterion ---------------")
    print("Training Acc:", trainAcc)
    print("Test Acc:", testAcc)


if __name__ == "__main__":
    main()
