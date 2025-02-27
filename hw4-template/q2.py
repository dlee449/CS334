from perceptron import Perceptron, calc_mistakes, file_to_numpy
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from q1 import model_assessment, build_vocab_map

# Read Binary datasets
binary_train = file_to_numpy("binary_train.csv")
binary_test = file_to_numpy("binary_test.csv")

# Read Count datasets
count_train = file_to_numpy("count_train.csv")
count_test = file_to_numpy("count_test.csv")

# Read y datasets
yTrain = file_to_numpy("yTrain.csv")
yTest = file_to_numpy("yTest.csv")

# set the epoch range to test for optimal performance
epoch_range = [1, 50, 100, 150, 200]

# function for getting the optimal epoch value
def optimal_epoch(xTrain, yTrain, folds, epoch_range):
    # get k folds
    kfold = KFold(n_splits = folds, shuffle = True, random_state = 1)
    # Initialize the mistakes dictionary
    mistakes = {}
    # for each epoch value
    for epoch in epoch_range:
        # Initialize the total number of mistakes
        total_mistakes = 0
        # for each fold
        for trainIndex, testIndex in kfold.split(xTrain):
            # Get the training and testing data of the fold
            xTrain_k, xTest_k = xTrain[trainIndex], xTrain[testIndex]
            yTrain_k, yTest_k = yTrain[trainIndex], yTrain[testIndex]
            # Create a perceptron model of the current epoch value
            model = Perceptron(epoch)
            # Train the model using the current fold's training data
            train_stats = model.train(xTrain_k, yTrain_k)
            # Get the predicted values on the testing data
            yHat = model.predict(xTest_k)
            # Get the total mistakes
            total_mistakes += calc_mistakes(yHat, yTest_k)
        # Calculate the average mistakes of the folds on the current epoch value
        average_mistakes = total_mistakes / folds
        # key of epoch value with average mistakes as value
        mistakes[epoch] = average_mistakes
    
    return mistakes

# Get the average mistakes of the epoch values with 5 folds
binary_epoch = optimal_epoch(binary_train, yTrain, 5, epoch_range)
count_epoch = optimal_epoch(count_train, yTrain, 5, epoch_range)

# Create a dataframe with the binary and count data set with the average mistakes on the epoch values
epoch_df = pd.DataFrame({'binary': binary_epoch, 'count': count_epoch}).T

print("------------------------------------------------------------")

print("Average number of mistakes based on size of epoch")
print(epoch_df)

print("------------------------------------------------------------")


# Get the optimal epoch value
best_epoch_binary = 0
min_mistakes_binary = float('inf')
for key, val in binary_epoch.items():
    if val < min_mistakes_binary:
        best_epoch_binary = key
        min_mistakes_binary = val
best_epoch_count = 0
min_mistakes_count = float('inf')
for key, val in count_epoch.items():
    if val < min_mistakes_count:
        best_epoch_count = key
        min_mistakes_count = val

print()
print("------------------------------------------------------------")

print("Training Binary Dataset")
# Using the optimal epoch value
print("Using epoch: ", best_epoch_binary)
# Create a perceptron model for binary dataset
binary_model = Perceptron(best_epoch_binary)
# train model
binary_stats = binary_model.train(binary_train, yTrain)
# predict on training data using model
binary_train_yHat = binary_model.predict(binary_train)
# calculate the number of mistakes
binary_train_mistakes = calc_mistakes(binary_train_yHat, yTrain)
# predict on test data using model
binary_test_yHat = binary_model.predict(binary_test)
# calculate the number of mistakes
binary_test_mistakes = calc_mistakes(binary_test_yHat, yTest)
print("Number of mistakes on training set: ", binary_train_mistakes)
print("Number of mistakes on test set: ", binary_test_mistakes)

print("------------------------------------------------------------")

print()
print("------------------------------------------------------------")

print("Training Count Dataset")
# Using the optimal epoch value
print("Using epoch: ", best_epoch_count)
# Create a perceptron model for count dataset
count_model = Perceptron(best_epoch_count)
# train model
count_stats = count_model.train(count_train, yTrain)
# predict on training data using model
count_train_yHat = count_model.predict(count_train)
# calculate the number of mistakes
count_train_mistakes = calc_mistakes(count_train_yHat, yTrain)
# predict on test data using model
count_test_yHat = count_model.predict(count_test)
# calculate the number of mistakes
count_test_mistakes = calc_mistakes(count_test_yHat, yTest)
print("Number of mistakes on training set: ", count_train_mistakes)
print("Number of mistakes on test set: ", count_test_mistakes)

print("------------------------------------------------------------")


# get the columns of the dataframe
binary_train_df = pd.read_csv("binary_train.csv")
words_list = binary_train_df.columns

def pos_neg_words(model, words_list):
    pos_indices = np.argsort(model.w)[::-1]
    words_pos = []
    for i in range(15):
        index = pos_indices[i]
        words_pos.append(words_list[index])
    
    neg_indices = np.argsort(model.w)
    words_neg = []
    for i in range(15):
        index = neg_indices[i]
        words_neg.append(words_list[index])

    return words_pos, words_neg

print()
print("------------------------------------------------------------")

print("Binary Dataset")
binary_pos, binary_neg = pos_neg_words(binary_model, words_list)
print("15 most positive words: ")
print(binary_pos)
print("15 most negative words: ")
print(binary_neg)

print("------------------------------------------------------------")


print()
print("------------------------------------------------------------")

print("Count Dataset")
binary_pos, binary_neg = pos_neg_words(count_model, words_list)
print("15 most positive words: ")
print(binary_pos)
print("15 most negative words: ")
print(binary_neg)

print("------------------------------------------------------------")


