import argparse
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from collections import defaultdict

def model_assessment(filename):
    """
    Given the entire data, split it into training and test set 
    so you can assess your different models 
    to compare perceptron, logistic regression,
    and naive bayes. 
    """
    # open the data file
    dat = open(filename)

    # initialize the labels and features lists
    labels = []
    features = []

    # for every line of the data file
    for line in dat:

        # append the first character as label (label is the first character) 
        labels.append(line[0])

        # append the text to features list (the text starts from the third character)
        features.append(line[2:])

    # split the data into training and test
    xTrain, xTest, yTrain, yTest = train_test_split(features, labels, test_size = 0.3, random_state = 1)

    # create dataframes for training and test
    train_dat = {'y': yTrain, 'text': xTrain}
    test_dat = {'y': yTest, 'text': xTest}
    train_df = pd.DataFrame(train_dat)
    test_df = pd.DataFrame(test_dat)

    # return train and test dataframes
    return train_df, test_df


def build_vocab_map(train_df):

    # create a default dict for counting unique vocabs in each email
    vocab_counts = defaultdict(int)

    # for every email
    for i in range(train_df.shape[0]):
        # create a list of vocabs of the email
        vocabs = train_df.text[i].split(" ")
        # make list into a list of unique vocabs
        vocabs = set(vocabs)
        # for each unique vocabs
        for vocab in vocabs:
            # count unique vocabs in each email
            vocab_counts[vocab] += 1

    # create a dictionary for the vocabulary map
    vocab_map = {}

    # for every vocab and its counts
    for word, count in vocab_counts.items():
        # select the words that appear in at least 30 emails
        if count >= 30:
            vocab_map[word] = count

    return vocab_map


def construct_binary(train_df, vocab_map):
    """
    Construct email datasets based on
    the binary representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is 1 if the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # create a list of words for the vocab map
    frequent_words = list(vocab_map.keys())

    # initialize the binary dataset
    binary_train = np.zeros((train_df.shape[0], len(frequent_words)))

    # for each email
    for i in range(train_df.shape[0]):
        # create a list of unique vocabs in an email
        vocabs = train_df.text[i].split(" ")
        vocabs = set(vocabs)

        # for each words in the vocabulary map
        for j in range(len(frequent_words)):
            # if the words in the vocabulary map is in the email
            if frequent_words[j] in vocabs:
                # set vector as 1
                binary_train[i, j] = 1

    return pd.DataFrame(binary_train, columns = frequent_words)


def construct_count(train_df, vocab_map):
    """
    Construct email datasets based on
    the count representation of the email.
    For each e-mail, transform it into a
    feature vector where the ith entry,
    $x_i$, is the number of times the ith word in the 
    vocabulary occurs in the email,
    or 0 otherwise
    """
    # create a list of words for the vocab map
    frequent_words = list(vocab_map.keys())

    # initialize the count dataset
    count_train = np.zeros((train_df.shape[0], len(frequent_words)))

    # for each email
    for i in range(train_df.shape[0]):
        # create a list of vocabs in an email
        vocabs = train_df.text[i].split(" ")

        # for each words in the vocabulary map
        for j in range(len(frequent_words)):
            # count the number of times the jth word appears in the email
            count_train[i,j] = vocabs.count(frequent_words[j])
    
    return pd.DataFrame(count_train, columns = frequent_words)


def main():
    """
    Main file to run from the command line.
    """
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",
                        default="spamAssassin.data",
                        help="filename of the input data")
    args = parser.parse_args()
    train_df, test_df = model_assessment(args.data)
    vocab_map = build_vocab_map(train_df)
    binary_train = construct_binary(train_df, vocab_map)
    binary_test = construct_binary(test_df, vocab_map)
    count_train = construct_count(train_df, vocab_map)
    count_test = construct_count(test_df, vocab_map)

    y_train = pd.DataFrame(train_df.y)
    y_test = pd.DataFrame(test_df.y)

    y_train.to_csv('yTrain.csv', index = False)
    y_test.to_csv('yTest.csv', index = False)
    binary_train.to_csv('binary_train.csv', index = False)
    binary_test.to_csv('binary_test.csv', index = False)
    count_train.to_csv('count_train.csv', index = False)
    count_test.to_csv('count_test.csv', index = False)



if __name__ == "__main__":
    main()
