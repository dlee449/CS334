from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from perceptron import file_to_numpy
from sklearn.linear_model import LogisticRegressionCV
import numpy as np

binary_train = file_to_numpy("binary_train.csv")
binary_test = file_to_numpy("binary_test.csv")
count_train = file_to_numpy("count_train.csv")
count_test = file_to_numpy("count_test.csv")
yTrain = file_to_numpy("yTrain.csv")
yTrain = np.ravel(yTrain)
yTest = file_to_numpy("yTest.csv")
yTest = np.ravel(yTest)

def multinomial(xTrain, yTrain, xTest, yTest):
    mistakes = 0
    mult_classifier = MultinomialNB()
    mult_classifier.fit(xTrain, yTrain)
    yHat = mult_classifier.predict(xTest)
    for i in range(len(yHat)):
        if(yHat[i] != yTest[i]):
            mistakes += 1
    return mistakes 

def bernoulli(xTrain, yTrain, xTest, yTest):
    mistakes = 0
    ber_classifier = BernoulliNB()
    ber_classifier.fit(xTrain, yTrain)
    yHat = ber_classifier.predict(xTest)
    for i in range(len(yHat)):
        if yHat[i] != yTest[i]:
            mistakes += 1
    return mistakes  

def logistic(xTrain, yTrain, xTest, yTest):
    mistakes = 0
    log_classifier = LogisticRegressionCV(cv = 5, random_state = 1, max_iter = 1000)
    log_classifier.fit(xTrain, yTrain)
    yHat = log_classifier.predict(xTest)
    for i in range(len(yHat)):
        if(yHat[i] != yTest[i]):
            mistakes += 1
    return mistakes 


print("------------------------------------------------------------")
print("Binary Dataset")
bernoulli_mistakes = bernoulli(binary_train, yTrain, binary_test, yTest)
multinomial_mistakes = multinomial(binary_train, yTrain, binary_test, yTest)
logictic_mistakes = logistic(binary_train, yTrain, binary_test, yTest)
print("BernoulliNB mistakes: ", bernoulli_mistakes)
print("MultinomialNB mistakes: ", multinomial_mistakes)
print("Logistic Regression mistakes: ", logictic_mistakes)
print("------------------------------------------------------------")

print()
print("------------------------------------------------------------")
print("Count Dataset")
bernoulli_mistakes = bernoulli(count_train, yTrain, count_test, yTest)
multinomial_mistakes = multinomial(count_train, yTrain, count_test, yTest)
logictic_mistakes = logistic(count_train, yTrain, count_test, yTest)
print("BernoulliNB mistakes: ", bernoulli_mistakes)
print("MultinomialNB mistakes: ", multinomial_mistakes)
print("Logistic Regression mistakes: ", logictic_mistakes)
print("------------------------------------------------------------")