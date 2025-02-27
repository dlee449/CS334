import argparse
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import warnings

# 3(a) / justification of k = 5 at the bottom
def gs_model(classifier, parameters, xTrain, yTrain):
    gs = GridSearchCV(estimator = classifier,
                      param_grid =  parameters,
                      cv = 5,
                      scoring = 'roc_auc')
    gs.fit(xTrain, yTrain)
    return gs

# 3(b,c)
def train_remove(model, xTrain, yTrain, xTest, yTest):
    train_size = len(xTrain)
    result = []
    for size_percentage in [1.00, 0.95, 0.90, 0.80]:
        removed_size = int(train_size*size_percentage)
        removed_index = np.random.choice(train_size, removed_size, replace=False)
        removed_xTrain = xTrain.iloc[removed_index, :]
        removed_yTrain = yTrain.loc[removed_index, 'label']
        removed_yTrain = removed_yTrain.to_numpy()
        subset = train_test(model.best_estimator_,
                          removed_xTrain,
                          removed_yTrain,
                          xTest, 
                          yTest)
        result.append(subset)
    return result

def train_test(model, xTrain, yTrain, xTest, yTest):
    # fit the data to the training dataset
    model.fit(xTrain, yTrain)
    # predict training and testing probabilties
    yHatTrain = model.predict_proba(xTrain)
    yHatTest = model.predict_proba(xTest)
    # calculate auc for training
    fpr, tpr, thresholds = metrics.roc_curve(yTrain,
                                             yHatTrain[:, 1])
    trainAuc = metrics.auc(fpr, tpr)
    # calculate auc for test dataset
    fpr, tpr, thresholds = metrics.roc_curve(yTest,
                                             yHatTest[:, 1])
    
    # calculate accuracy for training
    testAuc = metrics.auc(fpr, tpr)
    yHatTrain = model.predict(xTrain)
    trainAcc = metrics.accuracy_score(yTrain, yHatTrain)
    
    # calculate accuracy for test dataset
    yHatTest = model.predict(xTest)
    testAcc = metrics.accuracy_score(yTest, yHatTest)
    return [trainAuc, testAuc, trainAcc, testAcc]

def main():
    warnings.filterwarnings("ignore")

    xTrain = pd.read_csv("q4xTrain.csv")
    yTrain = pd.read_csv("q4yTrain.csv")
    xTest = pd.read_csv("q4xTest.csv")
    yTest = pd.read_csv("q4yTest.csv")

    #parameters for each model
    params_KNN = {'n_neighbors': range(1,20)}
    params_DT = {'criterion': ['gini', 'entropy'],
                 'max_depth': range(1, 9),
                 'min_samples_leaf': range(1, 9)}
    
    #create knn model using GridSearchCV
    knn_model = gs_model(KNeighborsClassifier(), 
                                params_KNN,
                                xTrain, yTrain)
    #create decision tree model using GridSearchCV
    DT_model = gs_model(DecisionTreeClassifier(),
                               params_DT,
                               xTrain, yTrain)
    
    #Train the k-nn with 3 additional removed datasets
    knn_result = train_remove(knn_model, xTrain, yTrain, xTest, yTest)
    #Train the decision tree with 3 additional removed datasets
    DT_result = train_remove(DT_model, xTrain, yTrain, xTest, yTest)

    #create dataframes for the results from knn and decision tree for printing
    knn_print = pd.DataFrame(knn_result, columns = ['0%', '5%', '10%', '20%'],
                             index = ['trainAuc', 'testAuc', 'trainAcc', 'testAcc'])
    DT_print = pd.DataFrame(DT_result, columns = ['0%', '5%', '10%', '20%'],
                             index = ['trainAuc', 'testAuc', 'trainAcc', 'testAcc'])
    
    print("---------------------   KNN   ------------------------")
    print(knn_model.best_params_)
    print(knn_print)
    print("---------------------   DT    ------------------------")
    print(DT_model.best_params_)
    print(DT_print)

    #Create table for reporting AUC and accuracy of the 8 different models
    knn = pd.DataFrame(knn_result).T
    knn.columns = ['trainAuc', 'testAuc', 'trainAcc', 'testAcc']
    knn.insert(0, "Removed", ['0%', '5%', '10%', '20%'])
    knn.index = ['k-nn', 'k-nn', 'k-nn', 'k-nn']

    DT = pd.DataFrame(DT_result).T
    DT.columns = ['trainAuc', 'testAuc', 'trainAcc', 'testAcc']
    DT.insert(0, "Removed", ['0%', '5%', '10%', '20%'])
    DT.index = ['Decision tree', 'Decision tree', 'Decision tree', 'Decision tree']
    result = pd.concat([knn, DT])
    result.to_csv('3(d).csv')

    


if __name__ == "__main__":
    main()

"""
3(a)
For k-nn, the optimal hyperparamter was 
{'n_neighbors': 14}.

For decision tree, the optimal hyperparameters were 
{'criterion': 'gini', 'max_depth': 4, 'min_samples_leaf': 7}

I chose k = 5 as the results from question 2 showed that k = 5 and 10 
resulted in a relatively high score for test AUC 
and I wanted to minimize the computational time so I chose 5 over 10.

3(d)
Table is saved as 3(d).csv.
I expected that the testAuc and testAcc to decrease while the 
removed percentage increased. However, 
trainAuc, testAuc, trainAcc, testAcc were varied in the differences
of removed percentage.

"""