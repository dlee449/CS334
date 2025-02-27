from sklearn import datasets
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# (a)
iris = datasets.load_iris()
iris = sns.load_dataset("iris")


# (b)
b_sepal_length = sns.boxplot(x = "species", y = "sepal_length", data = iris) # Sepal length boxplot of each species type
b_sepal_length.get_figure().savefig('q2(b)_sepal_length.png') # Save boxplot as png
plt.clf() # clear figure before creating next figure

b_sepal_width = sns.boxplot(x = "species", y = "sepal_width", data = iris) # Sepal width boxplot of each species type
b_sepal_width.get_figure().savefig('q2(b)_sepal_width.png') # Save boxplot as png
plt.clf() # clear figure before creating next figure

b_petal_length = sns.boxplot(x = "species", y = "petal_length", data = iris) # Petal length boxplot of each species type
b_petal_length.get_figure().savefig('q2(b)_petal_length.png') # Save boxplot as png
plt.clf()  # clear figure before creating next figure

b_petal_width = sns.boxplot(x = "species", y = "petal_width", data = iris) # Petal width boxplot of each species type
b_petal_width.get_figure().savefig('q2(b)_petal_width.png') # Save boxplot as png
plt.clf()  # clear figure before creating next figure


# (c)
sns.lmplot(x = "petal_length", y = "petal_width", data = iris, hue = "species", fit_reg = False) # Petal scatter plot of samples 
plt.savefig('q2(c)_petaldistribution.png') # Save scatterplot as png

sns.lmplot(x = "sepal_length", y = "sepal_width", data = iris, hue = "species", fit_reg = False) # Sepal scatter plot of samples 
plt.savefig('q2(c)_sepaldistribution.png') # Save scatterplot as png

# (d)
"""
Petal length and width seems to be more difficult to be used to classify the species type.
However, Sepal length and width seems useful as the species are generally categorized as
setosa in the smallest size with sepal length of 1-2 and sepal width of 0-0.75,
versicolor in the medium size with sepal length of 3-5 and sepal width of 1-1.75, and
virginica in the smallest size with sepal length of 5-7 and sepal width of 1.5-2.5.
"""