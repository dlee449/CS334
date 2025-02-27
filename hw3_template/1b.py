import selFeat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("eng_xTrain.csv")
yTrain = pd.read_csv("eng_yTrain.csv")
# extract features
df = selFeat.extract_features(df)
# add yTrain to df
df['y'] = yTrain

# Calculate Pearson correlation as correlation matrix
corrMat = df.corr()

fig = plt.figure(figsize=(10,10))
snsPlot = sns.heatmap(corrMat, cmap = 'coolwarm')
print(corrMat)
# filter out features with correlation greater than 0.5
filter_feat = corrMat[(abs(corrMat) < 0.5)]
print(filter_feat)
filter_feat = filter_feat.dropna(thresh=len(filter_feat)-1, axis=1)
print(filter_feat)
feature_to_use = list(filter_feat.loc[:, filter_feat.columns != 'y'].columns)
print(feature_to_use)



fig.savefig('1b.png')