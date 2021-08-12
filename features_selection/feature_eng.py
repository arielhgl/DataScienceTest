
# Feature Importance with Extra Trees Classifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# load data

cols = ['bar', 'xgt', 'qgg', 'lux', 'wsg', 'yyz', 'drt', 'gox', 'foo','boz', 'hrt', 'juu','target']

dataframe = pd.read_csv('training.csv', usecols=cols)

dataframe.corr()

fig, ax = plt.subplots()

corr = dataframe.corr()
ax = sns.heatmap(
    corr, 
    vmin=-.06, vmax=.06, center=0,
    cmap=sns.diverging_palette(20, 220, n=100),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
)

plt.show()


array = dataframe.to_numpy()
X = array[:,0:12]
Y = array[:,12]

np.abs(corr.iloc[12,:]).nlargest(8)
