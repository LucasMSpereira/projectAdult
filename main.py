#%%
import pip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sklearn as sk
import copy
import sklearn.feature_selection as ftSel
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

df = pd.read_csv('./docs/data/adult.csv') # read dataset
# general idea of dataset
df.head(8) # show first 8 lines
df.__dict__ # fields and info
df.columns # columns
df.shape # number of columns and lines
df.dtypes # types of each column
df.nunique() # amount of distinct values in each column
np.transpose(df.describe()) # descriptive statistics of numerical columns
# check which columns have missing values ('?')
for k, v in df.items():
  print(k, end = '\t')
  print(
    list(dict(df[k].value_counts()).keys())[0]
  )
#%%
## Dataprep
# replace missing values with the mode of the respective attribute
for k, v in df.items():
  counts = dict(df[k].value_counts())
  if '?' in counts:
    print(k, '\t', round(
      counts['?']/len(df.index)*100, 2
    ), '%')
    df[k] = df[k].replace('?', list( dict( counts ).keys() )[0])
# replace school grades with "school"
df['education'] = df['education'].replace('11th', 'school')
df['education'] = df['education'].replace('10th', 'school')
df['education'] = df['education'].replace('7th-8th', 'school')
df['education'] = df['education'].replace('5th-6th', 'school')
df['education'] = df['education'].replace('9th', 'school')
df['education'] = df['education'].replace('12th', 'school')
df['education'] = df['education'].replace('1st-4th', 'school')
# turn 'income' into binary
df.income = df.income.replace('<=50K', 0)
df.income = df.income.replace('>50K', 1)
#%%
## Plots
if False:
  sns.countplot(x = df['hours-per-week'], data = df) # histogram plot with seaborn
  df['hours-per-week'].plot(kind = "hist") # histogram plot with pyplot
  plt.xticks(rotation = 60)
  df.hist(figsize = (12, 12), layout = (3, 3), sharex = False) # general histogram (from pandas?)
  sns.heatmap(df.corr(), annot = True) # correlation heatmap
# %%
## Discretization of numerical features
numFeats = df.drop(['income', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'gender', 'native-country'], axis = 1) 
targets = df['income'] # separate targets
kbins = sk.preprocessing.KBinsDiscretizer(n_bins = 6, encode = 'ordinal', strategy = 'uniform')
discNumFeats = pd.DataFrame(kbins.fit_transform(numFeats))
# %%
## Label encoding
dfCategs = df.drop(['age', 'fnlwgt', 'educational-num', 'capital-gain', 'capital-loss', 'hours-per-week', 'income'], axis = 1)
dfCategsEncode = dfCategs.apply(LabelEncoder().fit_transform)
#%%
### Feature selection
amount = 7 # amount of resulting features
# concatenate encoded categorical features with discretized numerical features
newFeats = pd.concat([discNumFeats, dfCategsEncode], axis = 1)
# Study relationship between each feature and the targets
# featsChi2 = ftSel.SelectKBest(ftSel.chi2, k = amount).fit(newFeats, targets) # scoring by chi-squared
# featsF = ftSel.SelectKBest(ftSel.f_classif, k = amount).fit(newFeats, targets) # scoring by ANOVA F-value
# featsMutual = ftSel.SelectKBest(ftSel.mutual_info_classif, k = amount).fit(newFeats, targets) # scoring by mutual information
# selectFeatsChi2 = ftSel.SelectKBest(ftSel.chi2, k = amount).fit_transform(newFeats, targets) # scoring by chi-squared
selectFeatsF = ftSel.SelectKBest(ftSel.f_classif, k = amount).fit_transform(newFeats, targets) # scoring by ANOVA F-value
# selectFeatsMutual = ftSel.SelectKBest(ftSel.mutual_info_classif, k = amount).fit_transform(newFeats, targets) # scoring by mutual information
### Split train and test data
feats_train, feats_test, targets_train, targets_test = train_test_split(selectFeatsF, targets, test_size=0.25, random_state=0)
#%%
## kNN Classifier with cross-validation
# knn = sk.neighbors.KNeighborsClassifier(4, weights = 'distance')
params = {
  'algorithm': ['ball_tree', 'kd_tree'],
  'leaf_size': [30],
  'metric': ['minkowski', 'manhattan'],
  'metric_params': [None],
  'n_jobs': [None],
  'n_neighbors': [4, 5, 6],
  'p': [2],
  'weights': ['distance', 'uniform']
}
cvKNN = sk.model_selection.GridSearchCV(
  sk.neighbors.KNeighborsClassifier(), params, cv = 5,
  scoring = ['accuracy', 'precision', 'recall'], verbose = 5, refit = 'accuracy'
)
knn = cvKNN.fit(feats_train, targets_train)
print(
  'test accuracy', cvKNN['test_accuracy'], '\n',
  'test precision', cvKNN['test_precision'], '\n',
  'test recall', cvKNN['test_recall']
)