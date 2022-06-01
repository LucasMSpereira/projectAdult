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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

df = pd.read_csv('./docs/data/adult.csv') # read dataset
# print mode of each attribute
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
selectFeatsF = ftSel.SelectKBest(ftSel.f_classif, k = amount).fit_transform(newFeats, targets) # scoring by ANOVA F-value
### Split train and test data
feats_train, feats_test, targets_train, targets_test = train_test_split(selectFeatsF, targets, test_size=0.25)
#%%
## Function to build classifier, optimize its HP through CV and test it using holdout set
# Returns precision, accuracy and recall scores, alongside sk.model_selection.GridSearchCV object
def myModel(estimator, params, feats_train, targets_train, feats_test, targets_test, numFolds):
  # Grid search CV object
  cvModel = sk.model_selection.GridSearchCV(
    estimator, params, cv = numFolds,
    scoring = ['accuracy', 'precision', 'recall'], verbose = 3, refit = 'accuracy'
  )
  cvModel.fit(feats_train, targets_train) # Search for best hyperparameters
  bestCVmodel = cvModel.best_estimator_ # Best estimator obtained according to calssification accuracy
  # Return holdout tests with the best estimator
  modelPrec =  sk.metrics.precision_score(targets_test, bestCVmodel.predict(feats_test)) # precision
  modelAcc = sk.metrics.accuracy_score(targets_test, bestCVmodel.predict(feats_test)) # accuracy
  modelRec = sk.metrics.recall_score(targets_test, bestCVmodel.predict(feats_test)) # recall
  return [modelPrec, modelAcc, modelRec, cvModel]
#%%
## kNN Classifier
results = myModel(
  sk.neighbors.KNeighborsClassifier(),
  {
    'algorithm': ['ball_tree', 'kd_tree'],
    'leaf_size': [30],
    'metric': ['minkowski', 'manhattan'],
    'metric_params': [None],
    'n_jobs': [None],
    'n_neighbors': [4, 5, 6],
    'p': [2],
    'weights': ['distance', 'uniform']
  },
  feats_train, targets_train, feats_test, targets_test, 5
)
#%%
## Decision tree Classifier
results = myModel(
  DecisionTreeClassifier(),
  {
    'criterion': ['gini', 'entropy', 'log_loss'],
    'splitter': ['best', 'random'],
    'max_depth': [None],
    'min_samples_split': [2],
    "min_samples_leaf": [1, 5],
    "min_weight_fraction_leaf": [0.0],
    'max_features': ['sqrt', 'log2'],
    "random_state": [None],
    "max_leaf_nodes": [None],
    "min_impurity_decrease": [0.0],
    'class_weight': ['balanced', None],
    'ccp_alpha': [0.0],
  },
  feats_train, targets_train, feats_test, targets_test, 5
)
#%%
## Boosted tree Classifier
results = myModel(
  GradientBoostingClassifier(),
  {
    'loss': ['log_loss'],
    'learning_rate': [0.1, 0.7],
    'n_estimators': [100],
    'criterion': ['friedman_mse', 'squared_error'],
    'min_samples_split': [2],
    "min_samples_leaf": [1],
    "min_weight_fraction_leaf": [0.0],
    'max_depth': [None],
    "min_impurity_decrease": [0.0],
    'init': [None],
    "random_state": [None],
    'max_features': [1.0],
    "max_leaf_nodes": [None],
    'verbose': [1]
  },
  feats_train, targets_train, feats_test, targets_test, 4
)
#%%
## Multi-layer perceptron Classifier
results = myModel(
  MLPClassifier(),
  {
    'activation': ['logistic', 'relu'],
    'solver': ['lbfgs', 'adam'],
    'learning_rate': ['invscaling', 'adaptive'],
    'verbose': [True],
  },
  feats_train, targets_train, feats_test, targets_test, 5
)