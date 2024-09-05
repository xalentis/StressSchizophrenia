# Experiment 1a and 1b, 2-class models for both types of schizophrenia data

import numpy as np
import pandas as pd
import warnings
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import random
import shap
import seaborn as sns
from sklearn.metrics import confusion_matrix
from scipy.stats import norm

matplotlib.use("TkAgg")
warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
random.seed(42)

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0, vmax=1)
    else:
        plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title)
    plt.colorbar()


    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45,fontsize=15)
        plt.yticks(tick_marks, target_names,fontsize=15)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",fontsize=15,
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()

##################################################################################################################################################################
# Generalization of Task and its healthy controls
##################################################################################################################################################################
schizophrenia_task_S = pd.read_csv("Dataset_Schizophrenia_Task_H.csv")
schizophrenia_task_H = pd.read_csv("Dataset_Schizophrenia_Task_S.csv")
dataset = pd.concat([schizophrenia_task_S, schizophrenia_task_H], axis=0 ,ignore_index = True)

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values

model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42)
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5], subsample=[0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.3,0.5,0.8,0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

##################################################################################################################################################################
# XGBoost Train, test and score
##################################################################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42, n_estimators=200, max_depth=2, eta=0.5, subsample=0.5, colsample_bytree=0.8, \
                          eval_metric=["auc"], num_rounds=2000, early_stopping_rounds=10)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
preds = model.predict(X_test)
print("Test F1 Score :",f1_score(y_test, preds, average=None))
print("Test weighted F1 Score :",f1_score(y_test, preds, average="weighted"))
# Test F1 Score : [0.83333333 0.83333333]
# Test weighted F1 Score :0.8333333333333334

##################################################################################################################################################################
# XGBoost LOSO validation
##################################################################################################################################################################
subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
y_true = []
y_pred = []
for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject,:]
    val_subset = dataset.loc[dataset["Subject"] == subject,:]
    X = train_subset.iloc[:, :75]
    y = train_subset.iloc[:,75:76]["Condition"]
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    model = model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42, n_estimators=200, max_depth=2, eta=0.5, subsample=0.5, colsample_bytree=0.8, \
                                    eval_metric=["auc"], early_stopping_rounds=10,)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X = val_subset.iloc[:, :75]
    y = val_subset.iloc[:,75:76]["Condition"].values[0]
    yhat = model.predict(X)[0]
    y_true.append(y)
    y_pred.append(yhat)

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, target_names=["Healthy", "Task"])

##################################################################################################################################################################
# Generalization of Rest and its healthy controls
##################################################################################################################################################################
schizophrenia_rest_S = pd.read_csv("Dataset_Schizophrenia_Rest_H.csv")
schizophrenia_rest_H = pd.read_csv("Dataset_Schizophrenia_Rest_S.csv")

dataset = pd.concat([schizophrenia_rest_S, schizophrenia_rest_H], axis=0 ,ignore_index = True)

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values

model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42)
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5], subsample=[0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.3,0.5,0.8,0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

##################################################################################################################################################################
# XGBoost Train, test and score
##################################################################################################################################################################
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42, n_estimators=100, max_depth=2, eta=0.3, subsample=0.3, colsample_bytree=0.8, \
                          eval_metric=["auc"], num_rounds=2000, early_stopping_rounds=10)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
preds = model.predict(X_test)
print("Test F1 Score :",f1_score(y_test, preds, average=None))
print("Test weighted F1 Score :",f1_score(y_test, preds, average="weighted"))
# Test F1 Score : [0.66666667 0.83333333]
# Test weighted F1 Score : 0.7962962962962963

##################################################################################################################################################################
# XGBoost LOSO validation
##################################################################################################################################################################
subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
y_true = []
y_pred = []
for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject,:]
    val_subset = dataset.loc[dataset["Subject"] == subject,:]
    X = train_subset.iloc[:, :75]
    y = train_subset.iloc[:,75:76]["Condition"]
    y = y.values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
    model = model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42, n_estimators=100, max_depth=2, eta=0.3, subsample=0.3, colsample_bytree=0.8, \
                                    eval_metric=["auc"], early_stopping_rounds=10,)
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X = val_subset.iloc[:, :75]
    y = val_subset.iloc[:,75:76]["Condition"].values[0]
    yhat = model.predict(X)[0]
    y_true.append(y)
    y_pred.append(yhat)

cm = confusion_matrix(y_true, y_pred)
plot_confusion_matrix(cm, target_names=["Healthy", "Rest"])
