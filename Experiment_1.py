import random
import numpy as np
import pandas as pd
import warnings
import itertools
import shap
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from neuroCombat import neuroCombat
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

matplotlib.use("TkAgg")
warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
random.seed(42)

def plot_confusion_matrix(cm, target_names, title='Confusion matrix', cmap=None, normalize=True):
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
    return plt

##################################################################################################################################################################
# Generalization of Rest and its healthy controls
##################################################################################################################################################################
schizophrenia_rest_H = pd.read_csv("Dataset_Schizophrenia_Rest_H.csv")
schizophrenia_rest_S = pd.read_csv("Dataset_Schizophrenia_Rest_S.csv")
schizophrenia_rest_H['batch'] = 1
schizophrenia_rest_S['batch'] = 2
dataset_unharmonized = pd.concat([schizophrenia_rest_S, schizophrenia_rest_H], axis=0, ignore_index=True)

##################################################################################################################################################################
# ComBat Harmonization to Correct for Batch Effects
##################################################################################################################################################################
feature_cols = dataset_unharmonized.columns[:75].tolist()
metadata_cols = ['Condition', 'Subject', 'batch']
features_df = dataset_unharmonized[feature_cols]
metadata_df = dataset_unharmonized[metadata_cols]
features_transposed = features_df.T
covars = pd.DataFrame({
    'batch': metadata_df['batch']
})
harmonized_data_array = neuroCombat(
    dat=features_transposed,
    covars=covars,
    batch_col='batch'
)['data']
harmonized_features_df = pd.DataFrame(harmonized_data_array.T, columns=feature_cols)
dataset = pd.concat([harmonized_features_df, metadata_df.reset_index(drop=True)], axis=1)

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values

model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42)
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5, 0.7, 0.9], subsample=[0.1, 0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.1, 0.3, 0.5, 0.7, 0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
best_hyperparams = grid_result.best_params_

##################################################################################################################################################################
# XGBoost LOSO validation
##################################################################################################################################################################
subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
y_true = []
y_pred = []
y_score = []
subject_level_results_rest = []

for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject, :]
    val_subset = dataset.loc[dataset["Subject"] == subject, :]
    X_train_full = train_subset.iloc[:, :75]
    y_train_full = train_subset.iloc[:, 75:76]["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.30, random_state=42
    )
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        seed=42,
        num_class=2,
        eval_metric=["auc"],
        early_stopping_rounds=10,
        **best_hyperparams 
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X_val = val_subset.iloc[:, :75]
    y_val = val_subset.iloc[:, 75:76]["Condition"].values
    yhat = model.predict(X_val)
    subject_level_results_rest.append({'subject': subject, 'accuracy': balanced_accuracy_score(y_val, yhat), 'X_val': X_val, 'y_val': y_val, 'yhat': yhat})
    yhat_proba = model.predict_proba(X_val)[:, 1]
    y_true.extend(y_val)
    y_pred.extend(yhat)
    y_score.extend(yhat_proba)

balanced_auc = roc_auc_score(y_true, y_score)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced AUC: {balanced_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
cm = confusion_matrix(y_true, y_pred)
plt = plot_confusion_matrix(cm, target_names=["Healthy", "Rest"])
plt.show()

X_rest = dataset.iloc[:, :75]
y_rest = dataset.iloc[:, 75:76]["Condition"].values
model_rest = xgb.XGBClassifier(
    objective="multi:softmax",
    seed=42,
    num_class=2,
    **best_hyperparams
)
model_rest.fit(X_rest, y_rest)

##################################################################################################################################################################
# Generalization of Task and its healthy controls
##################################################################################################################################################################
schizophrenia_task_H = pd.read_csv("Dataset_Schizophrenia_Task_H.csv")
schizophrenia_task_S = pd.read_csv("Dataset_Schizophrenia_Task_S.csv")
dataset = pd.concat([schizophrenia_task_S, schizophrenia_task_H], axis=0 ,ignore_index = True)
schizophrenia_task_H['batch'] = 1
schizophrenia_task_S['batch'] = 2
dataset_unharmonized = pd.concat([schizophrenia_task_S, schizophrenia_task_H], axis=0, ignore_index=True)

##################################################################################################################################################################
# ComBat Harmonization to Correct for Batch Effects
##################################################################################################################################################################
feature_cols = dataset_unharmonized.columns[:75].tolist()
metadata_cols = ['Condition', 'Subject', 'batch']
features_df = dataset_unharmonized[feature_cols]
metadata_df = dataset_unharmonized[metadata_cols]
features_transposed = features_df.T
covars = pd.DataFrame({
    'batch': metadata_df['batch']
})
harmonized_data_array = neuroCombat(
    dat=features_transposed,
    covars=covars,
    batch_col='batch'
)['data']
harmonized_features_df = pd.DataFrame(harmonized_data_array.T, columns=feature_cols)
dataset = pd.concat([harmonized_features_df, metadata_df.reset_index(drop=True)], axis=1)

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values
model = xgb.XGBClassifier(objective="multi:softmax", num_class=2, seed=42)
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5, 0.7, 0.9], subsample=[0.1, 0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.1, 0.3, 0.5, 0.7, 0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
best_hyperparams = grid_result.best_params_

##################################################################################################################################################################
# XGBoost LOSO validation
##################################################################################################################################################################
subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
y_true = []
y_pred = []
y_score = []
subject_level_results_task = []

for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject, :]
    val_subset = dataset.loc[dataset["Subject"] == subject, :]
    X_train_full = train_subset.iloc[:, :75]
    y_train_full = train_subset.iloc[:, 75:76]["Condition"].values
    X_train, X_test, y_train, y_test = train_test_split(
        X_train_full, y_train_full, test_size=0.30, random_state=42
    )
    model = xgb.XGBClassifier(
        objective="multi:softmax",
        seed=42,
        num_class=2,
        eval_metric=["auc"],
        early_stopping_rounds=10,
        **best_hyperparams 
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X_val = val_subset.iloc[:, :75]
    y_val = val_subset.iloc[:, 75:76]["Condition"].values
    yhat = model.predict(X_val)
    subject_level_results_task.append({'subject': subject, 'accuracy': balanced_accuracy_score(y_val, yhat), 'X_val': X_val, 'y_val': y_val, 'yhat': yhat})
    yhat_proba = model.predict_proba(X_val)[:, 1]
    y_true.extend(y_val)
    y_pred.extend(yhat)
    y_score.extend(yhat_proba)

balanced_auc = roc_auc_score(y_true, y_score)
f1 = f1_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced AUC: {balanced_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
cm = confusion_matrix(y_true, y_pred)
plt = plot_confusion_matrix(cm, target_names=["Healthy", "Task"])
plt.show()

##################################################################################################################################################################
# SHAP FEATURE IMPORTANCE ANALYSIS
##################################################################################################################################################################
X_task = dataset.iloc[:, :75]
y_task = dataset.iloc[:, 75:76]["Condition"].values
model_task = xgb.XGBClassifier(
    objective="multi:softmax",
    seed=42,
    num_class=2,
    **best_hyperparams
)
model_task.fit(X_task, y_task)

explainer_rest = shap.TreeExplainer(model_rest)
shap_values_rest = explainer_rest.shap_values(X_rest)
plt.figure()
shap.summary_plot(
    shap_values_rest[1],
    X_rest,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance for REST vs Healthy", fontsize=16)
plt.tight_layout()

explainer_task = shap.TreeExplainer(model_task)
shap_values_task = explainer_task.shap_values(X_task)
plt.figure()
shap.summary_plot(
    shap_values_task[1],
    X_task,
    plot_type="bar",
    show=False
)
plt.title("SHAP Feature Importance for TASK vs Healthy", fontsize=16)
plt.tight_layout()
plt.show()

##################################################################################################################################################################
# DETAILED SUBJECT-LEVEL PERFORMANCE ANALYSIS
##################################################################################################################################################################
results_df_rest = pd.DataFrame(subject_level_results_rest)
results_df_task = pd.DataFrame(subject_level_results_task)
print("\n" + "="*80)
print("SUBJECT-LEVEL PERFORMANCE ANALYSIS: REST VS HEALTHY")
print("="*80)

accuracies_rest = results_df_rest['accuracy']
mean_acc_rest = np.mean(accuracies_rest)
std_acc_rest = np.std(accuracies_rest)
ci_95_rest = 1.96 * std_acc_rest / np.sqrt(len(accuracies_rest))

print("\n--- Subject-Level Accuracy Statistics (REST) ---")
print(f"| Metric                      | Value      |")
print(f"|-----------------------------|------------|")
print(f"| Mean Balanced Accuracy      | {mean_acc_rest:10.4f} |")
print(f"| Standard Deviation          | {std_acc_rest:10.4f} |")
print(f"| 95% Confidence Interval     |  +/- {ci_95_rest:7.4f} |")
print("----------------------------------------------")

print("\n" + "="*80)
print("SUBJECT-LEVEL PERFORMANCE ANALYSIS: TASK VS HEALTHY")
print("="*80)
accuracies_task = results_df_task['accuracy']
mean_acc_task = np.mean(accuracies_task)
std_acc_task = np.std(accuracies_task)
ci_95_task = 1.96 * std_acc_task / np.sqrt(len(accuracies_task))
print("\n--- Subject-Level Accuracy Statistics (TASK) ---")
print(f"| Metric                      | Value      |")
print(f"|-----------------------------|------------|")
print(f"| Mean Balanced Accuracy      | {mean_acc_task:10.4f} |")
print(f"| Standard Deviation          | {std_acc_task:10.4f} |")
print(f"| 95% Confidence Interval     |  +/- {ci_95_task:7.4f} |")
print("----------------------------------------------")
