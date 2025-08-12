import random
import numpy as np
import pandas as pd
import pickle
import warnings
import itertools
import shap
import xgboost as xgb
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp, ttest_ind
from statsmodels.stats.multitest import multipletests
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
# Re-classify model into 3 classes, healthy and schizophrenia rest and task
##################################################################################################################################################################
schizophrenia_rest = pd.read_csv("Dataset_Schizophrenia_Rest.csv")
schizophrenia_task = pd.read_csv("Dataset_Schizophrenia_Task.csv")
schizophrenia_task.loc[schizophrenia_task.Condition == 1, "Condition"] = 2
schizophrenia_rest['batch'] = 1
schizophrenia_task['batch'] = 2
dataset = pd.concat([schizophrenia_rest, schizophrenia_task], axis=0, ignore_index=True)

##################################################################################################################################################################
# Load stress model and predict acute stress response across all subjects
##################################################################################################################################################################
model = pickle.load(open("stress_model.xgb", "rb"))

with open("stress_model_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
model = pickle.load(open("stress_model.xgb", "rb"))

X = scaler.transform(dataset.iloc[:, :75].values)
y = dataset.iloc[:,75:76]["Condition"]
y = y.values
preds = model.predict_proba(X)
preds = preds[:, 1]
dataset["Stress"] = preds
mean_stress = dataset.groupby("Condition")["Stress"].mean()

plt.bar(mean_stress.index, mean_stress.values)
plt.xlabel("Condition")
plt.ylabel("Mean Stress")
plt.title("Mean Stress per Condition")
plt.xticks(mean_stress.index)
plt.show()

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values

model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, seed=42)
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5, 0.7, 0.9], subsample=[0.1, 0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.1, 0.3, 0.5, 0.7, 0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
best_hyperparams = grid_result.best_params_

##################################################################################################################################################################
# XGBoost LOSO validation
##################################################################################################################################################################
FEATURE_COLS = dataset.columns[:75].tolist()
LABEL_COL = dataset.columns[75]
SUBJECT_COL = dataset.columns[76]

subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
y_true = []
y_pred = []
y_score = []
subject_level_results = []
diagnostics_rows = []

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
        num_class=3,
        eval_metric=["auc"],
        early_stopping_rounds=10,
        **best_hyperparams 
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X_val = val_subset.iloc[:, :75]
    y_val = val_subset.iloc[:, 75:76]["Condition"].values
    yhat = model.predict(X_val)
    subject_level_results.append({'accuracy': balanced_accuracy_score(y_val, yhat)})
    yhat_proba = model.predict_proba(X_val)
    y_true.extend(y_val)
    y_pred.extend(yhat)
    y_score.extend(yhat_proba)

    # subject-specific metrics
    bacc = balanced_accuracy_score(y_val, yhat)
    try:
        auc = roc_auc_score(y_val, yhat_proba, multi_class='ovr')
    except ValueError:
        auc = np.nan
    f1 = f1_score(y_val, yhat, average='weighted', zero_division=0)
    prec = precision_score(y_val, yhat, average='weighted', zero_division=0)
    rec = recall_score(y_val, yhat, average='weighted', zero_division=0)

    # statistical tests for distribution differences
    ks_pvals = []
    t_pvals = []
    for fi in range(len(FEATURE_COLS)):
        subj_vals = X_val.iloc[:, fi].values
        train_vals = X_train_full.iloc[:, fi].values
        ks_pvals.append(ks_2samp(subj_vals, train_vals).pvalue)
        t_pvals.append(ttest_ind(subj_vals, train_vals, equal_var=False).pvalue)

    # multiple comparison correction
    ks_signif = multipletests(ks_pvals, alpha=0.05, method='fdr_bh')[0]
    t_signif = multipletests(t_pvals, alpha=0.05, method='fdr_bh')[0]
    diagnostics_rows.append({
        "subject": subject,
        "n_val_samples": len(y_val),
        "balanced_accuracy": bacc,
        "roc_auc": auc,
        "f1_score": f1,
        "precision": prec,
        "recall": rec,
        "mean_ks_pval": np.mean(ks_pvals),
        "mean_t_pval": np.mean(t_pvals),
        "prop_signif_features_ks": np.mean(ks_signif),
        "prop_signif_features_t": np.mean(t_signif)
    })

diagnostics_df = pd.DataFrame(diagnostics_rows)
diagnostics_df.to_csv("subject_diagnostics.csv", index=False)
y_score = np.vstack(y_score)
balanced_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced AUC: {balanced_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
cm = confusion_matrix(y_true, y_pred)
plt = plot_confusion_matrix(cm, target_names=["Healthy", "Rest", "Task"])
plt.show()

##################################################################################################################################################################
# SHAP FEATURE IMPORTANCE ANALYSIS
##################################################################################################################################################################
model = xgb.XGBClassifier(
    objective="multi:softmax",
    seed=42,
    num_class=3,
    **best_hyperparams
)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
class_names = ["Healthy", "Rest", "Task"]
plt.figure()
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    class_names=class_names,
    show=False
)
plt.title("SHAP Mean Feature Importance by Class", fontsize=16)
plt.tight_layout()
plt.show()

##################################################################################################################################################################
# DETAILED SUBJECT-LEVEL PERFORMANCE ANALYSIS
##################################################################################################################################################################
results_df = pd.DataFrame(subject_level_results)
print("\n" + "="*80)
print("SUBJECT-LEVEL PERFORMANCE ANALYSIS: 3-CLASS MODEL (Healthy, Rest, Task)")
print("="*80)
accuracies = results_df['accuracy']
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))
print("\n--- Subject-Level Accuracy Statistics ---")
print(f"| Metric                      | Value      |")
print(f"|-----------------------------|------------|")
print(f"| Mean Balanced Accuracy      | {mean_acc:10.4f} |")
print(f"| Standard Deviation          | {std_acc:10.4f} |")
print(f"| 95% Confidence Interval     |  +/- {ci_95:7.4f} |")
print("----------------------------------------------")

##################################################################################################################################################################
# XGBoost LOSO validation, after adjusting for acute stress
##################################################################################################################################################################
np.random.seed(42)
random.seed(42)

healthy_dataset = pd.read_csv("Dataset_Stress.csv")
relaxed_mean = healthy_dataset.loc[healthy_dataset["Condition"] == 0, dataset.columns[:75]].mean(axis=0)
stressed_mean = healthy_dataset.loc[healthy_dataset["Condition"] == 1, dataset.columns[:75]].mean(axis=0)
# adjust for acute stress:
diff = stressed_mean - relaxed_mean
dataset.loc[dataset["Condition"] == 2, dataset.columns[:75]] -= diff

##################################################################################################################################################################
# XGBoost Parameter search and tuning
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values
model = xgb.XGBClassifier(objective="multi:softmax", num_class=3, seed=42)
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
subject_level_results = []

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
        num_class=3,
        eval_metric=["auc"],
        early_stopping_rounds=10,
        **best_hyperparams 
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
    X_val = val_subset.iloc[:, :75]
    y_val = val_subset.iloc[:, 75:76]["Condition"].values
    yhat = model.predict(X_val)
    subject_level_results.append({'accuracy': balanced_accuracy_score(y_val, yhat)})
    yhat_proba = model.predict_proba(X_val)
    y_true.extend(y_val)
    y_pred.extend(yhat)
    y_score.extend(yhat_proba)

y_score = np.vstack(y_score)
balanced_auc = roc_auc_score(y_true, y_score, multi_class='ovr')
f1 = f1_score(y_true, y_pred, average='macro')
precision = precision_score(y_true, y_pred, average='macro')
recall = recall_score(y_true, y_pred, average='macro')
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced AUC: {balanced_auc:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"Balanced Accuracy: {balanced_acc:.4f}")
cm = confusion_matrix(y_true, y_pred)
plt = plot_confusion_matrix(cm, target_names=["Healthy", "Rest", "Task"])
plt.show()

##################################################################################################################################################################
# SHAP FEATURE IMPORTANCE ANALYSIS
##################################################################################################################################################################
model = xgb.XGBClassifier(
    objective="multi:softmax",
    seed=42,
    num_class=3,
    **best_hyperparams
)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)
class_names = ["Healthy", "Rest", "Task"]
plt.figure()
shap.summary_plot(
    shap_values,
    X,
    plot_type="bar",
    class_names=class_names,
    show=False
)
plt.title("SHAP Mean Feature Importance by Class", fontsize=16)
plt.tight_layout()
plt.show()

##################################################################################################################################################################
# DETAILED SUBJECT-LEVEL PERFORMANCE ANALYSIS
##################################################################################################################################################################
results_df = pd.DataFrame(subject_level_results)
print("\n" + "="*80)
print("SUBJECT-LEVEL PERFORMANCE ANALYSIS: 3-CLASS MODEL (Healthy, Rest, Task)")
print("="*80)
accuracies = results_df['accuracy']
mean_acc = np.mean(accuracies)
std_acc = np.std(accuracies)
ci_95 = 1.96 * std_acc / np.sqrt(len(accuracies))
print("\n--- Subject-Level Accuracy Statistics ---")
print(f"| Metric                      | Value      |")
print(f"|-----------------------------|------------|")
print(f"| Mean Balanced Accuracy      | {mean_acc:10.4f} |")
print(f"| Standard Deviation          | {std_acc:10.4f} |")
print(f"| 95% Confidence Interval     |  +/- {ci_95:7.4f} |")
print("----------------------------------------------")

