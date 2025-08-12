import os
import numpy as np
import pandas as pd
import mne
import pickle
import random
import warnings
from xgboost import XGBClassifier
from mne.preprocessing import ICA
from mne_icalabel import label_components
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)
np.random.seed(42)
random.seed(42)

####################################################################################################
# Parameters & containers
####################################################################################################
channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "Fz", "Pz", "Cz"]

####################################################################################################
# Helper functions
####################################################################################################

def permutation_test(y_true_array, y_pred_array, metric_fn, n_permutations=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    obs = metric_fn(y_true_array, y_pred_array)
    count = 0
    n = len(y_true_array)
    for i in range(n_permutations):
        perm = rng.permutation(y_true_array)
        stat = metric_fn(perm, y_pred_array)
        if stat >= obs:
            count += 1
    pval = (count + 1) / (n_permutations + 1)
    return pval

def bootstrap_metric(y_true_array, y_pred_array, metric_fn, n_bootstraps=1000, random_state=42):
    rng = np.random.RandomState(random_state)
    stats = []
    n = len(y_true_array)
    for i in range(n_bootstraps):
        idxs = rng.randint(0, n, n)  # sample indices with replacement
        stat = metric_fn(y_true_array[idxs], y_pred_array[idxs])
        stats.append(stat)
    stats = np.array(stats)
    lower = np.percentile(stats, 2.5)
    upper = np.percentile(stats, 97.5)
    return lower, upper, stats

def psd_frontal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks=["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], fmin=0.5, fmax=45)
    psds, freqs = spectrum.get_data(return_freqs=True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis=1)
    whole_psd = np.multiply(whole_psd, 1e12)
    powers = pd.DataFrame(whole_psd, columns=[frequency + "FP1", frequency + "FP2", frequency + "F3",
                                             frequency + "F4", frequency + "F7", frequency + "F8", frequency + "Fz"])
    return powers

def psd_central(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks=["C3", "C4", "Cz"], fmin=0.5, fmax=45)
    psds, freqs = spectrum.get_data(return_freqs=True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis=1)
    whole_psd = np.multiply(whole_psd, 1e12)
    powers = pd.DataFrame(whole_psd, columns=[frequency + "C3", frequency + "C4", frequency + "Cz"])
    return powers

def psd_parietal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks=["P3", "P4", "Pz"], fmin=0.5, fmax=45)
    psds, freqs = spectrum.get_data(return_freqs=True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis=1)
    whole_psd = np.multiply(whole_psd, 1e12)
    powers = pd.DataFrame(whole_psd, columns=[frequency + "P3", frequency + "P4", frequency + "Pz"])
    return powers

def psd_occipital(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks=["O1", "O2"], fmin=0.5, fmax=45)
    psds, freqs = spectrum.get_data(return_freqs=True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis=-1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis=1)
    whole_psd = np.multiply(whole_psd, 1e12)
    powers = pd.DataFrame(whole_psd, columns=[frequency + "O1", frequency + "O2"])
    return powers

####################################################################################################
# Load or create Dataset_Stress.csv
####################################################################################################
eegstress_dataset = pd.DataFrame()
subjects = list(range(0, 36))
for subject in subjects:
    if subject < 10:
        subject = "0" + str(subject)
    else:
        subject = str(subject)

    # Relax state (session 1)
    raw = mne.io.read_raw_edf(os.path.join("Data", "EEGStress", "Subject" + subject + "_1.edf"), infer_types=True)
    raw.load_data()
    raw.set_channel_types(mapping={"ECG": "misc"})
    raw.set_eeg_reference(ref_channels="average")
    raw = raw.copy().pick_channels(channels)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    data_filtered = raw.filter(1,100)
    data_filtered = data_filtered.resample(sfreq=200)
    ica = ICA(n_components = 10, method = "infomax", random_state = 42, fit_params=dict(extended=True))
    ica.fit(data_filtered)
    ic_labels = label_components(data_filtered, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)
    data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
    epoch_whole = mne.make_fixed_length_epochs(data_filtered, duration=int(np.floor(raw.times[-1])), preload=True)[0]

    alpha_frontal = psd_frontal(epoch_whole, "Alpha", 8, 12)
    alpha_central = psd_central(epoch_whole, "Alpha", 8, 12)
    alpha_parietal = psd_parietal(epoch_whole, "Alpha", 8, 12)
    alpha_occipital = psd_occipital(epoch_whole, "Alpha", 8, 12)
    beta_frontal = psd_frontal(epoch_whole, "Beta", 12, 30)
    beta_central = psd_central(epoch_whole, "Beta", 12, 30)
    beta_parietal = psd_parietal(epoch_whole, "Beta", 12, 30)
    beta_occipital = psd_occipital(epoch_whole, "Beta", 12, 30)
    delta_frontal = psd_frontal(epoch_whole, "Delta", 0.5, 4)
    delta_central = psd_central(epoch_whole, "Delta", 0.5, 4)
    delta_parietal = psd_parietal(epoch_whole, "Delta", 0.5, 4)
    delta_occipital = psd_occipital(epoch_whole, "Delta", 0.5, 4)
    theta_frontal = psd_frontal(epoch_whole, "Theta", 4, 8)
    theta_central = psd_central(epoch_whole, "Theta", 4, 8)
    theta_parietal = psd_parietal(epoch_whole, "Theta", 4, 8)
    theta_occipital = psd_occipital(epoch_whole, "Theta", 4, 8)
    gamma_frontal = psd_frontal(epoch_whole, "Gamma", 30, 45)
    gamma_central = psd_central(epoch_whole, "Gamma", 30, 45)
    gamma_parietal = psd_parietal(epoch_whole, "Gamma", 30, 45)
    gamma_occipital = psd_occipital(epoch_whole, "Gamma", 30, 45)

    subset = pd.concat([
        alpha_frontal, alpha_central, alpha_parietal, alpha_occipital,
        beta_frontal, beta_central, beta_parietal, beta_occipital,
        delta_frontal, delta_central, delta_parietal, delta_occipital,
        theta_frontal, theta_central, theta_parietal, theta_occipital,
        gamma_frontal, gamma_central, gamma_parietal, gamma_occipital
    ], axis=1)

    subset["Condition"] = 0
    subset["Subject"] = "EEGStressRelaxed_" + subject
    eegstress_dataset = pd.concat([eegstress_dataset, subset], axis=0, ignore_index=True)

    # Stressed state (session 2)
    raw = mne.io.read_raw_edf(os.path.join("Data", "EEGStress", "Subject" + subject + "_2.edf"), infer_types=True)
    raw.load_data()
    raw.set_channel_types(mapping={"ECG": "misc"})
    raw.set_eeg_reference(ref_channels=["A2-A1"])
    raw = raw.copy().pick_channels(channels)
    montage = mne.channels.make_standard_montage("standard_1020")
    raw.set_montage(montage)
    data_filtered = raw.copy().notch_filter(freqs=(25, 50), picks=channels)
    data_filtered = data_filtered.copy().filter(0.5, 45)
    data_filtered = data_filtered.copy().resample(sfreq=200)

    ica = ICA(n_components = 10, method = "infomax", random_state = 42, fit_params=dict(extended=True))
    ica.fit(data_filtered)
    ic_labels = label_components(data_filtered, ica, method="iclabel")
    labels = ic_labels["labels"]
    exclude_idx = [idx for idx, label in enumerate(labels) if label not in ["brain", "other"]]
    reconstructed = data_filtered.copy()
    ica.apply(reconstructed, exclude=exclude_idx)
    data_filtered = reconstructed.filter(l_freq=1, h_freq=45)
    epoch_whole = mne.make_fixed_length_epochs(data_filtered, duration=int(np.floor(raw.times[-1])), preload=True)[0]

    alpha_frontal = psd_frontal(epoch_whole, "Alpha", 8, 12)
    alpha_central = psd_central(epoch_whole, "Alpha", 8, 12)
    alpha_parietal = psd_parietal(epoch_whole, "Alpha", 8, 12)
    alpha_occipital = psd_occipital(epoch_whole, "Alpha", 8, 12)
    beta_frontal = psd_frontal(epoch_whole, "Beta", 12, 30)
    beta_central = psd_central(epoch_whole, "Beta", 12, 30)
    beta_parietal = psd_parietal(epoch_whole, "Beta", 12, 30)
    beta_occipital = psd_occipital(epoch_whole, "Beta", 12, 30)
    delta_frontal = psd_frontal(epoch_whole, "Delta", 0.5, 4)
    delta_central = psd_central(epoch_whole, "Delta", 0.5, 4)
    delta_parietal = psd_parietal(epoch_whole, "Delta", 0.5, 4)
    delta_occipital = psd_occipital(epoch_whole, "Delta", 0.5, 4)
    theta_frontal = psd_frontal(epoch_whole, "Theta", 4, 8)
    theta_central = psd_central(epoch_whole, "Theta", 4, 8)
    theta_parietal = psd_parietal(epoch_whole, "Theta", 4, 8)
    theta_occipital = psd_occipital(epoch_whole, "Theta", 4, 8)
    gamma_frontal = psd_frontal(epoch_whole, "Gamma", 30, 45)
    gamma_central = psd_central(epoch_whole, "Gamma", 30, 45)
    gamma_parietal = psd_parietal(epoch_whole, "Gamma", 30, 45)
    gamma_occipital = psd_occipital(epoch_whole, "Gamma", 30, 45)

    subset = pd.concat([
        alpha_frontal, alpha_central, alpha_parietal, alpha_occipital,
        beta_frontal, beta_central, beta_parietal, beta_occipital,
        delta_frontal, delta_central, delta_parietal, delta_occipital,
        theta_frontal, theta_central, theta_parietal, theta_occipital,
        gamma_frontal, gamma_central, gamma_parietal, gamma_occipital
    ], axis=1)

    subset["Condition"] = 1
    subset["Subject"] = "EEGStressStressed_" + subject
    eegstress_dataset = pd.concat([eegstress_dataset, subset], axis=0, ignore_index=True)

dataset = eegstress_dataset.copy()
dataset.to_csv("Dataset_Stress.csv", index=False)
print("Saved Dataset_Stress.csv")

####################################################################################################
# Modeling: parameter search + a train/test demonstration (no LOSO here)
####################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:, 75:76]["Condition"].values.ravel()

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5, 0.7, 0.9], subsample=[0.1, 0.3, 0.5, 0.7, 0.9], colsample_bytree=[0.1, 0.3,0.5,0.8,0.9])

kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
print("Best (CV) score: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y, random_state=42)
scaler_local = MinMaxScaler()
X_train_scaled = scaler_local.fit_transform(X_train.values)
X_test_scaled = scaler_local.transform(X_test.values)

best_params = grid_result.best_params_
model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                      n_estimators=best_params.get('n_estimators', 200),
                      max_depth=best_params.get('max_depth', 4),
                      learning_rate=best_params.get('learning_rate', 0.1),
                      subsample=best_params.get('subsample', 0.8),
                      colsample_bytree=best_params.get('colsample_bytree', 0.8))
model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)


with open("stress_model_scaler.pkl", "wb") as f:
    pickle.dump(scaler_local, f)

# future:
#with open("stress_model_scaler.pkl", "rb") as f:
#    scaler = pickle.load(f)
#model = pickle.load(open("stress_model.xgb", "rb"))
#
#new_df = pd.read_csv("Dataset_Schizophrenia_Rest.csv")  # raw PSDs
#X_new = scaler.transform(new_df.iloc[:, :75].values)
#y_pred = model.predict(X_new)


pickle.dump(model, open("stress_model.xgb", "wb"))
print("Saved stress_model.xgb (model trained from train/test split)")

####################################################################################################
# LOSO validation (subject-level)
####################################################################################################
subjects = np.unique(dataset["Subject"].values)
results = pd.DataFrame(columns=["Subject", "Y", "YHAT"])

for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject, :].reset_index(drop=True)
    val_subset = dataset.loc[dataset["Subject"] == subject, :].reset_index(drop=True)
    X_train = train_subset.iloc[:, :75].values
    y_train = train_subset.iloc[:, 75:76]["Condition"].values.ravel()
    X_val = val_subset.iloc[:, :75].values
    y_true_subject = np.unique(val_subset.iloc[:, 75:76]["Condition"].values)[0]
    sc = MinMaxScaler()
    X_train_scaled = sc.fit_transform(X_train)
    X_val_scaled = sc.transform(X_val)
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss',
                        n_estimators=200, max_depth=4, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8)
    clf.fit(X_train_scaled, y_train, verbose=False)
    y_val_epoch_preds = clf.predict(X_val_scaled)
    yhat_subject = int(np.round(np.mean(y_val_epoch_preds)))
    new_index = len(results)
    results.loc[new_index, "Subject"] = subject
    results.loc[new_index, "Y"] = int(y_true_subject)
    results.loc[new_index, "YHAT"] = int(yhat_subject)

results["Y"] = results["Y"].astype(int)
results["YHAT"] = results["YHAT"].astype(int)

####################################################################################################
# Compute LOSO metrics (subject-level) + bootstrap CIs + permutation test
####################################################################################################
y_true = results["Y"].values
y_pred = results["YHAT"].values
acc_obs = accuracy_score(y_true, y_pred)
bal_acc_obs = balanced_accuracy_score(y_true, y_pred)
prec_obs = precision_score(y_true, y_pred, zero_division=0)
rec_obs = recall_score(y_true, y_pred, zero_division=0)
f1_obs = f1_score(y_true, y_pred, zero_division=0)

print("LOSO subject-level metrics:")
print(f"Accuracy: {acc_obs:.4f}")
print(f"Balanced Accuracy: {bal_acc_obs:.4f}")
print(f"Precision: {prec_obs:.4f}")
print(f"Recall: {rec_obs:.4f}")
print(f"F1: {f1_obs:.4f}")

n_boot = 1000
acc_ci_low, acc_ci_high, acc_boot = bootstrap_metric(y_true, y_pred, accuracy_score, n_bootstraps=n_boot)
bal_acc_ci_low, bal_acc_ci_high, bal_acc_boot = bootstrap_metric(y_true, y_pred, balanced_accuracy_score, n_bootstraps=n_boot)
prec_ci_low, prec_ci_high, prec_boot = bootstrap_metric(y_true, y_pred, lambda a, b: precision_score(a, b, zero_division=0), n_bootstraps=n_boot)
rec_ci_low, rec_ci_high, rec_boot = bootstrap_metric(y_true, y_pred, lambda a, b: recall_score(a, b, zero_division=0), n_bootstraps=n_boot)
f1_ci_low, f1_ci_high, f1_boot = bootstrap_metric(y_true, y_pred, lambda a, b: f1_score(a, b, zero_division=0), n_bootstraps=n_boot)
print("Bootstrap 95% CIs (percentile):")
print(f"Accuracy CI: [{acc_ci_low:.4f}, {acc_ci_high:.4f}]")
print(f"Balanced Accuracy CI: [{bal_acc_ci_low:.4f}, {bal_acc_ci_high:.4f}]")
print(f"Precision CI: [{prec_ci_low:.4f}, {prec_ci_high:.4f}]")
print(f"Recall CI: [{rec_ci_low:.4f}, {rec_ci_high:.4f}]")
print(f"F1 CI: [{f1_ci_low:.4f}, {f1_ci_high:.4f}]")

n_perm = 1000
pval_acc = permutation_test(y_true, y_pred, accuracy_score, n_permutations=n_perm)
print(f"Permutation test (n={n_perm}) p-value for accuracy: {pval_acc:.4f}")
