import os
import numpy as np
import pandas as pd
import mne
import pickle
import warnings
import xgboost as xgb
from mne.preprocessing import ICA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
eegstress_dataset = pd.DataFrame()
sam_dataset = pd.DataFrame()
dataset = pd.DataFrame()
scaler = MinMaxScaler()

##################################################################################################################################################################
# EEG helper routines and parameters
##################################################################################################################################################################
n_components = 10 # no. of ICA components
reject = dict(mag=5e-12, grad=4000e-13) # ica reject params
channels = ["Fp1", "Fp2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7", "F8", "Fz", "Pz", "Cz"]

def normalize(dataframe):
    column_names_to_normalize = dataframe.columns[:70]
    x = dataframe[column_names_to_normalize].values
    x_scaled = scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column_names_to_normalize, index = dataframe.index)
    dataframe[column_names_to_normalize] = df_temp
    return dataframe

def psd_frontal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["Fp1", "Fp2", "F3", "F4", "F7", "F8", "Fz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "FP1", frequency + "FP2", frequency + "F3", frequency + "F4", frequency + "F7", frequency + "F8", frequency + "Fz"])
    return powers

def psd_central(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["C3", "C4", "Cz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "C3", frequency + "C4", frequency + "Cz"])
    return powers

def psd_parietal(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["P3", "P4", "Pz"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "P3", frequency + "P4", frequency + "Pz"])
    return powers

def psd_occipital(epoch_whole, frequency, low, high):
    freq_bands = {frequency: [low, high]}
    spectrum = epoch_whole.compute_psd(picks = ["O1", "O2"], fmin = 0.5, fmax = 45)
    psds, freqs = spectrum.get_data(return_freqs = True)
    X = []
    for fmin, fmax in freq_bands.values():
        psds_band = psds[:, :, (freqs >= fmin) & (freqs < fmax)].mean(axis = -1)
        X.append(psds_band.reshape(len(psds), -1))
    whole_psd = np.concatenate(X, axis = 1)
    whole_psd = np.multiply(whole_psd, 1e12)    
    powers = pd.DataFrame(whole_psd, columns = [frequency + "O1", frequency + "O2"])
    return powers

if os.path.exists("Dataset_Stress.csv"):
    dataset = pd.read_csv("Dataset_Stress.csv")
else:
    ##################################################################################################################################################################
    # Load and process EEGStress dataset
    ##################################################################################################################################################################
    subjects = list(range(0, 36))
    for subject in subjects:
        if subject < 10:  # deal with leading zero in subject
            subject = "0" + str(subject)
        else:
            subject = str(subject)
        # relax state
        raw = mne.io.read_raw_edf(os.path.join("Data", "EEGStress", "Subject" + subject + "_1.edf"), infer_types = True)
        raw.load_data()
        raw.set_channel_types(mapping={"ECG": "misc"})
        raw.set_eeg_reference(ref_channels = "average")
        raw = raw.copy().pick_channels(channels)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        eeg_index = raw.pick_types(meg=False, eeg=True, eog=False)
        data_filtered = raw.copy().notch_filter(freqs = (25, 50), picks = channels)
        data_filtered = data_filtered.copy().filter(0.5, 45) # bandbass filter
        data_filtered = data_filtered.copy().resample(sfreq = 200)
        # ICA
        ica = ICA(n_components = n_components, method = "fastica", random_state = 23) # create ICA object
        ica.fit(data_filtered, decim = 3, reject = reject)
        data_ica = ica.apply(data_filtered.copy())
        # epoch
        epoch_whole = mne.make_fixed_length_epochs(data_ica, duration=int(np.floor(raw.tmax)), preload=True)[0]
        # psd
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
        subset = pd.concat([alpha_frontal, alpha_central, alpha_parietal, alpha_occipital, \
                            beta_frontal, beta_central, beta_parietal, beta_occipital, \
                            delta_frontal, delta_central, delta_parietal, delta_occipital, \
                            theta_frontal, theta_central, theta_parietal, theta_occipital, \
                            gamma_frontal, gamma_central, gamma_parietal, gamma_occipital], axis=1)
        subset["Condition"] = 0
        subset["Subject"] = "EEGStressRelaxed_" + subject
        eegstress_dataset = pd.concat([eegstress_dataset, subset], axis=0 ,ignore_index = True)

        # stressed state
        raw = mne.io.read_raw_edf(os.path.join("Data", "EEGStress", "Subject" + subject + "_2.edf"), infer_types = True)
        raw.load_data()
        raw.set_channel_types(mapping={"ECG": "misc"})
        raw.set_eeg_reference(ref_channels = ["A2-A1"])
        raw = raw.copy().pick_channels(channels)
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage)
        eeg_index = raw.pick_types(meg=False, eeg=True, eog=False)
        data_filtered = raw.copy().notch_filter(freqs = (25, 50), picks = channels)
        data_filtered = data_filtered.copy().filter(0.5, 45) # bandbass filter
        data_filtered = data_filtered.copy().resample(sfreq = 200)
        # ICA
        ica = ICA(n_components = n_components, method = "fastica", random_state = 23) # create ICA object
        ica.fit(data_filtered, decim = 3, reject = reject)
        data_ica = ica.apply(data_filtered.copy())
        # epoch
        epoch_whole = mne.make_fixed_length_epochs(data_ica, duration=int(np.floor(raw.tmax)), preload=True)[0]
        # psd
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
        subset = pd.concat([alpha_frontal, alpha_central, alpha_parietal, alpha_occipital, \
                            beta_frontal, beta_central, beta_parietal, beta_occipital, \
                            delta_frontal, delta_central, delta_parietal, delta_occipital, \
                            theta_frontal, theta_central, theta_parietal, theta_occipital, \
                            gamma_frontal, gamma_central, gamma_parietal, gamma_occipital], axis=1)
        subset["Condition"] = 1
        subset["Subject"] = "EEGStressStressed_" + subject
        eegstress_dataset = pd.concat([eegstress_dataset, subset], axis=0 ,ignore_index = True)

    dataset = normalize(eegstress_dataset)
    dataset.to_csv("Dataset_Stress.csv", index = False)

##################################################################################################################################################################
# Modelling
##################################################################################################################################################################
X = dataset.iloc[:, :75]
y = dataset.iloc[:,75:76]["Condition"]
y = y.values

# parameter search
model = xgb.XGBRegressor()
param_grid = dict(max_depth=[2, 4, 6, 8], n_estimators=[100, 200, 300, 400, 500, 700, 900], eta=[0.1, 0.3, 0.5, 0.7], subsample=[0.3,0.5, 0.7, 0.9], colsample_bytree=[0.3,0.5,0.8,0.9])
kfold = StratifiedKFold(n_splits=4, shuffle=True, random_state=7)
grid_search = GridSearchCV(model, param_grid, n_jobs=-1, cv=kfold, verbose=1)
grid_result = grid_search.fit(X, y)
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# model training and scoring
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30)
model = xgb.XGBRegressor(n_estimators=700, max_depth=2, eta=0.1, subsample=0.9, colsample_bytree=0.3)
model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], eval_metric=["auc"], early_stopping_rounds=10)
# AUC: 1

pickle.dump(model, open("stress_model.xgb", "wb"))

##################################################################################################################################################################
# LOSO validation
##################################################################################################################################################################
subjects = np.unique(dataset.iloc[:,76:77]["Subject"].values)
results = pd.DataFrame(columns=["Subject", "Y", "YHAT"])
for subject in subjects:
    train_subset = dataset.loc[dataset["Subject"] != subject,:]
    val_subset = dataset.loc[dataset["Subject"] == subject,:]
    X = train_subset.iloc[:, :75]
    y = train_subset.iloc[:,75:76]["Condition"]
    y = y.values
    model = xgb.XGBRegressor(n_estimators=700, max_depth=2, eta=0.3, subsample=0.9, colsample_bytree=0.3)
    model.fit(X, y, verbose=False)
    X = val_subset.iloc[:, :75]
    y = np.unique(val_subset.iloc[:,75:76]["Condition"])[0]
    yhat = int(np.round(np.mean(model.predict(X))))
    newrow = {"Subject": subject, "Y": y, "YHAT": yhat} 
    results = results.append(newrow, ignore_index = True)
print(accuracy_score(results['Y'].values.astype(int), results['YHAT'].values.astype(int))) # 1
