# Generate CSV PSD's for Healthy Controls and Schizophrenia at Rest dataset, split by healthy and schizophrenia

import os
import numpy as np
import pandas as pd
import mne
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
scaler = MinMaxScaler()

##################################################################################################################################################################
# EEG helper routines and parameters
##################################################################################################################################################################
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


##################################################################################################################################################################
# Load and process schizophrenia rest dataset
##################################################################################################################################################################
dataset = pd.DataFrame()
subjects = ["s01", "s02", "s03", "s04", "s05", "s06","s07", "s08", "s09", "s10", "s11", "s12", "s13", "s14"]
for subject in subjects:
    raw = mne.io.read_raw_edf(os.path.join("Data", "Schizophrenia", subject + ".edf"), infer_types = True)
    raw.load_data()
    raw.set_montage("standard_1020")   
    raw.set_eeg_reference(ref_channels = "average")
    raw = raw.copy().pick_channels(channels)
    eeg_index = raw.pick_types(meg=False, eeg=True, eog=False)
    data_filtered = raw.copy().notch_filter(freqs = (25, 50), picks = channels)
    data_filtered = data_filtered.copy().filter(0.5, 45) # bandbass filter
    data_filtered = data_filtered.copy().resample(sfreq = 200)
    # epoch
    epoch_whole = mne.make_fixed_length_epochs(data_filtered, duration=int(np.floor(raw.tmax)), preload=True)[0]
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
    subset["Subject"] = "Schizophrenia_" + subject
    dataset = pd.concat([dataset, subset], axis=0 ,ignore_index = True)

dataset = normalize(dataset)
dataset.to_csv("Dataset_Schizophrenia_Rest_S.csv", index = False)

##################################################################################################################################################################
# Load and process healthy controls at rest
##################################################################################################################################################################
dataset = pd.DataFrame()
subjects = ["h01", "h02", "h03", "h04", "h05", "h06", "h07", "h08", "h09", "h10", "h11", "h12", "h13", "h14"]
for subject in subjects:
    raw = mne.io.read_raw_edf(os.path.join("Data", "Schizophrenia", subject + ".edf"), infer_types = True)
    raw.load_data()
    raw.set_montage("standard_1020")   
    raw.set_eeg_reference(ref_channels = "average")
    raw = raw.copy().pick_channels(channels)
    eeg_index = raw.pick_types(meg=False, eeg=True, eog=False)
    data_filtered = raw.copy().notch_filter(freqs = (25, 50), picks = channels)
    data_filtered = data_filtered.copy().filter(0.5, 45) # bandbass filter
    data_filtered = data_filtered.copy().resample(sfreq = 200)
    # epoch
    epoch_whole = mne.make_fixed_length_epochs(data_filtered, duration=int(np.floor(raw.tmax)), preload=True)[0]
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
    subset["Subject"] = "Healthy_" + subject
    dataset = pd.concat([dataset, subset], axis=0 ,ignore_index = True)

dataset = normalize(dataset)
dataset.to_csv("Dataset_Schizophrenia_Rest_H.csv", index = False)
