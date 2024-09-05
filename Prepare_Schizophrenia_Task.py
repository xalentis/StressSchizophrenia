# Generate CSV PSD's for Healthy Controls and Schizophrenia at Task dataset

import os
import numpy as np
import pandas as pd
import mne
import warnings
from sklearn.preprocessing import MinMaxScaler

warnings.simplefilter(action="ignore", category=FutureWarning)

np.random.seed(42)
dataset = pd.DataFrame()
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

if os.path.exists("Dataset_Schizophrenia_Task.csv"):
    dataset = pd.read_csv("Dataset_Schizophrenia_Task.csv")
else:
    ##################################################################################################################################################################
    # Load and process schizophrenia task dataset
    ##################################################################################################################################################################
    subjects = ["sub-S25", "sub-S26", "sub-S28", "sub-S29", "sub-S30", "sub-S31", "sub-S32", "sub-S33", "sub-S34", "sub-S35", "sub-S37", "sub-S38", "sub-S40", \
                "sub-S41", "sub-S42", "sub-S43", "sub-S44"]
    for subject in subjects:
        raw = mne.io.read_raw_eeglab(os.path.join("Data", "Schizophrenia2", subject, "eeg", subject + "_task-rdk_eeg.set"))
        raw.load_data()
        raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
        montage = mne.channels.make_standard_montage("standard_1020")
        raw.set_montage(montage = montage, on_missing="ignore")
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
        subset["Subject"] = subject.lower().replace("sub-","Schizophrenia2_")
        dataset = pd.concat([dataset, subset], axis=0 ,ignore_index = True)

    ##################################################################################################################################################################
    # Load and process healthy controls
    ##################################################################################################################################################################
    subjects = ["sub-S01", "sub-S02", "sub-S03", "sub-S04", "sub-S05", "sub-S06", "sub-S07", "sub-S08", "sub-S09", "sub-S10", "sub-S11", "sub-S12", "sub-S13", \
                "sub-S14", "sub-S15", "sub-S16", "sub-S17", "sub-S18", "sub-S19", "sub-S20", "sub-S21", "sub-S22", "sub-S23"]
    for subject in subjects:
        raw = mne.io.read_raw_eeglab(os.path.join("Data", "Schizophrenia2", subject, "eeg", subject + "_task-rdk_eeg.set"))
        raw.load_data()
        if "F7-1" in raw.ch_names:
            if not "F7" in raw.ch_names:
                raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2', 'F7-1': 'F7'})
        else:
            raw.rename_channels({'FP1': 'Fp1', 'FP2': 'Fp2'})
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
        subset["Subject"] = subject.lower().replace("sub-","Healthy2_")
        dataset = pd.concat([dataset, subset], axis=0 ,ignore_index = True)

    dataset = normalize(dataset)
    dataset.to_csv("Dataset_Schizophrenia_Task.csv", index = False)