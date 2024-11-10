from scipy.stats import kurtosis, skew
from scipy.signal import welch
import numpy as np

# Time domain features
def mean_voltage(data):
    return np.mean(data, axis=0)

def root_mean_square(data):
    return np.sqrt(np.mean(data**2, axis=0))

def zero_crossing_rate(data):
    return np.sum(np.diff(np.sign(data), axis=0) != 0, axis=0)

def signal_variability(data):
    return np.std(data, axis=0)

# Frequency domain features
def power_spectral_density(data, fs=512):
    f, psd = welch(data, fs=fs, nperseg=min(256, len(data)), axis=0)
    return np.mean(psd, axis=0)

def peak_frequency(data, fs=512):
    f, psd = welch(data, fs=fs, nperseg=min(256, len(data)), axis=0)
    return f[np.argmax(psd, axis=0)]

def spectral_entropy(data, fs=512):
    f, psd = welch(data, fs=fs, nperseg=min(256, len(data)), axis=0)
    psd_norm = psd / np.sum(psd, axis=0, keepdims=True)
    se = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=0)
    return se

# Existing feature extraction functions
def hjorth_activity(data):
    return np.var(data, axis=0)

def hjorth_mobility(data):
    return np.sqrt(np.var(np.diff(data, axis=0), axis=0) / np.var(data, axis=0))

def hjorth_complexity(data):
    return hjorth_mobility(np.diff(data, axis=0)) / hjorth_mobility(data)

def tsallis_entropy(data, q=2):
    prob_data = np.histogram(data, bins=256, density=True)[0]
    return (1 - np.sum(prob_data ** q)) / (q - 1)

def shannon_entropy(data):
    prob_data = np.histogram(data, bins=256, density=True)[0]
    return -np.sum(prob_data * np.log(prob_data + 1e-12))

# Function to extract features from selected EEG channels
def extract_features(data):
    num_trials = data.shape[2]
    num_channels = data.shape[1]  # Number of selected channels (8 in this case)

    feature_lists_time = []
    feature_lists_freq = []

    for trial in range(num_trials):
        trial_data = data[:, :, trial]
        trial_features_time = []
        trial_features_freq = []

        # Time domain features extraction
        for channel in range(num_channels):
            channel_data = trial_data[:, channel]
            mean_val = np.mean(channel_data)  # Calculate mean once for reuse
            channel_features_time = [
                np.std(channel_data),
                np.median(channel_data),
                kurtosis(channel_data),
                np.sqrt(np.mean(channel_data**2)),
                skew(channel_data),
                np.mean(np.abs(channel_data - mean_val)),
                zero_crossing_rate(channel_data),
                hjorth_activity(channel_data),
                hjorth_mobility(channel_data),
                hjorth_complexity(channel_data),
                np.mean(channel_data**2),
                np.mean(channel_data**2 - np.roll(channel_data, 1) * np.roll(channel_data, -1)),
                np.log(np.sqrt(np.sum(np.diff(channel_data)**2))),
                tsallis_entropy(channel_data, q=2),
            ]
            trial_features_time.extend(channel_features_time)

        # Frequency domain features extraction
        for channel in range(num_channels):
            channel_data = trial_data[:, channel]
            channel_features_freq = [
                power_spectral_density(channel_data),
                peak_frequency(channel_data),
                spectral_entropy(channel_data)
            ]
            trial_features_freq.extend(channel_features_freq)

        feature_lists_time.append(trial_features_time)
        feature_lists_freq.append(trial_features_freq)

    # Convert lists to numpy arrays
    feature_matrix_time = np.array(feature_lists_time)
    feature_matrix_freq = np.array(feature_lists_freq)

    return feature_matrix_time, feature_matrix_freq