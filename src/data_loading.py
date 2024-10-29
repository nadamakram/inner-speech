import scipy.io
import numpy as np

def load_and_combine_mat_data(mat_file_paths):
    """
    Load EEG data from multiple .mat files and combine them into a single dataset.

    Args:
        mat_file_paths (list): List of file paths to .mat files.

    Returns:
        X (numpy.ndarray): Combined EEG data of shape (samples, channels, trials).
        y (numpy.ndarray): Combined target labels.
        Channels (list): EEG channel names.
    """
    # Initialize lists to hold data
    X_all = []
    Y_all = []

    # Loop through file paths and load data
    for mat_file_path in mat_file_paths:
        mat_data = scipy.io.loadmat(mat_file_path)
        data = mat_data['Data'][0, 0]
        X = data['trials']  # EEG data (samples, channels, trials)
        y = data['Labels'].flatten()  # Target labels
        Channels = [ch[0] for ch in data['Channels'][0]]  # EEG channels

        X_all.append(X)
        Y_all.append(y)

    # Concatenate data along trials axis
    X_combined = np.concatenate(X_all, axis=2)
    y_combined = np.concatenate(Y_all)

    # Print shapes to verify
    print("Combined EEG Data Shape (Samples, Channels, Trials):", X_combined.shape)
    print("Combined Labels Shape:", y_combined.shape)
    print("Channels:", Channels)
    
    return X_combined, y_combined, Channels

# Example usage in a notebook
# import sys
# sys.path.append('../src')  
# from data_loading import load_and_combine_mat_data
# mat_file_paths = ['../data/Subject_1.mat']
# X, y, Channels = load_and_combine_mat_data(mat_file_paths)
