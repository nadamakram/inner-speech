import numpy as np
from scipy.signal import butter, filtfilt, medfilt
from scipy.ndimage import median_filter
from sklearn.preprocessing import RobustScaler

# Sampling rate
sampling_rate = 512  

# Bandpass Filter (8-50 Hz)
def bandpass_filter(data, low_freq=8, high_freq=50, fs=sampling_rate, order=4):
    """
    Apply a bandpass filter to the X.
    Args:
        X (np.ndarray): Input data array (samples, channels).
        low_freq (float): Lower frequency bound.
        high_freq (float): Upper frequency bound.
        fs (int): Sampling frequency.
        order (int): Filter order.
    Returns:
        np.ndarray: Filtered data.
    """
    nyquist = 0.5 * fs
    low = low_freq / nyquist
    high = high_freq / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# Median Filter
def apply_median_filter(data, size=3):
    """
    Apply a median filter to the data.
    Args:
        data (np.ndarray): Input data array (samples, channels).
        size (int): Size of the median filter.
    Returns:
        np.ndarray: Median-filtered data.
    """
    return median_filter(data, size=size)

# Common Average Referencing (CAR)
def common_average_reference(data):
    """
    Apply common average referencing (CAR) to the data.
    Args:
        data (np.ndarray): Input data array (samples, channels).
    Returns:
        np.ndarray: Data after CAR.
    """
    car_data = data - np.mean(data, axis=0, keepdims=True)

    return car_data


def preprocess_data(X):
    """
    Full preprocessing pipeline: Bandpass filter, median filter, CAR, and RobustScaler.
    Args:
        X (np.ndarray): Input data array (samples, channels).
    Returns:
        np.ndarray: Preprocessed data.
    """
    # Apply bandpass filter
    filtered_data = bandpass_filter(X)

    # Apply median filter
    filtered_data = apply_median_filter(filtered_data)

    # Apply CAR
    car_data = common_average_reference(filtered_data)

    # Reshape for scaling (samples, channels)
    reshaped_data = car_data.reshape(-1, car_data.shape[1])

    # RobustScaler Instance
    scaler = RobustScaler()

    # Apply RobustScaler
    scaled_data = scaler.fit_transform(reshaped_data)

    # Reshape back to original dimensions
    X_preprocessed = scaled_data.reshape(car_data.shape)

    return X_preprocessed


# Example usage in a notebook
# sys.path.append('../src')  
# from preprocessing import preprocess_data
# X_processed =  preprocess_data(X)