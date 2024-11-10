# Optimizing Inner Speech Decoding from EEG Signals through Feature Selection and Model Interpretability 
This project focuses on decoding inner speech from EEG signals, aiming to develop communication aids for individuals with motor impairments. It explores signal preprocessing, electrode reduction, and machine learning (ML) and deep learning (DL) methods for accurate classification. The study uses feature extraction in time and frequency domains, applying ML models and ensemble techniques, as well as CNNs and BiLSTMs for DL. Saliency mapping is employed for model interpretability. Findings highlight the potential of using reduced electrode sets for efficient and portable EEG-based communication devices.
# Setup 
1. Clone the repository

```
git clone https://github.com/nadamakram/inner-speech.git
cd inner-speech
```

2. Install dependencies
```
python3 -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
pip install -r requirements.txt
```

# Repo Structure

summary of the directory structure for the project:

`notebooks/deep_learning`: Contains notebook experiments related to deep learning approaches for inner speech decoding.

`notebooks/machine_learning`: Contains notebook experiments focusing on machine learning approaches for decoding inner speech.

`src`: Includes code functions for loading data, preprocessing, feature extraction, and model evaluation.
