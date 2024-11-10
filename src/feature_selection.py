
#dynamic
def apply_rfe(X, y, n_features=8, clf=None):
    """
    Apply Recursive Feature Elimination (RFE) to select top n_features.
    """
    if clf is None:
        clf = DecisionTreeClassifier(random_state=0)

    # Reshape X for RFE (samples, channels)
    X_reshaped = X.transpose(2, 0, 1).reshape(-1, X.shape[1])  
    y_reshaped = np.repeat(y, X.shape[0])  # Adjust to match trial repetition

    # Apply RFE
    rfe = RFE(estimator=clf, n_features_to_select=n_features)
    rfe.fit(X_reshaped, y_reshaped)

    selected_channel_indices = np.where(rfe.support_)[0]
    print('******** The selected channels indices are: {} **********',selected_channel_indices)
    # Select the corresponding channels and reshape for further processing
    X_selected = np.transpose(X[:, selected_channel_indices, :], (2, 0, 1))
    
    # Reshape to (trials, features)
    X_selected = X_selected.reshape(X.shape[2], -1)  # Assuming shape[2] is trials
    return X_selected, selected_channel_indices


#static
def get_subject_indices(subject_file_param):

    
    all_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8', 'A9', 'A10', 'A11', 'A12', 'A13', 'A14',
    'A15', 'A16', 'A17', 'A18', 'A19', 'A20', 'A21', 'A22', 'A23', 'A24', 'A25', 'A26', 'A27', 'A28', 'A29', 'A30', 
    'A31', 'A32', 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10', 'B11', 'B12', 'B13', 'B14', 'B15',
    'B16', 'B17', 'B18', 'B19', 'B20', 'B21', 'B22', 'B23', 'B24', 'B25', 'B26', 'B27', 'B28', 'B29', 'B30', 'B31',
    'B32', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C13', 'C14', 'C15', 'C16', 'C17', 'C18',
    'C19', 'C20', 'C21', 'C22', 'C23', 'C24', 'C25', 'C26', 'C27', 'C28', 'C29', 'C30', 'C31', 'C32', 'D1', 'D2', 'D3', 'D4', 'D5',
    'D6', 'D7', 'D8', 'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'D16', 'D17', 'D18', 'D19', 'D20', 'D21', 'D22', 'D23', 'D24', 
    'D25', 'D26', 'D27', 'D28', 'D29', 'D30', 'D31', 'D32']

    selected_channels = {
    'Subject_1.mat': ['A29', 'B10', 'B27', 'C5', 'D4', 'D9', 'D24', 'D27'],
    'Subject_2.mat': ['A1', 'A6', 'A12', 'B23', 'B27', 'C29', 'D10', 'D22'],
    'Subject_3.mat': ['A15', 'A20', 'B10', 'B16', 'C3', 'C7', 'D24', 'D32'],
    'Subject_4.mat': ['A14', 'A23', 'B7', 'B11', 'C8', 'C15', 'D6', 'D32'],
    'Subject_5.mat': ['A11', 'A25', 'A32', 'B11', 'B27', 'C18', 'D7', 'D25'],
    'Subject_6.mat': ['A12', 'A20', 'B2', 'B15', 'B24', 'B27', 'C6', 'D32'],
    'Subject_7.mat': ['A14', 'A22', 'A26', 'A29', 'B8', 'B15', 'C9', 'D32'],
    'Subject_8.mat': ['A16', 'A27', 'B6', 'B11', 'C10', 'C31', 'D19', 'D32'],
    'Subject_9.mat': ['A1', 'A20', 'B11', 'B17', 'C8', 'D6', 'D18', 'D23'],
    'Subject_10.mat': ['A3', 'A14', 'A17', 'A27', 'B25', 'C9', 'D23', 'D32']
    }

    # Retrieve selected channels for the given subject
    channels = selected_channels.get(subject_file_param)
    # Get indices of the selected channels in all_channels
    indices = [all_channels.index(channel) for channel in channels if channel in all_channels]
    return indices
