from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test, X_train, y_train):
    """
    Evaluate the given model on test and train datasets.

    Parameters:
        model (object): The trained model with a predict method.
        X_test (array-like): Test data features.
        y_test (array-like): Test data labels.
        X_train (array-like): Training data features.
        y_train (array-like): Training data labels.

    Returns:
        dict: Evaluation results containing accuracy and classification reports.
    """
    evaluation_results = {}

    # Evaluate on test set
    y_pred_test = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_report = classification_report(y_test, y_pred_test)
    evaluation_results['test_accuracy'] = test_accuracy
    evaluation_results['test_report'] = test_report

    print(f"Test Accuracy: {test_accuracy}")
    print("Classification Report Test:\n", test_report)

    # Evaluate on training set
    y_pred_train = model.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_report = classification_report(y_train, y_pred_train)
    evaluation_results['train_accuracy'] = train_accuracy
    evaluation_results['train_report'] = train_report

    print(f"Train Accuracy: {train_accuracy}")
    print("Classification Report Train:\n", train_report)

    return evaluation_results
