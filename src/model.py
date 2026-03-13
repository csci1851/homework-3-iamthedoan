from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score
import numpy as np

def svm_classifier(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale", probability=True):
    """
    TODO: Return a scikit-learn SVC model with the specified parameters.
    """
    return SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True)



def svm_regressor(kernel: str = "linear", C: float = 1.0, degree: int = 3, gamma: str = "scale"):
    """
    TODO: Return a scikit-learn SVR model with the specified parameters.
    """
    return SVR(C=C, kernel=kernel, degree=degree, gamma=gamma)


def evaluate_classifier(model, X_test, y_test):
    """
    TODO: Compute and return accuracy, precision, recall, and F1 score
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    auc=None
    if len(np.unique(y_test)) == 2:
        auc = roc_auc_score(y_test, y_proba)


    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": auc,
    }
    return metrics



def evaluate_regressor(model, X_test, y_test):
    """
    TODO: Compute and return MAE, RMSE, and R2
    """

    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return { "mae": mae, "rmse": float(rmse), "r2": r2}
    