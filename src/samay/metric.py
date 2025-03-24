import numpy as np


def MSE(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def MAE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def MASE(y_true, y_pred, freq='h'):
    DEFAULT_SEASONALITIES = {
        "S": 3600,  # 1 hour
        "s": 3600,  # 1 hour
        "T": 1440,  # 1 day
        "min": 1440,  # 1 day
        "H": 24,  # 1 day
        "h": 24,  # 1 day
        "D": 1,  # 1 day
        "W": 1,  # 1 week
        "M": 12,
        "ME": 12,
        "B": 5,
        "Q": 4,
        "QE": 4,
    }
    # seasonality = DEFAULT_SEASONALITIES[freq]
    y_t = y_true[1:] - y_true[:-1]
    return np.mean(np.abs(y_true - y_pred) / (np.mean(np.abs(y_t)) + 1e-5))


def MAPE(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5))


def RMSE(y_true, y_pred):
    return np.sqrt(MSE(y_true, y_pred))


def NRMSE(y_true, y_pred):
    return RMSE(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + 1e-5)


def SMAPE(y_true, y_pred):
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-5))


def MSIS(y_true, y_pred, alpha=0.05):
    q1 = np.percentile(y_true, 100 * alpha / 2)
    q2 = np.percentile(y_true, 100 * (1 - alpha / 2))
    denominator = q2 - q1
    penalties = 2 * ((y_true < q1) * (q1 - y_pred) + (y_true > q2) * (y_pred - q2))
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-5)) + np.mean(penalties / (denominator + 1e-5))


def ND(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-5)


def MWSQ(y_true, y_pred, quantiles):
    def quantile_loss(y_true, y_pred, q):
        return np.maximum(q * (y_true - y_pred), (q - 1) * (y_true - y_pred)).mean()
    return np.mean([quantile_loss(y_true, y_pred, q) for q in quantiles])
    

def CRPS(y_true, y_pred, quantiles):
    crps = np.mean((y_pred - y_true) ** 2 * np.abs(quantiles - (y_true <= y_pred).astype(float)))
    return crps