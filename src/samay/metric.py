import numpy as np


def MSE(y_true:np.array, y_pred:np.array):
    """Mean squared error"""
    return np.mean((y_true - y_pred) ** 2)


def MAE(y_true:np.array, y_pred:np.array):
    """Mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))


def MASE(y_true:np.array, y_pred:np.array, freq:str='h'):
    """Mean absolute scaled error"""
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
    seasonality = DEFAULT_SEASONALITIES[freq]
    y_t = y_true[seasonality:] - y_true[:-seasonality]
    return np.mean(np.abs(y_true - y_pred) / (np.mean(np.abs(y_t)) + 1e-5))


def MAPE(y_true:np.array, y_pred:np.array):
    """Mean absolute percentage error"""
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5))


def RMSE(y_true:np.array, y_pred:np.array):
    """Root mean squared error"""
    return np.sqrt(MSE(y_true, y_pred))


def NRMSE(y_true:np.array, y_pred:np.array):
    """Normalized root mean squared error"""
    return RMSE(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + 1e-5)


def SMAPE(y_true:np.array, y_pred:np.array):
    """Symmetric mean absolute percentage error"""
    return np.mean(2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-5))


def MSIS(y_true:np.array, y_pred:np.array, alpha:float=0.05):
    """Mean scaled interval score"""
    q1 = np.percentile(y_true, 100 * alpha / 2)
    q2 = np.percentile(y_true, 100 * (1 - alpha / 2))
    denominator = q2 - q1
    penalties = 2 * ((y_true < q1) * (q1 - y_pred) + (y_true > q2) * (y_pred - q2))
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-5)) + np.mean(penalties / (denominator + 1e-5))


def ND(y_true:np.array, y_pred:np.array):
    """Normalized deviation"""
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-5)


def MWSQ(y_true:np.array, y_pred:np.array, quantiles:np.array):
    """Mean weighted squared quantile loss"""
    def quantile_loss(y_true, y_pred, q):
        return np.maximum(q * (y_true - y_pred), (q - 1) * (y_true - y_pred)).mean()
    return np.mean([quantile_loss(y_true, y_pred, q) for q in quantiles])
    

def CRPS(y_true:np.array, y_pred:np.array, quantiles:np.array):
    """Continuous ranked probability score"""
    crps = np.mean((y_pred - y_true) ** 2 * np.abs(quantiles - (y_true <= y_pred).astype(float)))
    return crps