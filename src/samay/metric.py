import numpy as np
from typing import Literal


def MSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Mean squared error"""
    return np.mean((y_true - y_pred) ** 2)


def MAE(y_true: np.ndarray, y_pred: np.ndarray):
    """Mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))


def MASE(
    context: np.ndarray,   # (W, S, Lc)
    y_true: np.ndarray,    # (W, S, H)
    y_pred: np.ndarray,    # (W, S, H)
    reduce: Literal["none", "series", "window", "mean"] = "mean",
) -> np.ndarray | float:
    context = np.asarray(context, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true {y_true.shape} != y_pred {y_pred.shape}")
    if context.ndim != 3 or y_true.ndim != 3:
        raise ValueError("all inputs must be 3D arrays: (num_windows, num_series, seq_len)")
    if context.shape[:2] != y_true.shape[:2]:
        raise ValueError("context and y_true/y_pred must match on (W, S)")
    if context.shape[-1] <= 1:
        base = np.full(y_true.shape[:2], np.nan)  # (W, S)
        return _reduce_mase(base, reduce)

    diffs = np.abs(context[..., 1:] - context[..., :-1])   # (W, S, Lc-1)
    denom = diffs.mean(axis=-1)                            # (W, S)

    num = np.abs(y_true - y_pred).mean(axis=-1)            # (W, S)

    with np.errstate(divide="ignore", invalid="ignore"):
        mase_ws = num / denom
        mase_ws = np.where(denom > 0, mase_ws, np.nan)     # (W, S)

    return _reduce_mase(mase_ws, reduce)


def _reduce_mase(mase_ws: np.ndarray, reduce: str):
    if reduce == "none":
        return mase_ws
    if reduce == "series":
        return np.nanmean(mase_ws, axis=0)  # -> (S,)
    if reduce == "window":
        return np.nanmean(mase_ws, axis=1)  # -> (W,)
    if reduce == "mean":
        return float(np.nanmean(mase_ws))   # -> scalar
    raise ValueError(f"unknown reduce={reduce!r}")


def MAPE(y_true: np.ndarray, y_pred: np.ndarray):
    """Mean absolute percentage error"""
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error"""
    return np.sqrt(MSE(y_true, y_pred))


def NRMSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized root mean squared error"""
    return RMSE(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + 1e-5)


def SMAPE(y_true: np.ndarray, y_pred: np.ndarray):
    """Symmetric mean absolute percentage error"""
    return np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )


def MSIS(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05):
    """Mean scaled interval score"""
    q1 = np.percentile(y_true, 100 * alpha / 2)
    q2 = np.percentile(y_true, 100 * (1 - alpha / 2))
    denominator = q2 - q1
    penalties = 2 * ((y_true < q1) * (q1 - y_pred) + (y_true > q2) * (y_pred - q2))
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-5)) + np.mean(
        penalties / (denominator + 1e-5)
    )


def ND(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized deviation"""
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-5)


def MWSQ(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray):
    """
    Mean weighted squared quantile loss
    y_true: (num_seq, n_var, seq_len)
    y_pred: (q, num_seq, n_var, seq_len)
    quantiles: (q, )
    """

    y_true = np.expand_dims(y_true, axis=0)  # (1, num_seq, n_var, seq_len)
    diff = y_true - y_pred  # (q, num_seq, n_var, seq_len)
    quantiles = np.expand_dims(
        np.expand_dims(np.expand_dims(quantiles, axis=-1), axis=-1), axis=-1
    )  # (q, 1, 1, 1)
    pinball = np.maximum(quantiles * diff, (quantiles - 1) * diff)  # (num_seq, n_var, seq_len, q)
    mwsq = np.mean(pinball ** 2)  # (num_seq, n_var, seq_len)
    return mwsq


def CRPS(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray):
    """
    Continuous ranked probability score
    y_true: (num_seq, n_var, seq_len)
    y_pred: (q, num_seq, n_var, seq_len)
    quantiles: (q, )
    """
    y_true = np.expand_dims(y_true, axis=0)  # (1, num_seq, n_var, seq_len)
    diff = y_true - y_pred  # (q, num_seq, n_var, seq_len)
    quantiles = np.expand_dims(
        np.expand_dims(np.expand_dims(quantiles, axis=-1), axis=-1), axis=-1
    )  # (q, 1, 1, 1)
    pinball = np.maximum(quantiles * diff, (quantiles - 1) * diff)  # (num_seq, n_var, seq_len, q)
    return np.mean(pinball)
