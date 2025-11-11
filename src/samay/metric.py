import numpy as np
from typing import Literal


def MSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Mean squared error.

    Args:
        y_true (np.ndarray): Ground-truth array of shape (..., seq_len).
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Mean squared error between ``y_true`` and ``y_pred``.
    """
    return np.mean((y_true - y_pred) ** 2)


def MAE(y_true: np.ndarray, y_pred: np.ndarray):
    """Mean absolute error.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Mean absolute error between ``y_true`` and ``y_pred``.
    """
    return np.mean(np.abs(y_true - y_pred))

def MASE(
    context: np.ndarray,   # (W, S, Lc)
    y_true: np.ndarray,    # (W, S, H)
    y_pred: np.ndarray,    # (W, S, H)
    reduce: Literal["none", "series", "window", "mean"] = "mean",
) -> np.ndarray | float:
  """Mean absolute scaled error (MASE).

    MASE scales the absolute errors by the average in-sample one-step
    naive forecast error. This implementation approximates the scaling by
    using first differences along the sequence dimension.

    Args:
        y_true (np.ndarray): Ground-truth array. Shape can be either
            ``(num_seq, seq_len)`` or ``(batch, num_seq, seq_len)``.
        y_pred (np.ndarray): Predicted array with the same shape as
            ``y_true``.
        freq (str): Frequency string used to derive seasonality if needed.
            Currently provided for compatibility; default is ``"h"``.

    Returns:
        (float): The mean absolute scaled error.
    """
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
    """Mean absolute percentage error.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Mean absolute percentage error. A small epsilon is added to
            the denominator to avoid division by zero.
    """
    return np.mean(np.abs(y_true - y_pred) / (y_true + 1e-5))


def RMSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Root mean squared error.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Root mean squared error.
    """
    return np.sqrt(MSE(y_true, y_pred))


def NRMSE(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized root mean squared error.

    Normalizes RMSE by the range of the true values.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Normalized RMSE.
    """
    return RMSE(y_true, y_pred) / (np.max(y_true) - np.min(y_true) + 1e-5)


def SMAPE(y_true: np.ndarray, y_pred: np.ndarray):
    """Symmetric mean absolute percentage error.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): SMAPE value.
    """
    return np.mean(
        2.0 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )


def MSIS(y_true: np.ndarray, y_pred: np.ndarray, alpha: float = 0.05):
    """Mean scaled interval score (MSIS).

    Computes a simple interval scoring metric using empirical percentiles of
    the ground-truth data. This is a lightweight approximation useful for
    quick evaluation.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted values or interval endpoints.
        alpha (float): Significance level for the central prediction interval
            (default ``0.05`` corresponds to a 95% interval).

    Returns:
        (float): MSIS score.
    """
    q1 = np.percentile(y_true, 100 * alpha / 2)
    q2 = np.percentile(y_true, 100 * (1 - alpha / 2))
    denominator = q2 - q1
    penalties = 2 * ((y_true < q1) * (q1 - y_pred) + (y_true > q2) * (y_pred - q2))
    return np.mean(np.abs(y_true - y_pred) / (denominator + 1e-5)) + np.mean(
        penalties / (denominator + 1e-5)
    )


def ND(y_true: np.ndarray, y_pred: np.ndarray):
    """Normalized deviation.

    Args:
        y_true (np.ndarray): Ground-truth array.
        y_pred (np.ndarray): Predicted array with the same shape as ``y_true``.

    Returns:
        (float): Normalized deviation.
    """
    return np.mean(np.abs(y_true - y_pred)) / (np.mean(y_true) + 1e-5)


def MWSQ(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray):
    """Mean weighted squared quantile loss.

    This function computes a squared pinball loss across quantile forecasts.

    Args:
        y_true (np.ndarray): Ground-truth array with shape
            ``(num_seq, n_var, seq_len)``.
        y_pred (np.ndarray): Predicted quantiles with shape
            ``(q, num_seq, n_var, seq_len)`` where ``q`` is the number of
            quantiles.
        quantiles (np.ndarray): Array of quantile levels with shape ``(q,)``.

    Returns:
        (float): Mean squared pinball loss across quantiles and sequences.
    """

    y_true = np.expand_dims(y_true, axis=0)  # (1, num_seq, n_var, seq_len)
    diff = y_true - y_pred  # (q, num_seq, n_var, seq_len)
    quantiles = np.expand_dims(
        np.expand_dims(np.expand_dims(quantiles, axis=-1), axis=-1), axis=-1
    )  # (q, 1, 1, 1)
    pinball = np.maximum(quantiles * diff, (quantiles - 1) * diff)
    mwsq = np.mean(pinball ** 2)
    return mwsq


def CRPS(y_true: np.ndarray, y_pred: np.ndarray, quantiles: np.ndarray):
    """Continuous Ranked Probability Score (CRPS) using discrete quantiles.

    This implementation approximates CRPS by averaging the (non-squared)
    pinball loss across quantile levels.

    Args:
        y_true (np.ndarray): Ground-truth array with shape
            ``(num_seq, n_var, seq_len)``.
        y_pred (np.ndarray): Predicted quantiles with shape
            ``(q, num_seq, n_var, seq_len)``.
        quantiles (np.ndarray): Array of quantile levels with shape ``(q,)``.

    Returns:
        (float): Approximated CRPS (mean pinball loss over quantiles).
    """
    y_true = np.expand_dims(y_true, axis=0)  # (1, num_seq, n_var, seq_len)
    diff = y_true - y_pred  # (q, num_seq, n_var, seq_len)
    quantiles = np.expand_dims(
        np.expand_dims(np.expand_dims(quantiles, axis=-1), axis=-1), axis=-1
    )  # (q, 1, 1, 1)
    pinball = np.maximum(quantiles * diff, (quantiles - 1) * diff)  # (num_seq, n_var, seq_len, q)
    return np.mean(pinball)
