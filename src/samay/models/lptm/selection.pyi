from typing import Tuple

import numpy as np
import numpy.typing as npt

def pad_times(
    scores: npt.NDArray[np.float32], times: npt.NDArray[np.int32]
) -> npt.NDArray[np.float32]:
    """
    Pads the scores array with negative infinity based on the times array.

    Parameters:
    scores (np.ndarray): A 3D array of shape (batch_size, seq_len, seq_len) containing scores.
    times (np.ndarray): A 1D array of shape (batch_size,) containing time indices.

    Returns:
    np.ndarray: A 3D array of shape (batch_size, seq_len, seq_len) with padded scores.
    """
    ...

def prune_segments(
    segment_idxs: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float32],
    num_segments: int,
) -> Tuple[npt.NDArray[np.int32], npt.NDArray[np.float32]]:
    """
    Prunes the segments to keep only the top scoring ones.

    Parameters:
    segment_idxs (np.ndarray): A 3D array of shape (batch_size, seq_len, 2) containing segment indices.
    scores (np.ndarray): A 2D array of shape (batch_size, seq_len) containing scores.
    num_segments (int): The number of segments to keep.

    Returns:
    Tuple[np.ndarray, np.ndarray]: A tuple containing:
        - A 3D array of shape (batch_size, num_segments, 2) with pruned segment indices.
        - A 2D array of shape (batch_size, num_segments) with pruned scores.
    """
    ...
