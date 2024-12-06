import einops
import numpy as np
import numpy.typing as npt


def pad_times(scores: npt.NDArray[np.float32], times: npt.NDArray[np.int32]):
    """Pad the scores with -inf

    Args:
        scores (npt.NDArray[np.float32]): scores of the segments [batch, seq_len, seq_len]
        time (int): number of time steps to pad

    Returns:
        npt.NDArray[np.float32]: padded scores [batch, seq_len + time, seq_len + time]
    """
    scores_cp = np.copy(scores)
    for i, t in enumerate(times):
        scores_cp[i, t:, :] = -np.inf
        scores_cp[i, :, t:] = -np.inf
    return scores_cp


def select_highest_suffix(scores: npt.NDArray[np.float32]):
    """Select the highest scoring segments

    Args:
        scores (npt.NDArray[np.float32]): scores of the segments [batch, seq_len, seq_len]

    Returns:
        npt.NDArray[np.int32]: selected segments start and end indices [batch, num_segments, 2]
        npt.NDArray[np.float32]: selected segments scores [batch, num_segments]
    """
    scores_cp = np.copy(scores)
    # Make all values of the diagonal and below -inf
    mask = np.tri(scores.shape[1], dtype=bool)
    mask = einops.repeat(mask, "n m -> b n m", b=scores.shape[0])
    scores_cp[mask] = -np.inf
    first_idx = np.arange(scores.shape[1] - 1)
    first_idx = einops.repeat(
        first_idx, "n -> b n", b=scores.shape[0]
    )  # [batch, seq_len - 1]
    second_idx = scores_cp.argmax(axis=2)[:, :-1]  # [batch, seq_len - 1]
    idxs = np.stack([first_idx, second_idx], axis=2)  # [batch, seq_len -1, 2]
    batch_num = np.arange(scores.shape[0])
    batch_num = einops.repeat(
        batch_num, "n -> n b", b=scores.shape[1] - 1
    )  # [batch, seq_len - 1]
    idx_with_batch = np.stack(
        [batch_num, first_idx, second_idx], axis=2
    )  # [batch, seq_len - 1, 3]

    # Get the scores
    scores_cp = scores_cp[
        idx_with_batch[:, :, 0], idx_with_batch[:, :, 1], idx_with_batch[:, :, 2]
    ]
    return idxs, scores_cp


def prune_segments(
    segment_idxs: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float32],
    num_segments: int,
):
    """Prune the segments from lowest to highest such that segemnts cover the entire sequence.
    Choose the lower scoring segment. If the left over segments are not enough to cover the sequence,
    end the loop and return the selected segments. Else, remove the lowest scoring segment and repeat.


    Args:
        segment_idxs (npt.NDArray[np.int32]): selected segments start and end indices [num_segments, 2]
        scores (npt.NDArray[np.float32]): selected segments scores [num_segments]

    Returns:
        npt.NDArray[np.int32]: selected segments start and end indices [num_selected_segments, 2]
        npt.NDArray[np.float32]: selected segments scores [num_selected_segments]
    """
    # Sort the segments based on scores in ascending order
    sorted_idxs = np.argsort(scores)
    selected_idxs = []
    covered = np.zeros(segment_idxs.shape[0], dtype=bool)

    for idx in reversed(sorted_idxs):
        if len(selected_idxs) == num_segments or all(covered):
            break
        start, end = segment_idxs[idx]
        covered[start:end] = True

    return segment_idxs[selected_idxs], scores[selected_idxs]


def prune_segments_all(
    segment_idxs: npt.NDArray[np.int32],
    scores: npt.NDArray[np.float32],
    num_segments: int,
):
    """Prune the segments from lowest to highest such that segemnts cover the entire sequence.
    Choose the lower scoring segment. If the left over segments are not enough to cover the sequence,
    end the loop and return the selected segments. Else, remove the lowest scoring segment and repeat.


    Args:
        segment_idxs (npt.NDArray[np.int32]): selected segments start and end indices [batch, num_segments, 2]
        scores (npt.NDArray[np.float32]): selected segments scores [batch, num_segments]

    Returns:
        list[npt.NDArray[np.int32]]: selected segments start and end indices [num_selected_segments, 2]
        npt.NDArray[np.float32]: selected segments scores [num_selected_segments]
    """
    segments, out_scores = [], []
    for idx, score in zip(segment_idxs, scores):
        seg, score = prune_segments(idx, score, num_segments)
        segments.append(seg)
        out_scores.append(score)

    return segments, out_scores


def select_segments(scores: npt.NDArray[np.float32], num_segments: int):
    """Select the highest scoring segments

    Args:
        scores (npt.NDArray[np.float32]): scores of the segments [batch, seq_len, seq_len]
        num_segments (int): number of segments to select

    Returns:
        npt.NDArray[np.int32]: selected segments start and end indices [batch, num_segments, 2]
        npt.NDArray[np.float32]: selected segments scores [batch, num_segments]
    """
    idxs, scores = select_highest_suffix(scores)
    return prune_segments_all(idxs, scores, num_segments)
