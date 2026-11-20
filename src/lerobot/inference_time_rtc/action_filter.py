"""Chunk-local centered Savitzky-Golay smoothing for joint targets."""

import numpy as np

_KERNELS = {
    5: np.array([-3, 12, 17, 12, -3], dtype=np.float64) / 35,
    7: np.array([-2, 3, 6, 7, 6, 3, -2], dtype=np.float64) / 21,
}
ACTION_SG_WINDOWS = (0, *_KERNELS)


def filter_joint_targets(
    actions: np.ndarray, joint_columns: tuple[int, ...], window: int
) -> tuple[np.ndarray, int]:
    """Smooth supported interior rows and return the actual window length."""
    if window == 7 and len(actions) < window:
        window = 5
    if not window or not joint_columns or len(actions) < window:
        return actions.copy(), 0

    radius = window // 2
    joints = actions[:, joint_columns]
    windows = np.lib.stride_tricks.sliding_window_view(joints, window, axis=0)
    smoothed = np.clip(windows @ _KERNELS[window], windows.min(axis=-1), windows.max(axis=-1))
    # Preserve endpoints and non-position channels. Convert integer target
    # arrays to floats so smoothing does not truncate fractional targets.
    filtered = actions.astype(np.result_type(actions.dtype, np.float32), copy=True)
    filtered[radius:-radius, joint_columns] = smoothed
    return filtered, window
