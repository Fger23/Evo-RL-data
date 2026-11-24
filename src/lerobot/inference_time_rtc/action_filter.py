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


def filter_joint_targets_tensor(actions, joint_columns: tuple[int, ...], window: int, valid_mask):
    """Differentiable training equivalent for contiguous, right-padded chunks.

    S-G commutes with the native PI05 per-channel affine action normalization.
    Keep each sample's endpoints and gripper channels, including 7-to-5 fallback.
    Runtime still filters physical targets only after dropping expired rows.
    """
    import torch

    if window not in ACTION_SG_WINDOWS:
        raise ValueError("action_sg_window must be 0, 5, or 7")
    if not window or not joint_columns:
        return actions
    if actions.ndim != 3 or valid_mask.shape != actions.shape[:2]:
        raise ValueError("Expected [B,H,A] actions and [B,H] validity mask")
    lengths = valid_mask.sum(dim=1)
    positions = torch.arange(actions.shape[1], device=actions.device)[None, :]
    if not torch.equal(valid_mask, positions < lengths[:, None]):
        raise ValueError("S-G training validity must be contiguous and right-padded")
    result = actions.clone()
    for effective in ((7, 5) if window == 7 else (5,)):
        if actions.shape[1] < effective:
            continue
        radius = effective // 2
        values = actions[:, :, joint_columns].double().unfold(1, effective, 1)
        kernel = torch.as_tensor(_KERNELS[effective], device=actions.device)
        smooth = (values @ kernel).clamp(values.amin(dim=-1), values.amax(dim=-1)).to(actions.dtype)
        interior = positions[:, radius:-radius]
        selected = (interior < lengths[:, None] - radius) & (lengths[:, None] >= effective)
        if effective == 5 and window == 7:
            selected = selected & (lengths[:, None] < 7)
        result[:, radius:-radius, joint_columns] = torch.where(
            selected[:, :, None], smooth, result[:, radius:-radius, joint_columns]
        )
    return result
