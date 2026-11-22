"""Fresh CAN feedback and joint alignment for PiPER software teleoperation."""

import logging
import math
import struct
import time
from dataclasses import dataclass
from threading import Lock

from .piper_sdk import PIPER_JOINT_ACTION_KEYS, milli_to_unit

_LEADER_JOINT_IDS = (0x155, 0x156, 0x157)
_FEEDBACK_JOINT_IDS = (0x2A5, 0x2A6, 0x2A7)
_LEADER_GRIPPER_ID = 0x159
_FEEDBACK_GRIPPER_ID = 0x2A8
_STATUS_ID = 0x2A1
_WATCHED_IDS = {
    *_LEADER_JOINT_IDS,
    *_FEEDBACK_JOINT_IDS,
    _LEADER_GRIPPER_ID,
    _FEEDBACK_GRIPPER_ID,
    _STATUS_ID,
}
_MAX_FRAME_AGE_S = 0.15
_MAX_JOINT_FRAME_SKEW_S = 0.03

MAX_TELEOP_JOINT_STEP_DEG = 2.0
_ALIGNMENT_DEG = 3.0
_FOLLOWING_ERROR_DEG = 8.0


@dataclass(frozen=True)
class _ReceivedFrame:
    bus_time: float
    received_at: float
    data: bytes


class PiperTeleopFeedback:
    """Observe the pinned SDK's receive callback without opening another CAN socket.

    The SDK's aggregate joint getter has one timestamp for three independent CAN
    frames. Check each pair separately, so one live pair can't hide two stale pairs.
    """

    def __init__(self, arm):
        self._bus = arm.GetCanBus()
        self._sdk_callback = self._bus.callback_function
        self._frames: dict[int, _ReceivedFrame] = {}
        self._lock = Lock()
        self._bus.callback_function = self._receive

    def close(self):
        self._bus.callback_function = self._sdk_callback

    def _receive(self, frame):
        self._sdk_callback(frame)
        if (
            frame.arbitration_id in _WATCHED_IDS
            and frame.is_rx
            and not (frame.is_extended_id or frame.is_remote_frame or frame.is_error_frame)
            and len(frame.data) == 8
        ):
            packet = _ReceivedFrame(frame.timestamp, time.monotonic(), bytes(frame.data))
            with self._lock:
                self._frames[frame.arbitration_id] = packet

    def _recent(self, ids, since):
        with self._lock:
            frames = [self._frames.get(can_id) for can_id in ids]
        now_bus, now = time.time(), time.monotonic()
        fresh = [frame for frame in frames if self._is_recent(frame, since, now_bus, now)]
        return fresh if len(fresh) == len(ids) else None

    @staticmethod
    def _is_recent(frame, since, now_bus, now):
        return (
            frame is not None
            and now - frame.received_at <= _MAX_FRAME_AGE_S
            and max(since, now_bus - _MAX_FRAME_AGE_S) < frame.bus_time <= now_bus
        )

    def leader_frame_status(self, *, since):
        """Explain why a complete native-leader snapshot is unavailable."""
        with self._lock:
            frames = tuple(self._frames.get(can_id) for can_id in _LEADER_JOINT_IDS)
        now_bus, now = time.time(), time.monotonic()
        return " ".join(
            f"0x{can_id:03X}="
            + (
                "missing"
                if frame is None
                else f"{'fresh' if self._is_recent(frame, since, now_bus, now) else 'stale'} "
                f"({(now_bus - frame.bus_time) * 1000:.1f} ms)"
            )
            for can_id, frame in zip(_LEADER_JOINT_IDS, frames, strict=True)
        )

    def arm_status_detail(self, *, since):
        """Report the most recent status even when it fails the readiness check."""
        with self._lock:
            frame = self._frames.get(_STATUS_ID)
        if frame is None:
            return "no 0x2A1 status frame received"
        now_bus, now = time.time(), time.monotonic()
        state = "fresh" if self._is_recent(frame, since, now_bus, now) else "stale"
        return (
            f"0x2A1 {state} ({(now_bus - frame.bus_time) * 1000:.1f} ms), "
            f"ctrl=0x{frame.data[0]:02X}, arm=0x{frame.data[1]:02X}, "
            f"move=0x{frame.data[2]:02X}, teach=0x{frame.data[3]:02X}"
        )

    def joint_positions(self, *, leader=False, since=0.0):
        frames = self._recent(_LEADER_JOINT_IDS if leader else _FEEDBACK_JOINT_IDS, since)
        if frames is None:
            return None
        stamps = [frame.bus_time for frame in frames]
        if max(stamps) - min(stamps) > _MAX_JOINT_FRAME_SKEW_S:
            return None
        angles = [raw for frame in frames for raw in struct.unpack(">ii", frame.data)]
        if any(abs(raw) > 360_000 for raw in angles):
            return None
        return dict(zip(PIPER_JOINT_ACTION_KEYS, map(milli_to_unit, angles), strict=True))

    def leader_action(self, *, since, sync_gripper):
        joints = self.joint_positions(leader=True, since=since)
        if joints is None:
            return None
        if sync_gripper:
            width = self.gripper_width(leader=True, since=since)
            if width is not None:
                joints["gripper.pos"] = width
        return joints

    def gripper_width(self, *, leader=False, since=0.0):
        frames = self._recent((_LEADER_GRIPPER_ID if leader else _FEEDBACK_GRIPPER_ID,), since)
        if frames is None:
            return None
        data = frames[0].data
        # This driver uses the gripper width protocol. Angle commands must
        # never be interpreted as millimetres on the follower.
        if leader and data[6] not in (0, 1, 2, 3):
            return None
        width = abs(struct.unpack_from(">i", data)[0])
        return milli_to_unit(width) if width <= 150_000 else None

    def ctrl_mode(self, *, since=0.0):
        frames = self._recent((_STATUS_ID,), since)
        # A controller in a fault state is not a valid source for handoff.
        return frames[0].data[0] if frames is not None and frames[0].data[1] == 0 else None


class PiperTeleopHandoff:
    """Gate follower dispatch on alignment and keep joint commands near live feedback."""

    def __init__(self, read_joint_feedback):
        self._read_joint_feedback = read_joint_feedback
        self.reset()

    def reset(self):
        self._last_joints = None
        self._blocked_reason = None

    def _wait(self, reason, detail=None):
        if reason != self._blocked_reason:
            logging.warning("PiPER teleop holding: %s", detail or reason)
        self._last_joints = None
        self._blocked_reason = reason

    def prepare(self, action):
        if action is None:
            return self._wait("waiting for fresh leader CAN frames on every arm")

        pose = self._read_joint_feedback()
        if not pose or any(
            not math.isfinite(actual) or key not in action or not math.isfinite(action[key])
            for key, actual in pose.items()
        ):
            return self._wait("waiting for complete joint feedback and targets")

        previous = self._last_joints
        bound = _ALIGNMENT_DEG if previous is None else _FOLLOWING_ERROR_DEG
        farthest = max(pose, key=lambda key: abs(action[key] - pose[key]))
        error = abs(action[farthest] - pose[farthest])
        if error > bound:
            return self._wait(
                "joint alignment",
                f"align leader to follower; {farthest} differs by {error:.1f} degrees (limit {bound:.1f})",
            )

        limited = dict(action)
        for key, actual in pose.items():
            last = actual if previous is None else previous[key]
            if previous is not None and abs(action[key] - last) > _FOLLOWING_ERROR_DEG:
                return self._wait(f"{key} jumped since the last teleop command; realign to resume")
            lo = max(actual, last) - MAX_TELEOP_JOINT_STEP_DEG
            hi = min(actual, last) + MAX_TELEOP_JOINT_STEP_DEG
            if lo > hi:
                return self._wait(f"{key} moved away from the last teleop command; realign to resume")
            limited[key] = min(max(action[key], lo), hi)

        if previous is None:
            logging.info("PiPER teleop following: leader frames are fresh and all joints are aligned")
        self._blocked_reason = None
        self._last_joints = {key: limited[key] for key in pose}
        return limited
