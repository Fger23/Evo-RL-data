"""Keep normalized prefixes aligned with the client's command timeline."""

from dataclasses import dataclass
from threading import Lock

import numpy as np

from .action_filter import filter_joint_targets
from .protocol import RTCActionChunk


class RTCContinuityError(RuntimeError):
    """A stale action chunk arrived too late to merge into the current timeline."""


@dataclass(frozen=True)
class RTCSpliceDiagnostics:
    request_id: int
    actual_delay: int
    old_remaining: int
    effective_horizon: int | None
    overlap_delta: np.ndarray | None


@dataclass(frozen=True)
class RTCActionReservation:
    request_id: int
    step: int
    chunk_index: int
    action: np.ndarray
    rlt_applied: bool = False


class RTCActionBuffer:
    def __init__(self, *, sg_window: int = 0, joint_columns: tuple[int, ...] = ()):
        self.lock = Lock()
        self._sg_window = sg_window
        self._joint_columns = joint_columns
        self.applied_sg_window = 0
        self.step = 0
        self.started = False
        self.origin = 0
        self.last_request_id = -1
        self._chunk_offset = 0
        self.original: np.ndarray | None = None
        self.processed: np.ndarray | None = None
        self._rlt_action_mask: np.ndarray | None = None
        self.last_splice: RTCSpliceDiagnostics | None = None
        self._motion_paused = False
        self._draining = False

    @property
    def motion_paused(self) -> bool:
        with self.lock:
            return self._motion_paused

    @property
    def draining(self) -> bool:
        with self.lock:
            return self._draining

    def _remaining(self) -> int:
        if self.original is None:
            return 0
        return max(0, len(self.original) - (self.step - self.origin))

    def qsize(self) -> int:
        with self.lock:
            return self._remaining()

    def snapshot(self) -> tuple[int, np.ndarray | None, bool]:
        """Atomically copy the unconsumed model-space prefix and its timeline origin."""
        with self.lock:
            prefix = None
            if self._remaining():
                prefix = self.original[self.step - self.origin :].copy()
            return self.step, prefix, self.started

    def merge(
        self,
        chunk: RTCActionChunk,
        *,
        prefix_horizon: int | None = None,
        allow_late_splice: bool = False,
        drain_on_rejection: bool = False,
    ) -> int | None:
        """Replace future actions; drop slots elapsed since this request was captured.

        The first chunk starts the timeline on arrival, so model warmup cannot
        expire the entire bootstrap chunk. Subsequent delays include empty slots.
        """
        chunk.validate()
        if prefix_horizon is not None and prefix_horizon < 0:
            raise ValueError("prefix_horizon must be non-negative")
        with self.lock:
            if self._motion_paused or self._draining or chunk.request_id <= self.last_request_id:
                return None
            if chunk.start_step > self.step:
                raise ValueError("Action chunk starts in the future")
            delay = self.step - chunk.start_step if self.started else 0
            if delay >= len(chunk.original_actions):
                self._draining = drain_on_rejection
                raise RTCContinuityError(
                    "Entire RTC chunk expired before arrival; reduce fps or model/network latency"
                )
            remaining = self._remaining()
            delta = None
            if self.started and remaining and self.processed is not None:
                # Compare the old and proposed targets for the same command slot.
                # This is not a measured joint motion or a velocity bound.
                delta = chunk.processed_actions[delay].astype(np.float64) - self.processed[
                    self.step - self.origin
                ].astype(np.float64)
            self.last_splice = RTCSpliceDiagnostics(chunk.request_id, delay, remaining, prefix_horizon, delta)
            if (
                not allow_late_splice
                and self.started
                and prefix_horizon is not None
                and delay >= prefix_horizon
            ):
                self._draining = drain_on_rejection
                raise RTCContinuityError(
                    f"RTC prefix continuity lost: actual_delay={delay} >= "
                    f"effective_horizon={prefix_horizon}; refusing to replace future actions"
                )
            if not self._sg_window and getattr(chunk, "rlt_applied", False):
                processed = chunk.processed_actions[delay:].copy()
                applied_window = chunk.applied_sg_window
            else:
                processed, applied_window = filter_joint_targets(
                    chunk.processed_actions[delay:], self._joint_columns, self._sg_window
                )
            self._install_chunk(chunk, delay, processed, applied_window)
            return delay

    def _install_chunk(self, chunk, offset, processed, applied_window):
        self.original = chunk.original_actions[offset:].copy()
        self.processed = processed
        mask = getattr(chunk, "rlt_action_mask", None)
        self._rlt_action_mask = None if mask is None else mask[offset:].copy()
        self.applied_sg_window = applied_window
        self.origin = self.step
        self.last_request_id = chunk.request_id
        self._chunk_offset = offset
        self.started = True

    def paused_step(self) -> int | None:
        with self.lock:
            return self.step if self._motion_paused else None

    def resume(self, chunk: RTCActionChunk) -> bool:
        """Install a fresh chunk from row zero on the paused timeline."""
        chunk.validate()
        with self.lock:
            if not self._motion_paused or chunk.start_step != self.step or chunk.request_id <= self.last_request_id:
                return False
            if not self._sg_window and getattr(chunk, "rlt_applied", False):
                processed, window = chunk.processed_actions.copy(), chunk.applied_sg_window
            else:
                processed, window = filter_joint_targets(chunk.processed_actions, self._joint_columns, self._sg_window)
            self._install_chunk(chunk, 0, processed, window)
            self.last_splice = None
            self._motion_paused = False
            self._draining = False
            return True

    def advance(self, skipped_steps: int = 0) -> np.ndarray | None:
        """Reserve one command slot; missed deadlines never cause catch-up bursts."""
        reservation = self.reserve(skipped_steps)
        return None if reservation is None else reservation.action.copy()

    def reserve(
        self, skipped_steps: int = 0, *, pause_on_exhaustion: bool = False
    ) -> RTCActionReservation | None:
        """Reserve a command with its absolute slot and row in the received chunk."""
        if skipped_steps < 0:
            raise ValueError("skipped_steps must be non-negative")
        with self.lock:
            if not self.started or self._motion_paused:
                return None
            self.step += skipped_steps
            step = self.step
            index = step - self.origin
            self.step += 1
            if self.processed is None or index >= len(self.processed):
                # Latch under the merge lock so a result arriving after the
                # missed command slot cannot refill the live buffer.
                if pause_on_exhaustion:
                    self._motion_paused = True
                return None
            action = self.processed[index].copy()
            action.setflags(write=False)
            rlt_applied = bool(self._rlt_action_mask[index]) if self._rlt_action_mask is not None else False
            return RTCActionReservation(
                self.last_request_id, step, self._chunk_offset + index, action, rlt_applied
            )
