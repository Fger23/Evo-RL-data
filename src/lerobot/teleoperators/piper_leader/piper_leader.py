#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from collections.abc import Callable
from functools import cached_property
from typing import Any, TypeVar

from lerobot.motors import MotorCalibration
from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.piper_sdk import (
    PIPER_ACTION_KEYS,
    PIPER_JOINT_ACTION_KEYS,
    get_piper_sdk,
    parse_piper_log_level,
    unit_to_milli,
    wait_enable_piper,
)
from lerobot.utils.piper_teleop import PiperTeleopFeedback
from lerobot.utils.utils import enter_pressed, move_cursor_up

from ..teleoperator import Teleoperator
from .config_piper_leader import PiperLeaderConfig, PiperXLeaderConfig

logger = logging.getLogger(__name__)
PIPER_CALIB_KEYS = list(PIPER_ACTION_KEYS)
PIPER_CALIB_IDS = {key: idx for idx, key in enumerate(PIPER_CALIB_KEYS)}
PIPER_LEADER_ROLE = 0xFA
PIPER_CONTROLLED_ROLE = 0xFC
PIPER_STANDBY_MODE = 0x00
PIPER_COMMAND_MODE = 0x01
PIPER_ROLE_SWITCH_TIMEOUT_S = 1.5
PIPER_COMMAND_RETRY_INTERVAL_S = 0.05
PIPER_RESTORE_SPEED_RATIO = 20
T = TypeVar("T")


class PiperLeader(Teleoperator):
    """Piper leader arm used as a teleoperator through Piper SDK CAN messages."""

    config_class = PiperLeaderConfig
    name = "piper_leader"
    uses_firmware_leader_mode = True

    def __init__(self, config: PiperLeaderConfig | PiperXLeaderConfig):
        super().__init__(config)
        self.config = config
        self._is_connected = False
        self._manual_control_enabled: bool | None = None
        self._last_mode_refresh_t = 0.0
        self._mirror_mode_pending = False
        self._can_feedback: PiperTeleopFeedback | None = None
        self._role_requested_at = 0.0
        self._mirror_waiting = False
        self._last_leader_frame_warning = float("-inf")

        interface_cls, _ = get_piper_sdk()
        self.arm = interface_cls(
            can_name=self.config.port,
            judge_flag=self.config.judge_flag,
            can_auto_init=self.config.can_auto_init,
            logger_level=parse_piper_log_level(self.config.log_level),
        )

    @cached_property
    def action_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return dict.fromkeys(PIPER_ACTION_KEYS, float)

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True, *, for_policy: bool = False) -> None:
        self._can_feedback = PiperTeleopFeedback(self.arm)
        try:
            self.arm.ConnectPort()
            if self.config.startup_sleep_s > 0:
                time.sleep(self.config.startup_sleep_s)
            self._is_connected = True
            needs_calibration = not self.is_calibrated and calibrate and self.config.require_calibration
            self.set_manual_control(not for_policy or needs_calibration)
            if needs_calibration:
                logger.info(
                    "No piper-leader calibration file found for '%s'. Running lerobot-calibrate flow.",
                    self.id,
                )
                self.calibrate()
                if for_policy:
                    self.set_manual_control(False)
        except BaseException:
            self._close_connection()
            raise

        logger.info("%s connected.", self)

    def _use_uncalibrated_passthrough(self) -> bool:
        return not self.is_calibrated and not self.config.require_calibration

    @property
    def is_calibrated(self) -> bool:
        if not all(key in self.calibration for key in PIPER_CALIB_KEYS):
            return False
        for key in PIPER_CALIB_KEYS:
            cal = self.calibration[key]
            if cal.range_max <= cal.range_min:
                return False
        return True

    def calibrate(self) -> None:
        if self.calibration and self.is_calibrated:
            user_input = input(
                f"Press ENTER to use existing calibration file for id '{self.id}', "
                "or type 'c' and press ENTER to run a new calibration: "
            )
            if user_input.strip().lower() != "c":
                return

        if not self.config.sync_gripper:
            raise RuntimeError("PiPER leader calibration requires gripper feedback.")
        self.set_manual_control(True)
        logger.info("Running calibration for %s", self)
        input("Move piper-leader to your desired neutral/center pose, then press ENTER...")
        neutral = self._read_raw_action()
        print("Move all piper-leader joints through full range. Press ENTER to stop recording...")
        range_mins, range_maxes = self._record_ranges_of_motion()

        self.calibration = {}
        for key in PIPER_CALIB_KEYS:
            min_deg = range_mins[key]
            max_deg = range_maxes[key]
            if max_deg <= min_deg:
                raise ValueError(f"Invalid range for {key}: min={min_deg:.3f}, max={max_deg:.3f}")

            neutral_deg = min(max_deg, max(min_deg, neutral[key]))
            self.calibration[key] = MotorCalibration(
                id=PIPER_CALIB_IDS[key],
                drive_mode=0,
                homing_offset=self._to_calibration_units(neutral_deg),
                range_min=self._to_calibration_units(min_deg),
                range_max=self._to_calibration_units(max_deg),
            )

        self._save_calibration()
        print(f"Calibration saved to {self.calibration_fpath}")

    def _send_command_mode(self, *, restoring: bool = False) -> None:
        follow_mode = 0xAD if self.config.command_high_follow and not restoring else 0x00
        speed = self.config.command_speed_ratio
        if restoring:
            speed = min(speed, PIPER_RESTORE_SPEED_RATIO)
        self.arm.MotionCtrl_2(PIPER_COMMAND_MODE, 0x01, speed, follow_mode)
        self._last_mode_refresh_t = time.monotonic()

    def _refresh_command_mode_if_needed(self) -> None:
        if self._mirror_mode_pending:
            self._send_command_mode()
            self._mirror_mode_pending = False
            return
        interval_s = self.config.mode_refresh_interval_s
        if interval_s <= 0:
            return
        now = time.monotonic()
        if now - self._last_mode_refresh_t >= interval_s:
            self._send_command_mode()

    def _wait_enable(self, timeout_s: float) -> bool:
        return wait_enable_piper(self.arm, timeout_s)

    def _send_gripper_ctrl(self, gripper_pos_raw: int) -> None:
        self.arm.GripperCtrl(
            gripper_pos_raw,
            self.config.gripper_effort_default,
            self.config.gripper_status_code,
            0x00,
        )

    @check_if_not_connected
    def set_manual_control(self, enabled: bool) -> None:
        if self._manual_control_enabled is enabled:
            return

        can_feedback = self._can_feedback
        assert can_feedback is not None
        self._manual_control_enabled = None
        self._role_requested_at = time.time()
        role = PIPER_LEADER_ROLE if enabled else PIPER_CONTROLLED_ROLE
        self.arm.MasterSlaveConfig(role, 0, 0, 0)

        if enabled:
            # Firmware supplies gravity compensation in leader/zero-force drag
            # mode. Do not send host MIT torques or position targets in this role.
            self._last_leader_frame_warning = float("-inf")
            logger.info("[%s] PiPER firmware leader drag mode is active", self.config.port)
        else:
            self._restore_position_control(can_feedback)
            grip = can_feedback.gripper_width(since=self._role_requested_at)
            if self.config.sync_gripper and grip is not None:
                self._send_gripper_ctrl(unit_to_milli(grip))
            logger.info("[%s] PiPER CAN position-control mode is active", self.config.port)
        self._manual_control_enabled = enabled

    def _restore_position_control(self, can_feedback: PiperTeleopFeedback) -> None:
        role_requested_at = self._role_requested_at

        def read_controlled_pose(*, since):
            if can_feedback.ctrl_mode(since=since) not in (PIPER_STANDBY_MODE, PIPER_COMMAND_MODE):
                return None
            return can_feedback.joint_positions(since=since)

        try:
            self._wait_for(
                lambda: read_controlled_pose(since=role_requested_at),
                "fresh controlled-mode status and joint pose after 0xFC",
            )
        except RuntimeError as exc:
            raise RuntimeError(f"{exc} {can_feedback.arm_status_detail(since=role_requested_at)}.") from exc
        logger.info(
            "[%s] PiPER exited leader mode: %s",
            self.config.port,
            can_feedback.arm_status_detail(since=role_requested_at),
        )
        # The SDK MOVE J example confirms enable before selecting position
        # control. Do this only after receiving feedback from the controlled role.
        if not self._wait_enable(self.config.enable_timeout_s):
            raise RuntimeError(
                f"[{self.config.port}] PiPER did not enable after leaving leader mode; "
                f"{can_feedback.arm_status_detail(since=role_requested_at)}."
            )

        command_requested_at: float | None = None
        next_attempt_at = 0.0
        attempts = 0

        def activate_command_mode():
            nonlocal command_requested_at, next_attempt_at, attempts
            if command_requested_at is not None and (
                can_feedback.ctrl_mode(since=command_requested_at) == PIPER_COMMAND_MODE
            ):
                return True
            now = time.monotonic()
            if now < next_attempt_at:
                return None
            pose = read_controlled_pose(since=role_requested_at)
            if pose is None:
                return None
            # Retry only at the arm's live pose, never the last model target
            # or the joint sample from before the role switch.
            hold = tuple(unit_to_milli(pose[key]) for key in PIPER_JOINT_ACTION_KEYS)
            self.arm.JointCtrl(*hold)
            requested_at = time.time()
            self._send_command_mode(restoring=True)
            self.arm.JointCtrl(*hold)
            if command_requested_at is None:
                command_requested_at = requested_at
            attempts += 1
            next_attempt_at = now + PIPER_COMMAND_RETRY_INTERVAL_S
            return None

        try:
            self._wait_for(activate_command_mode, "CAN position-control status after restoring command mode")
        except RuntimeError as exc:
            detail = can_feedback.arm_status_detail(since=command_requested_at or role_requested_at)
            raise RuntimeError(f"{exc} Attempts: {attempts}; {detail}.") from exc
        # Apply the normal mirroring settings with the next position target.
        self._mirror_mode_pending = True

    def configure(self) -> None:
        self.set_manual_control(True)

    def _wait_for(self, read: Callable[[], T | None], description: str) -> T:
        deadline = time.monotonic() + PIPER_ROLE_SWITCH_TIMEOUT_S
        while time.monotonic() < deadline:
            value = read()
            if value is not None:
                return value
            time.sleep(0.005)
        raise RuntimeError(f"[{self.config.port}] PiPER role transition timed out waiting for {description}.")

    def _read_leader_action(self) -> RobotAction | None:
        assert self._can_feedback is not None
        return self._can_feedback.leader_action(
            since=self._role_requested_at, sync_gripper=self.config.sync_gripper
        )

    def _read_raw_action(self) -> RobotAction:
        if self._manual_control_enabled is not True:
            raise RuntimeError(f"[{self.config.port}] PiPER is not in firmware leader mode.")

        def read_calibration_pose():
            action = self._read_leader_action()
            return action if action is not None and "gripper.pos" in action else None

        return self._wait_for(read_calibration_pose, "fresh leader joint and gripper frames for calibration")

    def _to_calibration_units(self, angle_deg: float) -> int:
        return int(round(angle_deg * self.config.calibration_scale))

    def _from_calibration_units(self, value: int) -> float:
        return float(value) / float(self.config.calibration_scale)

    def _calibrated_to_offset(self, key: str, raw_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        bounded = min(max_deg, max(min_deg, raw_deg))
        centered = bounded - home_deg
        return -centered if cal.drive_mode else centered

    def _offset_to_calibrated(self, key: str, offset_deg: float) -> float:
        cal = self.calibration[key]
        min_deg = self._from_calibration_units(cal.range_min)
        max_deg = self._from_calibration_units(cal.range_max)
        home_deg = self._from_calibration_units(cal.homing_offset)
        centered = -offset_deg if cal.drive_mode else offset_deg
        target = home_deg + centered
        return min(max_deg, max(min_deg, target))

    def _record_ranges_of_motion(self) -> tuple[dict[str, float], dict[str, float]]:
        current = self._read_raw_action()
        mins = current.copy()
        maxes = current.copy()

        while True:
            current = self._read_raw_action()
            mins = {key: min(mins[key], current[key]) for key in PIPER_CALIB_KEYS}
            maxes = {key: max(maxes[key], current[key]) for key in PIPER_CALIB_KEYS}

            print("\n-----------------------------")
            print("JOINT       |    MIN |    POS |    MAX")
            for key in PIPER_CALIB_KEYS:
                print(f"{key:<11} | {mins[key]:>6.2f} | {current[key]:>6.2f} | {maxes[key]:>6.2f}")

            if enter_pressed():
                break
            move_cursor_up(len(PIPER_CALIB_KEYS) + 3)

        return mins, maxes

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        action = self.get_action_if_ready()
        if action is None:
            raise RuntimeError(f"[{self.config.port}] PiPER leader has no fresh action.")
        return action

    @check_if_not_connected
    def get_action_if_ready(self) -> RobotAction | None:
        if self._manual_control_enabled is not True:
            raise RuntimeError(f"[{self.config.port}] PiPER is not in firmware leader mode.")
        if not self.is_calibrated and not self._use_uncalibrated_passthrough():
            raise RuntimeError(
                f"{self} is not calibrated. Run `lerobot-calibrate --teleop.type={self.config.type} --teleop.id={self.id}` first."
            )

        raw_action = self._read_leader_action()
        if raw_action is None:
            now = time.monotonic()
            if now - self._last_leader_frame_warning >= 2.0:
                assert self._can_feedback is not None
                logger.warning(
                    "[%s] PiPER leader joint snapshot unavailable: %s",
                    self.config.port,
                    self._can_feedback.leader_frame_status(since=self._role_requested_at),
                )
                self._last_leader_frame_warning = now
            return None
        if self._use_uncalibrated_passthrough():
            return raw_action
        return {key: self._calibrated_to_offset(key, value) for key, value in raw_action.items()}

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, Any]) -> None:
        if not self.is_calibrated and not self._use_uncalibrated_passthrough():
            raise RuntimeError(
                f"{self} is not calibrated. Run `lerobot-calibrate --teleop.type={self.config.type} --teleop.id={self.id}` first."
            )

        if self._manual_control_enabled is not False:
            raise RuntimeError(f"[{self.config.port}] PiPER must exit leader mode before accepting targets.")
        assert self._can_feedback is not None
        pose = self._can_feedback.joint_positions(since=self._role_requested_at)
        if pose is None or self._can_feedback.ctrl_mode(since=self._role_requested_at) != PIPER_COMMAND_MODE:
            if not self._mirror_waiting:
                logger.warning(
                    "[%s] Pausing PiPER leader mirroring until CAN position feedback is fresh",
                    self.config.port,
                )
            self._mirror_waiting = True
            return
        self._mirror_waiting = False
        self._refresh_command_mode_if_needed()

        joint_keys = PIPER_JOINT_ACTION_KEYS
        has_all_joints = all(key in feedback for key in joint_keys)
        if has_all_joints:
            if self._use_uncalibrated_passthrough():
                joint_targets = [feedback[key] for key in joint_keys]
            else:
                joint_targets = [self._offset_to_calibrated(key, feedback[key]) for key in joint_keys]
            self.arm.JointCtrl(*(unit_to_milli(target) for target in joint_targets))

        if self.config.sync_gripper and "gripper.pos" in feedback:
            if self._use_uncalibrated_passthrough():
                gripper_target = feedback["gripper.pos"]
            else:
                gripper_target = self._offset_to_calibrated("gripper.pos", feedback["gripper.pos"])
            gripper_pos_raw = unit_to_milli(gripper_target)
            self._send_gripper_ctrl(gripper_pos_raw)

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self.config.disable_on_disconnect and self._manual_control_enabled is False:
                self.arm.DisableArm(7)
        finally:
            # Leave the firmware supporting the leader. An I/O fault should
            # not switch its role or remove motor torque.
            self._close_connection()

    def _close_connection(self) -> None:
        try:
            self.arm.DisconnectPort()
        finally:
            if self._can_feedback is not None:
                self._can_feedback.close()
                self._can_feedback = None
            self._is_connected = False
            self._manual_control_enabled = None
            logger.info("%s disconnected.", self)


class PiperXLeader(PiperLeader):
    config_class = PiperXLeaderConfig
    name = "piperx_leader"
