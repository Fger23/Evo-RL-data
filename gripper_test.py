#!/usr/bin/env python
"""Piper 夹爪小幅度开合控制脚本.

在指定 CAN 口（默认 can3，左从臂）上使能夹爪并做小幅开合循环：
close_pos <-> open_pos（单位: 度，范围 0-90），按 rate Hz 连续下发 GripperCtrl
（与 lerobot 遥操时持续 send_action 的方式一致）。

用法示例:
    python gripper_small_cycle.py                     # can3, 0 <-> 15 deg, 5 次
    python gripper_small_cycle.py --port can2 --min 0 --max 20 --cycles 8
    python gripper_small_cycle.py --min 5 --max 20 --hold 2.0 --rate 60
"""
import argparse
import sys
import time

from piper_sdk import C_PiperInterface_V2, LogLevel


def read_gripper_deg(arm) -> float:
    st = arm.GetArmGripperMsgs().gripper_state
    return abs(getattr(st, "grippers_angle", 0) * 1e-3)


def print_foc_status(arm, tag: str) -> None:
    st = arm.GetArmGripperMsgs().gripper_state
    foc = getattr(st, "foc_status", None)
    if foc is None:
        return
    print(
        f"[{tag}] angle={st.grippers_angle / 1e3:.2f}deg effort={st.grippers_effort} "
        f"status_code=0x{st.status_code:X} driver_enable={foc.driver_enable_status} "
        f"homing={getattr(foc, 'homing_status', '?')} overcurrent={foc.driver_overcurrent} "
        f"voltage_low={foc.voltage_too_low}"
    )


def run_target(arm, target: float, hold_s: float, rate_hz: float, effort: int) -> float:
    """按 rate_hz 持续下发目标位置 hold_s 秒，返回最终读数。"""
    cmd = int(round(target * 1e3))
    period = 1.0 / max(rate_hz, 1.0)
    t0 = time.monotonic()
    last = read_gripper_deg(arm)
    next_print = t0
    while time.monotonic() - t0 < hold_s:
        arm.GripperCtrl(cmd, effort, 0x01, 0x00)
        last = read_gripper_deg(arm)
        if time.monotonic() >= next_print:  # 每 0.5s 打一次读数
            print(f"    readback: {last:7.2f} deg (target {target:.0f})")
            next_print += 0.5
        time.sleep(period)
    return last


def main() -> int:
    ap = argparse.ArgumentParser(description="Piper gripper small open/close cycle")
    ap.add_argument("--port", default="can3", help="CAN 接口名 (默认 can3)")
    ap.add_argument("--min", type=float, default=0.0, help="闭合位置 deg (0-90)")
    ap.add_argument("--max", type=float, default=15.0, help="张开位置 deg (0-90)")
    ap.add_argument("--cycles", type=int, default=5, help="开合循环次数")
    ap.add_argument("--hold", type=float, default=1.5, help="每个目标保持时间 s")
    ap.add_argument("--rate", type=float, default=50.0, help="指令下发频率 Hz")
    ap.add_argument("--effort", type=int, default=1000, help="夹爪力度 0-5000")
    args = ap.parse_args()

    lo, hi = sorted((max(0.0, min(90.0, args.min)), max(0.0, min(90.0, args.max))))
    if hi - lo < 1.0:
        print("FAIL: --min 和 --max 至少相差 1 deg")
        return 1
    print(f"[{args.port}] small cycle: {lo:.0f} <-> {hi:.0f} deg x {args.cycles} cycles "
          f"(hold {args.hold}s, {args.rate:.0f}Hz, effort {args.effort})")

    arm = C_PiperInterface_V2(
        can_name=args.port, judge_flag=False, can_auto_init=True, logger_level=LogLevel.WARNING
    )
    arm.ConnectPort()
    time.sleep(0.5)
    if getattr(arm.GetArmStatus(), "time_stamp", 0.0) <= 0.0:
        print(f"FAIL: [{args.port}] 收不到机械臂状态报文，检查 CAN 接线/供电")
        arm.DisconnectPort()
        return 1
    print(f"[{args.port}] CAN OK")
    print_foc_status(arm, "before enable")

    # 与 lerobot piper_follower 相同的使能流程
    arm.MotionCtrl_2(0x01, 0x01, 100, 0xAD)
    time.sleep(0.1)
    for _ in range(15):
        if arm.EnablePiper():
            break
        time.sleep(0.2)
    else:
        print("FAIL: 机械臂使能超时")
        arm.DisconnectPort()
        return 1
    arm.EnableArm(7, 2)  # 显式使能夹爪电机 (joint 7)
    time.sleep(0.5)
    print_foc_status(arm, "after enable")

    moved = False
    try:
        for i in range(args.cycles):
            for target in (hi, lo):
                print(f"[cycle {i + 1}/{args.cycles}] gripper -> {target:.0f} deg")
                final = run_target(arm, target, args.hold, args.rate, args.effort)
                if abs(final - target) > 2.0:
                    moved = moved or abs(final - (lo if target == hi else hi)) > 5.0
                if abs(final - target) > 5.0:
                    print(f"    WARN: 最终 {final:.2f} deg 与目标 {target:.0f} 偏差较大")
    except KeyboardInterrupt:
        print("\n中断，退出")
    finally:
        print_foc_status(arm, "after test")
        arm.DisconnectPort()

    print("提示: 若读数始终为 0 且夹爪无动作，检查夹爪供电/航插线（driver_enable 应为 True）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
