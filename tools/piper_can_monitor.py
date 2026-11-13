#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Piper 机械臂 CAN 口实时监测工具（用于识别 can0~can3 分别接的是哪条臂）。

原理：Piper 机械臂上电后会主动通过 CAN 周期广播反馈报文，本脚本只被动监听、
不发送任何报文（绝对安全）。轻轻晃动某条机械臂，对应 CAN 口的「活动度」就会飙升，
以此识别每个 CAN 口接的是哪条臂。

会解码的报文（Piper V2 协议，大端字节序）：
    0x2A1 / 0x2B1 / 0x2C1  机械臂状态反馈（控制模式 / 状态 / 故障码）
    0x2A5/0x2A6/0x2A7 (及 0x2B*/0x2C* 偏移)  关节角度反馈，单位 0.001°
    0x2A8                  夹爪反馈（角度/电流）
    0x251~0x256            6 个关节驱动器高速反馈（速度/电流/位置）
    0x261~0x266            6 个关节驱动器低速反馈

用法（在装有 python-can 与 piper 环境里，例如 conda activate evo-rl）：
    python piper_can_monitor.py                 # 监听 can0~can3，实时表格
    python piper_can_monitor.py --setup         # 先用 sudo 配置并拉起 CAN 口再监听
    python piper_can_monitor.py --raw           # 打印原始报文流（类似 candump）
    python piper_can_monitor.py --interfaces can0,can1
    python piper_can_monitor.py --duration 30   # 30 秒后自动退出
"""

import argparse
import subprocess
import sys
import threading
import time
from collections import Counter, deque

try:
    import can
except ModuleNotFoundError:
    print("缺少 python-can，请在正确的环境里运行，例如: conda activate evo-rl")
    sys.exit(1)

# ---------------- Piper V2 CAN 报文解码 ----------------

CTRL_MODES = {
    0x00: "待机",
    0x01: "CAN控制",
    0x02: "示教模式",
    0x03: "以太网控制",
    0x04: "WiFi控制",
    0x05: "遥控器控制",
    0x06: "联动示教输入",
    0x07: "离线轨迹",
}
ARM_STATUS = {
    0x00: "正常",
    0x01: "急停",
    0x02: "无解",
    0x03: "奇异点",
    0x04: "目标超限",
    0x05: "关节通信异常",
    0x06: "抱闸未释放",
    0x07: "发生碰撞",
    0x08: "示教超速",
    0x09: "关节状态异常",
    0x0A: "其他错误",
}

# 反馈帧基址：常规 0x2A0，主臂(联动示教输入)可为 0x2B0/0x2C0 偏移
FEEDBACK_BASES = (0x2A0, 0x2B0, 0x2C0)


def _i16(data: bytes, off: int) -> int:
    v = int.from_bytes(data[off : off + 2], "big")
    return v - 0x10000 if v >= 0x8000 else v


def _u16(data: bytes, off: int) -> int:
    return int.from_bytes(data[off : off + 2], "big")


def _i32(data: bytes, off: int) -> int:
    v = int.from_bytes(data[off : off + 4], "big")
    return v - 0x100000000 if v >= 0x80000000 else v


class PortState:
    """每个 CAN 口的实时状态（线程安全）。"""

    def __init__(self, name: str, on_frame_cb=None):
        self.name = name
        self.on_frame_cb = on_frame_cb  # raw 模式下逐帧打印
        self.lock = threading.Lock()
        self.error: str | None = None
        self.total = 0
        self.times: deque[float] = deque(maxlen=4096)  # 用于算 fps
        self.last_rx: float = 0.0
        self.joints: list[float | None] = [None] * 6  # 角度 deg
        self.gripper: float | None = None
        self.gripper_effort: int | None = None
        self.ctrl_mode: int | None = None
        self.arm_status: int | None = None
        self.err_code: int | None = None
        self.motor_last_rx: list[float] = [0.0] * 6  # 驱动器 1~6 最近报文时间
        self.unknown_ids: Counter[int] = Counter()

    def on_frame(self, msg) -> None:
        cid = msg.arbitration_id
        data = bytes(msg.data)
        now = time.monotonic()
        with self.lock:
            self.total += 1
            self.times.append(now)
            self.last_rx = now
            self.error = None

            decoded = self._decode(cid, data, now)
            if not decoded:
                self.unknown_ids[cid] += 1
        if self.on_frame_cb is not None:
            self.on_frame_cb(msg)

    def _decode(self, cid: int, data: bytes, now: float) -> bool:
        # 机械臂状态反馈
        for base in FEEDBACK_BASES:
            off = cid - base
            if off == 1 and len(data) >= 8:
                self.ctrl_mode = data[0]
                self.arm_status = data[1]
                self.err_code = _u16(data, 6)
                return True
            if off == 5 and len(data) >= 8:  # J1 J2
                self.joints[0] = _i32(data, 0) * 0.001
                self.joints[1] = _i32(data, 4) * 0.001
                return True
            if off == 6 and len(data) >= 8:  # J3 J4
                self.joints[2] = _i32(data, 0) * 0.001
                self.joints[3] = _i32(data, 4) * 0.001
                return True
            if off == 7 and len(data) >= 8:  # J5 J6
                self.joints[4] = _i32(data, 0) * 0.001
                self.joints[5] = _i32(data, 4) * 0.001
                return True
            if off == 8 and len(data) >= 7:  # 夹爪
                self.gripper = _i32(data, 0) * 0.001
                self.gripper_effort = _i16(data, 4)
                return True

        # 驱动器信息反馈 0x251~0x256 / 0x261~0x266 -> 电机 1~6 在线
        if 0x251 <= cid <= 0x256 or 0x261 <= cid <= 0x266:
            self.motor_last_rx[(cid - 0x251) % 0x10] = now
            return True
        return False

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            fps = sum(1 for t in self.times if now - t <= 1.0)
            online = (now - self.last_rx) <= 1.5
            motors = [i + 1 for i, t in enumerate(self.motor_last_rx) if now - t <= 1.5]
            return {
                "fps": fps,
                "online": online,
                "joints": list(self.joints),
                "gripper": self.gripper,
                "ctrl_mode": self.ctrl_mode,
                "arm_status": self.arm_status,
                "err_code": self.err_code,
                "motors": motors,
                "total": self.total,
                "unknown": len(self.unknown_ids),
                "error": self.error,
            }


def _friendly_error(name: str, exc: Exception) -> str:
    text = str(exc)
    if "Network is down" in text:
        return f"接口 DOWN：加 --setup 或 sudo ip link set {name} up"
    if "No such device" in text or "not found" in text.lower():
        return f"接口不存在：{name}"
    return text


def listener(state: PortState, bitrate: int) -> None:
    while True:
        try:
            bus = can.interface.Bus(channel=state.name, interface="socketcan", bitrate=bitrate)
        except Exception as e:
            state.error = _friendly_error(state.name, e)
            return
        try:
            while True:
                msg = bus.recv(timeout=0.5)
                if msg is not None:
                    state.on_frame(msg)
        except Exception as e:
            state.error = _friendly_error(state.name, e)
            bus.shutdown()
            time.sleep(0.5)  # 接口被重新拉起后自动重连
        finally:
            try:
                bus.shutdown()
            except Exception:
                pass


def setup_interfaces(interfaces: list[str], bitrate: int) -> None:
    subprocess.run(["sudo", "modprobe", "gs_usb"], capture_output=True)
    for name in interfaces:
        subprocess.run(["sudo", "ip", "link", "set", name, "down"], capture_output=True)
        r = subprocess.run(
            ["sudo", "ip", "link", "set", name, "type", "can", "bitrate", str(bitrate)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            print(f"  ✗ {name}: 配置失败 {(r.stderr or '').strip()}")
            continue
        r = subprocess.run(["sudo", "ip", "link", "set", name, "up"], capture_output=True, text=True)
        mark = "✓" if r.returncode == 0 else "✗"
        print(f"  {mark} {name}: UP @ {bitrate} bps")


# ---------------- 终端渲染 ----------------

USE_COLOR = sys.stdout.isatty()
CURSOR_OK = sys.stdout.isatty()


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def render(states: list[PortState], prev_joints: dict[str, list], dt: float) -> list[str]:
    lines = []
    now_str = time.strftime("%H:%M:%S")
    lines.append(
        c("1", f"Piper CAN 实时监测 (被动监听·只收不发)  {now_str}   ")
        + c("2", "晃动机械臂→看哪个口的活动度飙升   Ctrl+C 退出")
    )
    for st in states:
        s = st.snapshot()
        name = st.name
        prev = prev_joints.get(name, [None] * 6)

        if s["error"]:
            lines.append(f" {name:<6} {c('31', '[无法打开]')} {s['error']}")
            continue

        if s["online"]:
            head = f" {name:<6} {c('32', f'[在线]')}  {s['fps']:>4} 帧/秒"
        else:
            total = f"，历史 {s['total']} 帧" if s["total"] else ""
            lines.append(f" {name:<6} {c('31', '[无数据]')}{total}")
            continue

        # 活动度 = 本刷新周期内 6 关节变化量之和 (deg/s)
        activity = 0.0
        marks = []
        for i, (cur, old) in enumerate(zip(s["joints"], prev)):
            if cur is None:
                marks.append(" " * 7)
                continue
            if old is not None:
                activity += abs(cur - old)
            marks.append(f"J{i + 1}={cur:6.1f}" + ("*" if old is not None and abs(cur - old) > 0.15 else " "))
        degps = activity / dt if dt > 0 else 0.0
        bar = "█" * min(12, int(degps / 15)) if degps > 0.5 else "·"
        act_str = c("1;33", f"活动度 {degps:6.1f}°/s {bar:<12}") if degps > 0.5 else f"活动度 {degps:6.1f}°/s {'':<12}"

        mode = CTRL_MODES.get(s["ctrl_mode"], f"0x{s['ctrl_mode']:02X}") if s["ctrl_mode"] is not None else "-"
        astat = ARM_STATUS.get(s["arm_status"], f"0x{s['arm_status']:02X}") if s["arm_status"] is not None else "-"
        err = f" 故障:0x{s['err_code']:04X}" if s["err_code"] else ""
        grip = f"夹爪={s['gripper']:6.1f}°" if s["gripper"] is not None else "夹爪=  -"
        motors = "".join(str(m) if m in s["motors"] else "·" for m in range(1, 7))

        lines.append(f"{head}  {act_str}  电机[{motors}]")
        lines.append(f"        {'  '.join(marks)}  {grip}  模式:{mode} 状态:{astat}{err}")
        prev_joints[name] = list(s["joints"])
    return lines


def final_summary(states: list[PortState]) -> None:
    print()
    print("=" * 100)
    print("识别结果汇总")
    print("=" * 100)
    online, moving = [], []
    for st in states:
        s = st.snapshot()
        if s["error"]:
            print(f" {st.name:<6} ✗ 无法打开: {s['error']}")
            continue
        if not s["online"]:
            print(f" {st.name:<6} ✗ 无机械臂（或未上电）")
            continue
        online.append(st.name)
        mode = CTRL_MODES.get(s["ctrl_mode"], "-") if s["ctrl_mode"] is not None else "-"
        joints = "  ".join(
            f"J{i + 1}={v:7.2f}°" if v is not None else f"J{i + 1}=   -   "
            for i, v in enumerate(s["joints"])
        )
        print(f" {st.name:<6} ✓ 机械臂在线 | 帧率 {s['fps']}/s | 电机 {[m for m in s['motors']]} | 模式 {mode}")
        print(f"         {joints}")
    print()
    if online:
        print(f" 接有机械臂的 CAN 口: {', '.join(online)}")
        print(" 提示: 晃动各机械臂时看「活动度」，即可确定 can 口 ↔ 机械臂 的对应关系；")
        print("       示教臂(主臂)通常处于「示教模式/联动示教输入」，从臂为「CAN控制/待机」。")


def main() -> None:
    ap = argparse.ArgumentParser(description="Piper 机械臂 CAN 口实时监测（被动监听识别 can 口）")
    ap.add_argument("--interfaces", default="can0,can1,can2,can3", help="逗号分隔的 CAN 口列表")
    ap.add_argument("--bitrate", type=int, default=1_000_000, help="CAN 波特率（Piper 默认 1M）")
    ap.add_argument("--setup", action="store_true", help="先配置并拉起 CAN 口（需要 sudo）")
    ap.add_argument("--rate", type=float, default=5.0, help="界面刷新频率 Hz")
    ap.add_argument("--duration", type=float, default=0.0, help="运行秒数，0 表示直到 Ctrl+C")
    ap.add_argument("--raw", action="store_true", help="打印原始报文流（不画表格）")
    args = ap.parse_args()

    interfaces = [x.strip() for x in args.interfaces.split(",") if x.strip()]
    if args.setup:
        print("配置 CAN 接口（经典 CAN，非 FD）...")
        setup_interfaces(interfaces, args.bitrate)
        print()

    def make_raw_printer(st_name: str):
        def cb(msg) -> None:
            data = " ".join(f"{b:02X}" for b in msg.data)
            hint = ""
            d = bytes(msg.data)
            cid = msg.arbitration_id
            if len(d) >= 8 and 0x2A5 <= cid <= 0x2CF and (cid & 0xF) in (5, 6, 7):
                j = ((cid - 0x2A0) & 0xF) * 2 - 1
                hint = f"   <- J{j}={_i32(d, 0) * 0.001:.2f}° J{j + 1}={_i32(d, 4) * 0.001:.2f}°"
            print(f"{st_name:<6} {time.monotonic():10.3f}  0x{cid:03X}  [{len(d)}]  {data}{hint}", flush=True)

        return cb

    states = [PortState(n, make_raw_printer(n) if args.raw else None) for n in interfaces]
    threads = []
    for st in states:
        t = threading.Thread(target=listener, args=(st, args.bitrate), daemon=True)
        t.start()
        threads.append(t)

    if args.raw:
        deadline = time.monotonic() + args.duration if args.duration > 0 else None
        try:
            while True:
                time.sleep(0.2)
                if deadline and time.monotonic() > deadline:
                    break
        except KeyboardInterrupt:
            pass
        final_summary(states)
        return

    # 表格模式
    prev_joints: dict[str, list] = {}
    period = 1.0 / max(0.5, args.rate)
    if CURSOR_OK:
        print("\033[?25l", end="", flush=True)  # 隐藏光标
    painted = 0
    start = time.monotonic()
    last = start
    try:
        while True:
            time.sleep(period)
            now = time.monotonic()
            dt = now - last
            last = now
            lines = render(states, prev_joints, dt)
            if CURSOR_OK and painted:
                print(f"\033[{painted}A\033[J", end="")
            print("\n".join(lines), flush=True)
            if not CURSOR_OK:
                print()
            painted = len(lines) if CURSOR_OK else 0
            if args.duration > 0 and now - start >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        if CURSOR_OK:
            print("\033[?25h", end="", flush=True)  # 恢复光标
    final_summary(states)


if __name__ == "__main__":
    main()
