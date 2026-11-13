#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""内存实时监控 + 持续落盘记录（数据采集工作机用）。

默认持续把内存用量写入 CSV（自动按启动时间命名，逐行 flush，中断不丢数据）：
    <脚本目录>/mem_logs/mem_YYYYmmdd_HHMMSS.csv

CSV 列: datetime, used_gib, avail_gib, used_pct, swap_used_gib, top_proc, top_proc_rss_mb

用法（纯标准库，任意 python3）：
    python mem_monitor.py                        # 监控 + 默认落盘，Ctrl+C 退出
    python mem_monitor.py --watch lerobot        # 只看名称含 lerobot 的进程（落盘记录其中最大者）
    python mem_monitor.py --gpu                  # 附加 nvidia-smi 显存
    python mem_monitor.py --log /path/mem.csv    # 自定义 CSV 路径（追加模式）
    python mem_monitor.py --no-log               # 不落盘，只看实时界面
    python mem_monitor.py --interval 2 --top 10  # 刷新间隔 2 秒，显示前 10 进程
"""

import argparse
import csv
import os
import subprocess
import sys
import time
from datetime import datetime

PAGE = os.sysconf("SC_PAGE_SIZE")
HZ = os.sysconf("SC_CLK_TCK")
USE_COLOR = sys.stdout.isatty()
CURSOR_OK = sys.stdout.isatty()
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if USE_COLOR else s


def fmt_gb(kib: float) -> str:
    return f"{kib / 1048576:7.2f}G"


def read_meminfo() -> dict[str, float]:
    info: dict[str, float] = {}
    with open("/proc/meminfo", encoding="ascii") as f:
        for line in f:
            key, val, *_ = line.split()
            info[key.rstrip(":")] = float(val)  # 单位 KiB
    return info


def read_processes() -> dict[int, dict]:
    """读取所有进程的 RSS 与 CPU 时间。返回 {pid: {...}}。"""
    procs: dict[int, dict] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        try:
            with open(f"/proc/{pid}/statm", encoding="ascii") as f:
                rss_pages = int(f.read().split()[1])
            with open(f"/proc/{pid}/stat", encoding="ascii") as f:
                stat = f.read()
            # comm 可能含空格/括号，取最后一对括号之后的部分
            rparen = stat.rfind(")")
            fields = stat[rparen + 2 :].split()
            utime, stime = int(fields[11]), int(fields[12])
            with open(f"/proc/{pid}/comm", encoding="utf-8", errors="replace") as f:
                comm = f.read().strip()
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().split(b"\0")
                name = os.path.basename(cmdline[0].decode("utf-8", "replace")) if cmdline and cmdline[0] else comm
                extra = cmdline[1].decode("utf-8", "replace") if len(cmdline) > 1 and cmdline[1] else ""
                if len(extra) > 28:
                    extra = extra[:27] + "…"
            except Exception:
                name, extra = comm, ""  # 内核线程
            procs[pid] = {
                "name": name,
                "extra": extra,
                "rss": rss_pages * PAGE,
                "cpu_ticks": utime + stime,
            }
        except (OSError, ValueError, IndexError):
            continue
    return procs


def gpu_mem() -> list[tuple[str, float, float]] | None:
    """[(gpu名, 已用MiB, 总MiB)]，无 nvidia-smi 返回 None。"""
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=3,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if r.returncode != 0:
        return None
    out = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 3:
            out.append((parts[0], float(parts[1]), float(parts[2])))
    return out or None


def render(args, prev_procs, procs, mem, peak, gpu, elapsed_s) -> tuple[list[str], tuple | None]:
    """渲染界面；返回 (行列表, 用于落盘的 (top_proc, top_rss_mb) 或 None)。"""
    total = mem.get("MemTotal", 0.0)
    avail = mem.get("MemAvailable", mem.get("MemFree", 0.0))
    used = total - avail
    pct = used / total * 100 if total else 0.0
    swap_total = mem.get("SwapTotal", 0.0)
    swap_used = swap_total - mem.get("SwapFree", 0.0)
    swap_pct = swap_used / swap_total * 100 if swap_total else 0.0
    if used > peak["used"]:
        peak["used"] = used
        peak["pct"] = pct

    def bar(p: float, width: int = 20) -> str:
        filled = int(p / 100 * width)
        return "▓" * filled + "░" * (width - filled)

    ram_color = "31" if pct >= 90 else "33" if pct >= 75 else "32"
    lines = []
    lines.append(
        c("1", f"内存监控  {time.strftime('%H:%M:%S')}   运行 {int(elapsed_s)//60:02d}:{int(elapsed_s)%60:02d}   ")
        + c("2", f"峰值 {fmt_gb(peak['used'])} ({peak['pct']:.0f}%)   Ctrl+C 退出")
    )
    lines.append(
        f" RAM   总 {fmt_gb(total)} | 已用 {c(ram_color, fmt_gb(used))} ({pct:5.1f}%) {bar(pct)} | 可用 {fmt_gb(avail)}"
    )
    if swap_total > 0:
        sw_color = "33" if swap_pct > 20 else "2"
        lines.append(f" SWAP  总 {fmt_gb(swap_total)} | 已用 {c(sw_color, fmt_gb(swap_used))} ({swap_pct:5.1f}%) {bar(swap_pct)}")
    if gpu:
        for name, used_m, total_m in gpu:
            gp = used_m / total_m * 100 if total_m else 0
            lines.append(f" GPU   {name[:20]:<20} | 显存 {used_m:7.0f}M ({gp:5.1f}%) {bar(gp)}")

    # 进程列表：RSS 降序
    rows = []
    dt = max(1e-6, elapsed_s if prev_procs is None else args.interval)
    for pid, p in procs.items():
        if args.watch and args.watch.lower() not in p["name"].lower() and args.watch.lower() not in p["extra"].lower():
            continue
        cpu_pct = 0.0
        old = prev_procs.get(pid) if prev_procs else None
        if old:
            cpu_pct = (p["cpu_ticks"] - old["cpu_ticks"]) / HZ / dt * 100
        rows.append((p["rss"], cpu_pct, pid, p["name"], p["extra"]))
    rows.sort(reverse=True)

    lines.append(f" 进程 TOP{min(args.top, len(rows))} (按 RSS){'   过滤: ' + args.watch if args.watch else ''}")
    for rss, cpu_pct, pid, name, extra in rows[: args.top]:
        lines.append(
            f"   {pid:>7}  {name[:18]:<18} {extra:<28} RSS {rss/1048576:8.1f}M  CPU {cpu_pct:5.1f}%"
        )
    if not rows:
        lines.append("   （无匹配进程）")

    top_row = rows[0] if rows else None
    top_log = (f"{top_row[3]}({top_row[2]})", top_row[0] / 1048576) if top_row else None
    return lines, top_log


def open_log(args) -> tuple | None:
    """打开 CSV 日志（追加模式，文件已存在且非空则不重复写表头）。"""
    path = args.log
    if not path:
        os.makedirs(os.path.join(SCRIPT_DIR, "mem_logs"), exist_ok=True)
        path = os.path.join(SCRIPT_DIR, "mem_logs", f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    need_header = not os.path.exists(path) or os.path.getsize(path) == 0
    f = open(path, "a", newline="", encoding="utf-8")
    w = csv.writer(f)
    if need_header:
        w.writerow(["datetime", "used_gib", "avail_gib", "used_pct", "swap_used_gib", "top_proc", "top_proc_rss_mb"])
    return f, w, path


def main() -> None:
    ap = argparse.ArgumentParser(description="内存实时监控 + 持续 CSV 落盘")
    ap.add_argument("--interval", type=float, default=1.0, help="刷新/记录间隔秒（默认 1）")
    ap.add_argument("--top", type=int, default=15, help="显示前 N 个进程")
    ap.add_argument("--watch", default="", help="只显示名称/参数包含该关键字的进程")
    ap.add_argument("--gpu", action="store_true", help="显示 GPU 显存（需要 nvidia-smi）")
    ap.add_argument("--log", default="", help="CSV 路径（默认自动生成 mem_logs/mem_时间戳.csv，追加模式）")
    ap.add_argument("--no-log", action="store_true", help="不落盘，只看实时界面")
    ap.add_argument("--duration", type=float, default=0.0, help="运行秒数，0 表示直到 Ctrl+C")
    args = ap.parse_args()

    log = None
    if not args.no_log:
        log = open_log(args)
        print(f"日志: {log[2]}  (每 {args.interval}s 记录一条，Ctrl+C 结束后可查看)")
        log[0].flush()

    peak = {"used": 0.0, "pct": 0.0}
    prev_procs = None
    painted = 0
    gpu_cache, gpu_ts = None, 0.0
    if CURSOR_OK:
        print("\033[?25l", end="", flush=True)
    start = time.monotonic()
    try:
        while True:
            procs = read_processes()
            mem = read_meminfo()
            if args.gpu and time.monotonic() - gpu_ts > 5:
                gpu_cache, gpu_ts = gpu_mem(), time.monotonic()
            lines, top_log = render(args, prev_procs, procs, mem, peak, gpu_cache if args.gpu else None, time.monotonic() - start)
            prev_procs = procs
            if CURSOR_OK and painted:
                print(f"\033[{painted}A\033[J", end="")
            print("\n".join(lines), flush=True)
            if not CURSOR_OK:
                print()
            painted = len(lines) if CURSOR_OK else 0
            if log:
                total = mem.get("MemTotal", 0.0)
                avail = mem.get("MemAvailable", 0.0)
                used = total - avail
                log[1].writerow([
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    f"{used / 1048576:.3f}",
                    f"{avail / 1048576:.3f}",
                    f"{used / total * 100:.1f}" if total else "0",
                    f"{(mem.get('SwapTotal', 0) - mem.get('SwapFree', 0)) / 1048576:.3f}",
                    top_log[0] if top_log else "",
                    f"{top_log[1]:.1f}" if top_log else "",
                ])
                log[0].flush()  # 逐行落盘，中断不丢数据
            if args.duration > 0 and time.monotonic() - start >= args.duration:
                break
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass
    finally:
        if CURSOR_OK:
            print("\033[?25h", end="", flush=True)
        if log:
            log[0].close()
    mem = read_meminfo()
    total = mem.get("MemTotal", 0.0)
    print()
    print(f"峰值 RAM 用量: {fmt_gb(peak['used'])} / {fmt_gb(total)} ({peak['pct']:.1f}%)")
    if log:
        print(f"日志已保存: {log[2]}")


if __name__ == "__main__":
    main()
