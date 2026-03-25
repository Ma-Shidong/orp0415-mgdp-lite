import argparse
import json
import math
import os
import statistics
import time
from pathlib import Path


def parse_blocks(log_path: Path):
    blocks = []
    current = {}
    try:
        lines = log_path.read_text(errors="ignore").splitlines()
    except FileNotFoundError:
        return blocks
    for line in lines:
        if not line.strip():
            if "iter" in current:
                blocks.append(current)
            current = {}
            continue
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        key = key.strip()
        value = value.strip()
        try:
            current[key] = float(value) if key != "iter" else int(float(value))
        except Exception:
            continue
    if "iter" in current:
        blocks.append(current)
    for block in blocks:
        if "curriculum/level" not in block and "curriculum/level_env" in block:
            block["curriculum/level"] = block["curriculum/level_env"]
    return blocks


def latest_window(blocks, key, window=50):
    vals = [b[key] for b in blocks if key in b]
    if not vals:
        return []
    return vals[-window:]


def finite_or_none(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def classify(blocks):
    latest = blocks[-1]
    issues = []
    status = "OK"

    for key in (
        "rollout_fps",
        "entropy",
        "value_loss",
        "policy_loss",
        "critic_grad_norm",
        "actor_grad_norm",
        "explained_var",
    ):
        v = latest.get(key)
        if v is None or not math.isfinite(v):
            issues.append(f"non_finite:{key}")
            status = "ALERT"

    if latest.get("train/stats.done_nan", 0.0) > 0.0:
        issues.append("done_nan>0")
        status = "ALERT"

    fps_hist = latest_window(blocks, "rollout_fps", window=50)
    if len(fps_hist) >= 10:
        med = statistics.median(fps_hist)
        cur = latest.get("rollout_fps", med)
        if med > 0 and cur < 0.65 * med:
            issues.append(f"rollout_fps_drop:{cur:.0f}<{0.65*med:.0f}")
            status = "WARN" if status != "ALERT" else status

    if latest.get("train/stats.done_safety", 0.0) > 0.85:
        issues.append("done_safety_high")
        status = "WARN" if status != "ALERT" else status

    if latest.get("train/stats.done_timeout", 0.0) > 0.50:
        issues.append("done_timeout_high")
        status = "WARN" if status != "ALERT" else status

    if latest.get("explained_var", 0.0) < -0.10:
        issues.append("explained_var_negative")
        status = "WARN" if status != "ALERT" else status

    if latest.get("critic_grad_norm", 0.0) > 50.0:
        issues.append("critic_grad_spike")
        status = "WARN" if status != "ALERT" else status

    if latest.get("actor_grad_norm", 0.0) > 20.0:
        issues.append("actor_grad_spike")
        status = "WARN" if status != "ALERT" else status

    summary = {
        "status": status,
        "issues": issues,
        "iter": int(latest.get("iter", -1)),
        "env_frames": int(latest.get("env_frames", -1)),
        "level": finite_or_none(latest.get("curriculum/level")),
        "rollout_fps": finite_or_none(latest.get("rollout_fps")),
        "done_success": finite_or_none(latest.get("train/stats.done_success")),
        "done_safety": finite_or_none(latest.get("train/stats.done_safety")),
        "done_timeout": finite_or_none(latest.get("train/stats.done_timeout")),
        "done_bound": finite_or_none(latest.get("train/stats.done_bound")),
        "avg_speed": finite_or_none(latest.get("train/stats.avg_speed")),
        "entropy": finite_or_none(latest.get("entropy")),
        "explained_var": finite_or_none(latest.get("explained_var")),
        "actor_grad_norm": finite_or_none(latest.get("actor_grad_norm")),
        "critic_grad_norm": finite_or_none(latest.get("critic_grad_norm")),
    }
    return summary


def append_status(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--interval-iters", type=int, default=500)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--stall-seconds", type=int, default=1800)
    args = parser.parse_args()

    log_path = Path(args.log_path)
    status_path = Path(args.status_path)
    interval_iters = max(1, int(args.interval_iters))
    poll_seconds = max(5, int(args.poll_seconds))
    stall_seconds = max(60, int(args.stall_seconds))

    last_bucket = -1
    last_mtime = 0.0
    last_stall_report = 0.0

    append_status(
        status_path,
        {
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
            "status": "START",
            "log_path": str(log_path),
            "interval_iters": interval_iters,
            "poll_seconds": poll_seconds,
        },
    )

    while True:
        now = time.time()
        if log_path.exists():
            mtime = log_path.stat().st_mtime
            if mtime > last_mtime:
                last_mtime = mtime
            elif now - last_mtime >= stall_seconds and now - last_stall_report >= stall_seconds:
                append_status(
                    status_path,
                    {
                        "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "ALERT",
                        "issues": ["log_stalled"],
                        "seconds_since_update": int(now - last_mtime),
                    },
                )
                last_stall_report = now

            blocks = parse_blocks(log_path)
            if blocks:
                latest_iter = int(blocks[-1].get("iter", -1))
                bucket = latest_iter // interval_iters
                if latest_iter >= interval_iters and bucket > last_bucket:
                    summary = classify(blocks)
                    summary["ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    summary["bucket"] = bucket
                    append_status(status_path, summary)
                    last_bucket = bucket

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
