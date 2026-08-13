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


def latest_with_key(blocks, key):
    for block in reversed(blocks):
        if key in block:
            return block
    return {}


def finite_or_none(v):
    if v is None:
        return None
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def classify(blocks):
    latest = blocks[-1]
    latest_eval = latest_with_key(blocks, "eval/stats.done_success")
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

    if latest.get("train/stats.done_acc_limit", 0.0) > 0.35:
        issues.append("done_acc_limit_high")
        status = "WARN" if status != "ALERT" else status

    if latest.get("train/stats.target_acc_clip_ratio", 0.0) > 0.20:
        issues.append("target_acc_clip_high")
        status = "WARN" if status != "ALERT" else status

    if latest.get("train/stats.control_cmd_clip_ratio", 0.0) > 0.10:
        issues.append("control_cmd_clip_high")
        status = "WARN" if status != "ALERT" else status

    eval_done_success_latest = latest_eval.get("eval/stats.done_success")
    eval_done_height_low_latest = latest_eval.get("eval/stats.done_height_low")
    train_done_success_latest = latest.get("train/stats.done_success")
    eval_gap_success = None
    if (
        isinstance(train_done_success_latest, (int, float))
        and math.isfinite(train_done_success_latest)
        and isinstance(eval_done_success_latest, (int, float))
        and math.isfinite(eval_done_success_latest)
    ):
        eval_gap_success = float(train_done_success_latest - eval_done_success_latest)

    eval_success_hist = latest_window(blocks, "eval/stats.done_success", window=3)
    if len(eval_success_hist) >= 3 and all(abs(float(v)) <= 1.0e-9 for v in eval_success_hist[-3:]):
        issues.append("eval_done_success_zero_x3")
        status = "ALERT"

    if (
        isinstance(eval_done_height_low_latest, (int, float))
        and math.isfinite(eval_done_height_low_latest)
        and eval_done_height_low_latest > 0.40
    ):
        issues.append("eval_done_height_low_high")
        status = "ALERT"

    if eval_gap_success is not None and eval_gap_success > 0.25:
        issues.append("eval_gap_success_high")
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
        "done_acc_limit": finite_or_none(latest.get("train/stats.done_acc_limit")),
        "train_done_height_low": finite_or_none(latest.get("train/stats.done_height_low")),
        "eval_done_success": finite_or_none(eval_done_success_latest),
        "eval_done_height_low": finite_or_none(eval_done_height_low_latest),
        "eval_done_safety": finite_or_none(latest_eval.get("eval/stats.done_safety")),
        "eval_gap_success": finite_or_none(eval_gap_success),
        "avg_speed": finite_or_none(latest.get("train/stats.avg_speed")),
        "action_abs_max": finite_or_none(latest.get("train/stats.action_abs_max")),
        "target_acc_abs_max": finite_or_none(latest.get("train/stats.target_acc_abs_max")),
        "target_acc_clip_ratio": finite_or_none(latest.get("train/stats.target_acc_clip_ratio")),
        "control_cmd_abs_max": finite_or_none(latest.get("train/stats.control_cmd_abs_max")),
        "control_cmd_clip_ratio": finite_or_none(latest.get("train/stats.control_cmd_clip_ratio")),
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
