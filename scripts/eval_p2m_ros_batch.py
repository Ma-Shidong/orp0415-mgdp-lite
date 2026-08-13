#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
import signal
import socket
import subprocess
import time
from datetime import datetime
from pathlib import Path


REPO = Path("/home/csj/msd/orp0415/orp")
DEFAULT_OUT_ROOT = Path("/media/share/csj/msd/orp_eval")
DEFAULT_TMP = Path("/media/share/csj/msd/orp_tmp")

CHECKPOINTS = {
    "balanced_367067136": "/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_367067136.pt",
    "high_success_170459136": "/media/share/csj/msd/orp_runs/p2m_train_8000_from_52494336_20260715_111956/wandb/offline-run-20260715_112007-c3400gih/files/checkpoint_170459136.pt",
}

SCENARIOS = {
    "dyn7_static7": {"label": "7 dynamic + 7 static", "static": 7, "dynamic": 7, "p2m_eta": 95.0, "best_non_p2m_eta": 90.0},
    "dyn13_static13": {"label": "13 dynamic + 13 static", "static": 13, "dynamic": 13, "p2m_eta": 65.0, "best_non_p2m_eta": 30.0},
    "dyn25_static19": {"label": "25 dynamic + 19 static", "static": 19, "dynamic": 25, "p2m_eta": 40.0, "best_non_p2m_eta": 25.0},
    "dyn25_static0": {"label": "25 dynamic only", "static": 0, "dynamic": 25, "p2m_eta": 50.0, "best_non_p2m_eta": 25.0},
    "dyn0_static44": {"label": "44 static only", "static": 44, "dynamic": 0, "p2m_eta": 60.0, "best_non_p2m_eta": 80.0},
}


def parse_seeds(text):
    if ":" in text:
        start, end = [int(x) for x in text.split(":", 1)]
        return list(range(start, end + 1))
    return [int(x) for x in text.split(",") if x.strip()]


def bash_env():
    return (
        "source /opt/ros/noetic/setup.bash\n"
        "source /home/csj/anaconda3/etc/profile.d/conda.sh\n"
        "conda activate orp\n"
        f"export PYTHONPATH={REPO}:$PYTHONPATH\n"
    )


def start_process(name, cmd, log_path, env):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "w", buffering=1)
    proc = subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=str(REPO),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        text=True,
    )
    return {"name": name, "proc": proc, "log_file": log_file, "log_path": log_path}


def stop_process(item, timeout=5):
    proc = item["proc"]
    if proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=timeout)
        except Exception:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                pass
    item["log_file"].close()


def wait_ros_master(port, timeout=20):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", int(port)), timeout=1.0):
                return True
        except OSError:
            pass
        time.sleep(0.5)
    return False


def load_result(path):
    text = path.read_text(errors="replace")
    matches = re.findall(r"\{[\s\S]*\}", text)
    if not matches:
        return None
    for candidate in reversed(matches):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def mean(values):
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def run_trial(args, checkpoint_name, checkpoint_path, scenario_name, scenario, seed, run_dir):
    trial_name = f"{checkpoint_name}__{scenario_name}__seed{seed:03d}"
    trial_dir = run_dir / "trials" / trial_name
    result_path = trial_dir / "result.json"
    if args.skip_existing and result_path.exists():
        result = json.loads(result_path.read_text())
        result["skipped_existing"] = True
        return result

    env = os.environ.copy()
    env.update(
        {
            "ROS_MASTER_URI": f"http://localhost:{args.ros_port}",
            "ROS_IP": "127.0.0.1",
            "SIM_DEVICE": args.device,
            "TMPDIR": str(DEFAULT_TMP),
        }
    )
    DEFAULT_TMP.mkdir(parents=True, exist_ok=True)

    commands = {
        "roscore": f"source /opt/ros/noetic/setup.bash && roscore -p {args.ros_port}",
        "map": (
            "source /opt/ros/noetic/setup.bash\n"
            f"export ROS_MASTER_URI=http://localhost:{args.ros_port}\n"
            "export ROS_IP=127.0.0.1\n"
            f"{REPO}/devel/lib/map_generator/dynamic_env "
            "_map/x_size:=10 _map/y_size:=25 _map/z_size:=5 _map/resolution:=0.1 "
            f"_ObstacleShape/seed:={seed} "
            "_map/obs_vel_l:=0.34 _map/obs_vel_h:=1.68 "
            f"_map/obs_num:={scenario['static']} _map/moving_obs_num:={scenario['dynamic']} "
            "_ObstacleShape/sobs_lower_width:=0.5 _ObstacleShape/sobs_upper_width:=0.7 "
            "_ObstacleShape/dobs_lower_width:=0.4 _ObstacleShape/dobs_upper_width:=0.5 "
            "_ObstacleShape/lower_hei:=3.5 _ObstacleShape/upper_hei:=4.0 "
            "_ObstacleShape/radius_l:=0.5 _ObstacleShape/radius_h:=0.7 "
            "_ObstacleShape/z_l:=0.7 _ObstacleShape/z_h:=0.8 "
            "_pub_rate:=50.0 _min_distance:=0.8"
        ),
        "lidar": (
            "source /opt/ros/noetic/setup.bash\n"
            f"source {REPO}/devel/.private/lidar/setup.bash\n"
            f"export ROS_MASTER_URI=http://localhost:{args.ros_port}\n"
            "export ROS_IP=127.0.0.1\n"
            f"export ROS_PACKAGE_PATH={REPO}/src:{REPO}/src/lidar:{REPO}/src/uav_simulator:{REPO}/src/utils:/opt/ros/noetic/share\n"
            f"roslaunch {REPO}/src/lidar/launch/scanner.launch"
        ),
        "monitor": (
            "source /opt/ros/noetic/setup.bash\n"
            f"export ROS_MASTER_URI=http://localhost:{args.ros_port}\n"
            f"python3 {REPO}/scripts/eval_p2m_ros_trial.py "
            f"--timeout {args.timeout} --start 0 -15 2 --goal 0 15 2 "
            f"--reach-goal-dis {args.reach_goal_dis} --collision-dis {args.collision_dis} "
            "--stop-on-collision"
        ),
        "infer": (
            bash_env()
            + f"export ROS_MASTER_URI=http://localhost:{args.ros_port}\n"
            "export ROS_IP=127.0.0.1\n"
            f"export SIM_DEVICE={args.device}\n"
            f"export TMPDIR={DEFAULT_TMP}\n"
            f"cd {REPO}/scripts\n"
            "python infer.py "
            f"task.ckpt_path={checkpoint_path} "
            f"task.input_mode={args.input_mode} "
            "task.lidar_v_sample=2 "
            "task.start_xyz='[0.0,-15.0,2.0]' "
            "task.default_goal_xyz='[0.0,15.0,2.0]' "
            "task.auto_test=true "
            "task.auto_test_toggle=false "
            "task.stop_when_reach_goal=false "
            f"task.reach_goal_dis={args.reach_goal_dis} "
            "task.collision_stop=true "
            "task.safety_shield_enable=true"
        ),
    }

    if args.dry_run:
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "commands.json").write_text(json.dumps(commands, indent=2, ensure_ascii=False))
        return {
            "checkpoint": checkpoint_name,
            "scenario": scenario_name,
            "scenario_label": scenario["label"],
            "input_mode": args.input_mode,
            "seed": seed,
            "dry_run": True,
        }

    procs = []
    try:
        procs.append(start_process("roscore", commands["roscore"], trial_dir / "roscore.log", env))
        if not wait_ros_master(args.ros_port):
            raise RuntimeError("ROS master did not become ready")
        procs.append(start_process("map", commands["map"], trial_dir / "map.log", env))
        time.sleep(args.startup_sleep)
        procs.append(start_process("lidar", commands["lidar"], trial_dir / "lidar.log", env))
        time.sleep(args.startup_sleep)
        monitor = start_process("monitor", commands["monitor"], trial_dir / "monitor.log", env)
        procs.append(monitor)
        time.sleep(1.0)
        infer = start_process("infer", commands["infer"], trial_dir / "infer.log", env)
        procs.append(infer)
        monitor["proc"].wait(timeout=args.timeout + 15)
        result = load_result(trial_dir / "monitor.log")
        if result is None:
            result = {"success": False, "error": "no monitor JSON result"}
    except subprocess.TimeoutExpired:
        result = {"success": False, "error": "trial timeout"}
    except Exception as exc:
        result = {"success": False, "error": str(exc)}
    finally:
        for item in reversed(procs):
            stop_process(item)

    result.update(
        {
            "checkpoint": checkpoint_name,
            "checkpoint_path": checkpoint_path,
            "scenario": scenario_name,
            "scenario_label": scenario["label"],
            "input_mode": args.input_mode,
            "seed": seed,
            "static_obstacles": scenario["static"],
            "dynamic_obstacles": scenario["dynamic"],
        }
    )
    trial_dir.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def write_outputs(run_dir, rows):
    csv_path = run_dir / "results.csv"
    fields = [
        "checkpoint",
        "scenario",
        "scenario_label",
        "input_mode",
        "seed",
        "success",
        "reached_goal",
        "collision",
        "elapsed_s",
        "avg_speed_mps",
        "path_efficiency",
        "min_safety_distance_m",
        "min_safety_distance_raw_m",
        "error",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summary = []
    for checkpoint in sorted({r["checkpoint"] for r in rows}):
        for scenario in sorted({r["scenario"] for r in rows}):
            subset = [r for r in rows if r["checkpoint"] == checkpoint and r["scenario"] == scenario]
            if not subset:
                continue
            successful = [r for r in subset if r.get("success")]
            info = SCENARIOS[scenario]
            eta = 100.0 * len(successful) / len(subset)
            summary.append(
                {
                    "checkpoint": checkpoint,
                    "scenario": scenario,
                    "scenario_label": info["label"],
                    "n": len(subset),
                    "success_rate_pct": eta,
                    "avg_speed_success_mps": mean([r.get("avg_speed_mps") for r in successful]),
                    "path_eff_success": mean([r.get("path_efficiency") for r in successful]),
                    "safety_success_m": mean([r.get("min_safety_distance_m") for r in successful]),
                    "p2m_eta_pct": info["p2m_eta"],
                    "best_non_p2m_eta_pct": info["best_non_p2m_eta"],
                }
            )

    summary_path = run_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()) if summary else [])
        if summary:
            writer.writeheader()
            writer.writerows(summary)

    md = ["# P2M 批量评测结果", "", f"生成时间：`{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`", ""]
    md.append("| Checkpoint | 场景 | N | 成功率 | P2M论文 | 最强非P2M | 成功样本速度 | 成功样本路径效率 | 成功样本安全距离 |")
    md.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in summary:
        md.append(
            "| {checkpoint} | {scenario_label} | {n} | {success_rate_pct:.1f}% | {p2m_eta_pct:.1f}% | {best_non_p2m_eta_pct:.1f}% | {speed} | {path} | {safety} |".format(
                **r,
                speed="-" if r["avg_speed_success_mps"] is None else f"{r['avg_speed_success_mps']:.2f}",
                path="-" if r["path_eff_success"] is None else f"{r['path_eff_success']:.2f}",
                safety="-" if r["safety_success_m"] is None else f"{r['safety_success_m']:.2f}",
            )
        )
    (run_dir / "summary.md").write_text("\n".join(md) + "\n")
    return csv_path, summary_path, run_dir / "summary.md"


def main():
    parser = argparse.ArgumentParser(description="Run paper-style P2M ROS benchmark trials.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoints", default="balanced_367067136,high_success_170459136")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS.keys()))
    parser.add_argument("--seeds", default="0:19")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--reach-goal-dis", type=float, default=1.0)
    parser.add_argument("--collision-dis", type=float, default=0.3)
    parser.add_argument("--ros-port", type=int, default=11312)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-mode", default="p2m", choices=["p2m", "mgdp_lite", "mgdp_lite_v2"])
    parser.add_argument("--startup-sleep", type=float, default=2.0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected_checkpoints = [x.strip() for x in args.checkpoints.split(",") if x.strip()]
    selected_scenarios = [x.strip() for x in args.scenarios.split(",") if x.strip()]
    seeds = parse_seeds(args.seeds)
    for name in selected_checkpoints:
        if name not in CHECKPOINTS:
            raise SystemExit(f"Unknown checkpoint name: {name}")
        if not Path(CHECKPOINTS[name]).is_file():
            raise SystemExit(f"Checkpoint not found: {CHECKPOINTS[name]}")
    for name in selected_scenarios:
        if name not in SCENARIOS:
            raise SystemExit(f"Unknown scenario name: {name}")

    run_name = args.run_name or f"p2m_paper_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = Path(args.out_root) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, ensure_ascii=False))

    rows = []
    total = len(selected_checkpoints) * len(selected_scenarios) * len(seeds)
    index = 0
    for checkpoint_name in selected_checkpoints:
        for scenario_name in selected_scenarios:
            for seed in seeds:
                index += 1
                print(f"[{index}/{total}] {checkpoint_name} {scenario_name} seed={seed}", flush=True)
                row = run_trial(
                    args,
                    checkpoint_name,
                    CHECKPOINTS[checkpoint_name],
                    scenario_name,
                    SCENARIOS[scenario_name],
                    seed,
                    run_dir,
                )
                rows.append(row)
                write_outputs(run_dir, rows)

    csv_path, summary_path, md_path = write_outputs(run_dir, rows)
    print(f"results: {csv_path}")
    print(f"summary: {summary_path}")
    print(f"markdown: {md_path}")


if __name__ == "__main__":
    main()
