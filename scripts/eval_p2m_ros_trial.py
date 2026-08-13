#!/usr/bin/env python3
import argparse
import json
import math
import time

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


class TrialMonitor:
    def __init__(self, start, goal, timeout, reach_goal_dis, collision_dis, stop_on_collision):
        self.start = start
        self.goal = goal
        self.timeout = timeout
        self.reach_goal_dis = reach_goal_dis
        self.collision_dis = collision_dis
        self.stop_on_collision = stop_on_collision
        self.start_time = None
        self.last_time = None
        self.last_pos = None
        self.path_len = 0.0
        self.speed_sum = 0.0
        self.speed_count = 0
        self.min_depth = float("inf")
        self.min_depth_raw = float("inf")
        self.collision_hits = 0
        self.success = False
        self.collision = False
        self.final_pos = None
        rospy.Subscriber("/sim/odom", Odometry, self.odom_cb, queue_size=20)
        rospy.Subscriber("/ray2array_hits", Float32MultiArray, self.lidar_cb, queue_size=5)

    def lidar_cb(self, msg):
        if self.start_time is None:
            return
        if self.last_pos is None:
            return
        data = [float(v) for v in msg.data if math.isfinite(v)]
        distances = []
        for idx in range(0, len(data) - 2, 3):
            hit = (data[idx], data[idx + 1], data[idx + 2])
            distances.append(math.dist(hit, self.last_pos))
        distances = [v for v in distances if math.isfinite(v) and v > 0.0]
        if distances:
            self.min_depth_raw = min(self.min_depth_raw, min(distances))
        filtered = [v for v in distances if v >= 0.05]
        if filtered:
            current_min = min(filtered)
            self.min_depth = min(self.min_depth, current_min)
            if current_min < self.collision_dis:
                self.collision_hits += 1
            else:
                self.collision_hits = 0
            if self.collision_hits >= 5:
                self.collision = True

    def odom_cb(self, msg):
        now = time.time()
        p = msg.pose.pose.position
        pos = (float(p.x), float(p.y), float(p.z))
        if self.start_time is None:
            self.start_time = now
        if self.last_pos is not None:
            dt = max(now - self.last_time, 1e-6)
            step = math.dist(pos, self.last_pos)
            self.path_len += step
            self.speed_sum += step / dt
            self.speed_count += 1
        self.last_pos = pos
        self.last_time = now
        self.final_pos = pos
        if math.dist(pos, self.goal) <= self.reach_goal_dis:
            self.success = True

    def run(self):
        deadline = time.time() + self.timeout
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self.success:
                break
            if self.stop_on_collision and self.collision:
                break
            rate.sleep()
        elapsed = 0.0 if self.start_time is None else time.time() - self.start_time
        straight = math.dist(self.start, self.goal)
        result = {
            "success": bool(self.success and not self.collision),
            "reached_goal": bool(self.success),
            "collision": bool(self.collision),
            "elapsed_s": elapsed,
            "avg_speed_mps": self.path_len / elapsed if elapsed > 1e-6 else 0.0,
            "path_len_m": self.path_len,
            "path_efficiency": self.path_len / straight if straight > 1e-6 else None,
            "min_safety_distance_m": self.min_depth if math.isfinite(self.min_depth) else None,
            "min_safety_distance_raw_m": self.min_depth_raw if math.isfinite(self.min_depth_raw) else None,
            "final_pos": self.final_pos,
            "start": self.start,
            "goal": self.goal,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", nargs=3, type=float, default=[0.0, -15.0, 2.0])
    parser.add_argument("--goal", nargs=3, type=float, default=[0.0, 15.0, 2.0])
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--reach-goal-dis", type=float, default=1.0)
    parser.add_argument("--collision-dis", type=float, default=0.3)
    parser.add_argument("--stop-on-collision", action="store_true")
    args = parser.parse_args()
    rospy.init_node("p2m_ros_trial_monitor", anonymous=True)
    monitor = TrialMonitor(
        tuple(args.start),
        tuple(args.goal),
        args.timeout,
        args.reach_goal_dis,
        args.collision_dis,
        args.stop_on_collision,
    )
    monitor.run()


if __name__ == "__main__":
    main()
