"""Export a pool-test rosbag (mcap) to the npz layout that tools/bag_replay.py loads.

Run with the ROS python (NOT the uv env):
    source /opt/ros/jazzy/setup.bash && source ros2_ws/install/setup.bash
    python3 tools/bag_export_npz.py <bagdir> <out.npz> [--grid-hz 50] [--angle-unit rad]

Command source. The 2026-08-21 export (docs/calibration_plan.md) read the RL-path topics
``/cmd/direct/thruster_controller/output_{lf,lb,rb,rf}``. The C++ control path (2026-10-03 pool)
publishes no such topics: the final per-thruster command is the thruster_controller state echoed
by gate_controller in ``/state/thruster_state_all`` — ``commanded_angle`` [rad, LOGICAL sign,
+ = up; taken BEFORE the CAN-boundary flip of control 1d48d60] and ``duty_cycle`` [post
rate-limit / max_duty clamp / is_forward sign — the value that reaches the VESC, same quantity as
the 8/21 ``output_*`` duty]. Default is that topic; ``--cmd-prefix`` switches back to the
RL-path topics when they exist.

Units. bag_replay divides cmd angles by 90 (deg). The bag angle is rad (control >= 2026-09-14),
so this exporter CONVERTS TO DEG on output (``--angle-unit`` names the INPUT unit); npz key
``meta_angle_unit`` records the output unit.

Time base. bag_replay assumes 1 cmd tick = 1 sim control step = 1/50 s (the 8/21 bag was 50 Hz).
The 10/03 bags publish IMU / thruster state at ~30-48 Hz, so ``--grid-hz`` (default 50)
resamples commands (zero-order hold) and the IMU (linear interp of gyro/acc, nlerp of quat) onto
a uniform grid. ``--grid-hz 0`` keeps native timestamps. The native series are always kept under
``raw_*`` keys. Receive time (bag timestamp) is used throughout, as in the 8/21 export.

Extra keys: ``imu_acc`` (sensor linear_acceleration), ``thr_*`` (echo: angle [deg], est angle,
rpm, duty), ``robot_state_t/robot_state`` (RobotState: -1 off, 0 standby, 1 manual, 2 auto,
3 debug), ``runnable_t/runnable`` (ThrusterRunnableAll as (N, 4, 2) [esc, servo]).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

POS = ("lf", "lb", "rb", "rf")


def read_bag(bag_dir, topics):
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="mcap"),
                rosbag2_py.ConverterOptions("", ""))
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    present = [t for t in topics if t in types]
    missing = [t for t in topics if t not in types]
    reader.set_filter(rosbag2_py.StorageFilter(topics=present))
    while reader.has_next():
        topic, data, t = reader.read_next()
        yield topic, deserialize_message(data, get_message(types[topic])), t * 1e-9
    if missing:
        print(f"  (topics absent from bag: {missing})")


def zoh(ts, vals, grid):
    """Zero-order hold: value of the latest sample at or before each grid time."""
    idx = np.searchsorted(ts, grid, side="right") - 1
    return vals[np.clip(idx, 0, len(ts) - 1)]


def lerp_rows(ts, vals, grid):
    return np.stack([np.interp(grid, ts, vals[:, k]) for k in range(vals.shape[1])], axis=1)


def nlerp_quat(ts, q, grid):
    q = q.copy()
    for i in range(1, len(q)):          # keep neighbours in the same hemisphere
        if np.dot(q[i], q[i - 1]) < 0:
            q[i] = -q[i]
    out = lerp_rows(ts, q, grid)
    return out / np.linalg.norm(out, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("bag_dir")
    ap.add_argument("out_npz")
    ap.add_argument("--imu-topic", default="/state/imu")
    ap.add_argument("--thr-topic", default="/state/thruster_state_all",
                    help="ThrusterStateAll echo; also the command source unless --cmd-prefix")
    ap.add_argument("--cmd-prefix", default=None,
                    help="RL-path ThrusterOutput topics '<prefix>{lf,lb,rb,rf}' "
                         "(e.g. /cmd/direct/thruster_controller/output_); default: --thr-topic")
    ap.add_argument("--robot-state-topic", default="/robot_state")
    ap.add_argument("--runnable-topic", default="/cmd/thruster_runnable_all")
    ap.add_argument("--angle-unit", choices=("rad", "deg"), default="rad",
                    help="unit of the angle IN THE BAG (output is always deg)")
    ap.add_argument("--grid-hz", type=float, default=50.0,
                    help="resample cmd (ZOH) and IMU (interp) onto this grid; 0 = native")
    args = ap.parse_args()

    to_deg = np.degrees if args.angle_unit == "rad" else (lambda x: x)
    cmd_topics = ([args.cmd_prefix + p for p in POS] if args.cmd_prefix else [])
    topics = [args.imu_topic, args.thr_topic, args.robot_state_topic, args.runnable_topic] + cmd_topics

    raw = {k: [] for k in ("imu_t", "imu_quat", "imu_gyro", "imu_acc", "thr_t", "thr_angle",
                           "thr_est_angle", "thr_rpm", "thr_duty", "robot_state_t", "robot_state",
                           "runnable_t", "runnable")}
    for p in POS:
        raw[f"cmd_{p}_t"], raw[f"cmd_{p}_angle"], raw[f"cmd_{p}_duty"] = [], [], []

    print(f"reading {args.bag_dir}")
    for topic, m, t in read_bag(args.bag_dir, topics):
        if topic == args.imu_topic:
            q, g, a = m.orientation, m.angular_velocity, m.linear_acceleration
            raw["imu_t"].append(t)
            raw["imu_quat"].append([q.w, q.x, q.y, q.z])
            raw["imu_gyro"].append([g.x, g.y, g.z])
            raw["imu_acc"].append([a.x, a.y, a.z])
        elif topic == args.thr_topic:
            s = [getattr(m, p) for p in POS]
            raw["thr_t"].append(t)
            raw["thr_angle"].append([to_deg(x.commanded_angle) for x in s])
            raw["thr_est_angle"].append([to_deg(x.estimated_angle) for x in s])
            raw["thr_rpm"].append([x.rpm for x in s])
            raw["thr_duty"].append([x.duty_cycle for x in s])
            if not args.cmd_prefix:
                for p, x in zip(POS, s):
                    raw[f"cmd_{p}_t"].append(t)
                    raw[f"cmd_{p}_angle"].append(to_deg(x.commanded_angle))
                    raw[f"cmd_{p}_duty"].append(x.duty_cycle)
        elif topic in cmd_topics:
            p = topic[len(args.cmd_prefix):]
            raw[f"cmd_{p}_t"].append(t)
            raw[f"cmd_{p}_angle"].append(to_deg(m.angle))
            raw[f"cmd_{p}_duty"].append(m.duty_cycle)
        elif topic == args.robot_state_topic:
            raw["robot_state_t"].append(t)
            raw["robot_state"].append(m.state)
        elif topic == args.runnable_topic:
            raw["runnable_t"].append(t)
            raw["runnable"].append([[getattr(m, p).esc, getattr(m, p).servo] for p in POS])

    raw = {k: np.array(v) for k, v in raw.items()}
    out = {}
    n_imu, n_cmd = len(raw["imu_t"]), len(raw["cmd_lf_t"])
    if n_imu == 0 or n_cmd == 0:
        raise SystemExit(f"nothing to export: imu {n_imu} msgs, cmd {n_cmd} msgs")

    if args.grid_hz > 0:
        # Chronological order is per-topic in the bag; sort defensively before resampling.
        oi, oc = np.argsort(raw["imu_t"]), np.argsort(raw["cmd_lf_t"])
        t0 = max(raw["imu_t"][oi][0], raw["cmd_lf_t"][oc][0])
        t1 = min(raw["imu_t"][oi][-1], raw["cmd_lf_t"][oc][-1])
        grid = t0 + np.arange(0, t1 - t0, 1.0 / args.grid_hz)
        out["imu_t"] = grid
        out["imu_quat"] = nlerp_quat(raw["imu_t"][oi], raw["imu_quat"][oi], grid)
        out["imu_gyro"] = lerp_rows(raw["imu_t"][oi], raw["imu_gyro"][oi], grid)
        out["imu_acc"] = lerp_rows(raw["imu_t"][oi], raw["imu_acc"][oi], grid)
        for p in POS:
            o = np.argsort(raw[f"cmd_{p}_t"])
            ts = raw[f"cmd_{p}_t"][o]
            out[f"cmd_{p}_t"] = grid
            out[f"cmd_{p}_angle"] = zoh(ts, raw[f"cmd_{p}_angle"][o], grid)
            out[f"cmd_{p}_duty"] = zoh(ts, raw[f"cmd_{p}_duty"][o], grid)
        for k, v in raw.items():
            out[f"raw_{k}"] = v
        for k in ("thr_t", "thr_angle", "thr_est_angle", "thr_rpm", "thr_duty",
                  "robot_state_t", "robot_state", "runnable_t", "runnable"):
            out[k] = raw[k]
    else:
        out.update(raw)

    out["meta_angle_unit"] = np.array("deg")
    out["meta_cmd_source"] = np.array(args.cmd_prefix or args.thr_topic)
    out["meta_grid_hz"] = np.array(args.grid_hz)
    out["meta_bag"] = np.array(str(Path(args.bag_dir).name))
    Path(args.out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out_npz, **out)

    dt_imu = np.median(np.diff(np.sort(raw["imu_t"]))) if n_imu > 1 else float("nan")
    dt_cmd = np.median(np.diff(np.sort(raw["cmd_lf_t"]))) if n_cmd > 1 else float("nan")
    print(f"wrote {args.out_npz}")
    print(f"  imu {n_imu} msgs (median dt {dt_imu * 1e3:.1f} ms), cmd {n_cmd} msgs "
          f"(median dt {dt_cmd * 1e3:.1f} ms), grid {len(out['cmd_lf_t'])} ticks @ {args.grid_hz} Hz")
    ang = np.stack([raw[f"cmd_{p}_angle"] for p in POS], 1)
    duty = np.stack([raw[f"cmd_{p}_duty"] for p in POS], 1)
    print(f"  cmd angle [deg] min {ang.min(0).round(1)} max {ang.max(0).round(1)}; "
          f"|duty| max {np.abs(duty).max(0).round(3)}; duty!=0 fraction {np.mean(np.any(duty != 0, 1)):.3f}")
    if len(raw["robot_state"]):
        vals, cnt = np.unique(raw["robot_state"], return_counts=True)
        print(f"  robot_state histogram {dict(zip(vals.tolist(), cnt.tolist()))}")


if __name__ == "__main__":
    main()
