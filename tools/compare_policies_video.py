"""Side-by-side policy comparison video with force / command / velocity arrows.

Runs several policies through the SAME scenario (same seed, same attitude target, same velocity
command, same esc cap) and renders them next to each other with 3-D arrows drawn into the scene:

  per-thruster thrust   red arrows at each pivot, length ∝ |force| — the wasted (null) thrust is
                        visible directly as up/down arrows that cancel out
  commanded velocity    green arrow at the CoM
  actual velocity       blue arrow at the CoM
  target heading        yellow arrow (target body +X)      actual heading  white arrow

A caption bar under each pane carries the live numbers (speed along command vs commanded, the
running null share of vertical power, mean |esc|), so tracking and wasted thrust can be compared
frame by frame.

    MUJOCO_GL=egl python -m tools.compare_policies_video \
        --policies "0825 (av_cal1)=models/av_cal1_best_rep103,new (av_mode13)=models/av_mode13" \
        --out /srv/share/policy_compare.mp4
"""
import argparse
import pickle
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import yaml
from PIL import Image, ImageDraw
from stable_baselines3 import PPO

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

_RGBA = {"thrust": (0.95, 0.20, 0.15, 0.9), "v_cmd": (0.20, 0.85, 0.30, 0.9),
         "v_act": (0.25, 0.55, 1.00, 0.9), "head_t": (1.00, 0.85, 0.10, 0.9),
         "head_a": (1.00, 1.00, 1.00, 0.9)}


def _arrow(scene, frm, to, rgba, width=0.008):
    """Append one arrow geom to a rendered scene (no-op if the scene is full)."""
    if scene.ngeom >= scene.maxgeom:
        return
    g = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3),
                        np.zeros(9), np.asarray(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, width,
                         np.asarray(frm, dtype=np.float64), np.asarray(to, dtype=np.float64))
    scene.ngeom += 1


_VERT_SIGNS = {"lf": (1, 1, 1), "lb": (1, -1, -1), "rb": (-1, -1, 1), "rf": (-1, 1, -1)}
# Where each unit sits when the vehicle is seen from above (nose up): (col, row).
_LAYOUT = {"lf": (0, 0), "rf": (1, 0), "lb": (0, 1), "rb": (1, 1)}


def unit_frame_forces(state, R, thrust_axes):
    """Per-unit thrust resolved into the unit's own (horizontal tangent, body-up) plane.

    This is the frame the allocation actually works in: the servo sweeps the thrust between the
    unit's tangent (h) and body-up (v), so (h, v) IS the azimuth+magnitude the operator sets.
    """
    up = R[:, 1]
    h, v = [], []
    for k, ax in enumerate(thrust_axes):
        f = np.asarray(state["thrust_world"][k], dtype=float)
        t = R @ (np.asarray(ax, dtype=float) / max(np.linalg.norm(ax), 1e-9))
        h.append(float(f @ t))
        v.append(float(f @ up))
    return np.array(h), np.array(v)


def draw_thruster_panel(d, x0, y0, w, h_px, names, h_comp, v_comp, f_cap):
    """Per-thruster azimuth dials (top-view layout) + the vertical-mode decomposition bars.

    The dials show what the 3-D arrows cannot: each unit's thrust ANGLE and MAGNITUDE side by
    side, in the vehicle's own layout. The null mode is the (+,-,+,-) checkerboard of the
    VERTICAL components, so it is visible as dials that alternate up/down around the square
    while the bar on the right shows how much of the vertical force it eats.
    """
    r, pad, lab_h = 30, 10, 14      # dial radius / margin / label strip under each dial
    for name, (cx, cy) in _LAYOUT.items():
        k = names.index(name)
        ox = x0 + pad + r + cx * (2 * r + 26)
        oy = y0 + pad + r + cy * (2 * r + lab_h + 8)
        d.ellipse([ox - r, oy - r, ox + r, oy + r], outline=(90, 90, 100))
        d.line([ox - r, oy, ox + r, oy], fill=(60, 60, 70))          # horizontal (tangent) axis
        d.line([ox, oy - r, ox, oy + r], fill=(60, 60, 70))          # vertical (body-up) axis
        mag = float(np.hypot(h_comp[k], v_comp[k]))
        scale = min(1.0, mag / max(f_cap, 1e-9)) * (r - 3)
        if mag > 1e-4:
            ang = np.arctan2(v_comp[k], h_comp[k])
            ex, ey = ox + scale * np.cos(ang), oy - scale * np.sin(ang)   # screen y is flipped
            col = (255, 90, 70) if abs(v_comp[k]) > abs(h_comp[k]) else (255, 170, 90)
            d.line([ox, oy, ex, ey], fill=col, width=3)
            d.ellipse([ex - 3, ey - 3, ex + 3, ey + 3], fill=col)
        d.text((ox - r, oy + r + 2), f"{name} {mag:4.1f}N", fill=(180, 180, 190))

    # vertical-mode decomposition: m = S^T v / 2 with S the Walsh columns (heave, roll, pitch, null)
    S = np.array([[1.0, *(float(s) for s in _VERT_SIGNS[n])] for n in names]).T / 2.0
    m = S @ v_comp
    bx = x0 + 2 * (2 * r + 26) + 26
    bw = w - (bx - x0) - 14
    labels = ("heave", "roll", "pitch", "NULL")
    for i, (lab, val) in enumerate(zip(labels, m)):
        by = y0 + 12 + i * 22
        d.text((bx, by), f"{lab:5s}", fill=(255, 120, 110) if i == 3 else (185, 185, 195))
        zone = max(40, bw - 44 - 56)          # bar zone, leaving room for the value text
        zx = bx + 44 + zone // 2
        d.line([zx, by - 2, zx, by + 14], fill=(80, 80, 92))
        px = int(np.clip(val / max(f_cap, 1e-9), -1, 1) * (zone // 2 - 2))
        col = (255, 80, 70) if i == 3 else (120, 200, 130)
        d.rectangle([min(zx, zx + px), by, max(zx, zx + px), by + 12], fill=col)
        d.text((bx + 44 + zone + 8, by), f"{val:+5.2f}N", fill=(200, 200, 210))
    return float(m[3])


def load_policy(mdir):
    mdir = Path(mdir)
    meta = yaml.safe_load((mdir / "meta.yaml").read_text())
    cfg = load_config(meta.get("config", "configs/train_ppo.yaml"))
    for k in ("task", "obs_mode", "proprio_mode", "obs_frame", "action_mode",
              "vel_cmd_cone_deg", "yaw_target_deg", "tilt_target_deg"):
        if meta.get(k) is not None:
            cfg["env"][k] = meta[k]
    cfg["env"]["observe_max_duty"] = bool(meta.get("observe_max_duty", False))
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg.setdefault("disturbance", {})["enabled"] = False
    model = PPO.load(str(mdir / "final.zip"), device="cpu")
    vn = pickle.load(open(mdir / "vecnormalize.pkl", "rb"))
    rms = vn.obs_rms

    def norm(o):
        return np.clip((o - rms.mean) / np.sqrt(rms.var + vn.epsilon),
                       -vn.clip_obs, vn.clip_obs).astype(np.float32)

    return model, norm, cfg


def rollout(mdir, cap, v_cmd, steps, size, seed):
    """Render one policy through the fixed scenario; returns (frames, caption lines)."""
    model, norm, cfg = load_policy(mdir)
    env = UmiusiPoseEnv(cfg)
    env.sim.max_duty = cap
    env._base["max_duty"] = cap
    obs, _ = env.reset(seed=seed)
    env.sim.max_duty = cap
    env._base["max_duty"] = cap
    env.target_quat = np.array([1.0, 0.0, 0.0, 0.0])          # hold level
    env.v_cmd = np.array([v_cmd, 0.0, 0.0])                   # steady forward command
    Rt = np.zeros(9)
    mujoco.mju_quat2Mat(Rt, env.target_quat)
    env.v_cmd_world = Rt.reshape(3, 3) @ env.v_cmd

    # unit pivots in ACTION order (the simulator keeps ids, the config keeps the coordinates)
    units_cfg = {u["name"]: u for u in env.sim.cfg["thrusters"]["units"]}
    pivots = [np.asarray(units_cfg[n]["pivot"], dtype=float) for n in env.sim.unit_names]

    f_cap_arrow = env.sim.thrust_per_cmd * cap ** env.sim.thrust_curve_exp
    renderer = mujoco.Renderer(env.sim.model, size[1], size[0])
    frames, caps = [], []
    nf_w, vp_w, esc_all = 0.0, 0.0, []
    for _ in range(steps):
        a, _ = model.predict(norm(obs), deterministic=True)
        obs, _r, term, trunc, info = env.step(a)
        st = env.sim.get_state()
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, st["quat"])
        R = R.reshape(3, 3)
        com = st["pos"]

        renderer.update_scene(env.sim.data, camera="track")
        sc = renderer.scene
        # per-thruster force (world frame), anchored at the unit pivots
        for k, pv in enumerate(pivots):
            pivot = com + R @ pv
            f = np.asarray(st["thrust_world"][k], dtype=float)
            if np.linalg.norm(f) > 1e-3:
                _arrow(sc, pivot, pivot + f * (0.32 / max(f_cap_arrow, 1e-9)),
                       _RGBA["thrust"], 0.012)
        top = com + np.array([0.0, 0.22, 0.0])
        _arrow(sc, top, top + env.v_cmd_world * 1.2, _RGBA["v_cmd"], 0.012)
        _arrow(sc, top, top + st["lin_vel"] * 1.2, _RGBA["v_act"], 0.012)
        nose = com + np.array([0.0, 0.12, 0.0])
        _arrow(sc, nose, nose + Rt.reshape(3, 3)[:, 0] * 0.35, _RGBA["head_t"], 0.008)
        _arrow(sc, nose, nose + R[:, 0] * 0.35, _RGBA["head_a"], 0.008)
        frames.append(renderer.render())

        nf_w += info["null_frac"] * info["vert_power"]
        vp_w += info["vert_power"]
        esc_all.extend(np.abs(info["esc_applied"]).tolist())
        vv = st["lin_vel"]
        nv, nc = float(np.linalg.norm(vv)), float(np.linalg.norm(env.v_cmd_world))
        cosang = float(vv @ env.v_cmd_world) / (nv * nc) if nv > 1e-4 and nc > 1e-6 else float("nan")
        hc, vc = unit_frame_forces(st, R, env.sim.thrust_axes)
        f_cap = env.sim.thrust_per_cmd * cap ** env.sim.thrust_curve_exp
        caps.append((info.get("vel_along", 0.0), info["ori_err"],
                     (nf_w / vp_w * 100 if vp_w > 1e-12 else 0.0), float(np.mean(esc_all)),
                     hc, vc, f_cap, list(env.sim.unit_names),
                     float(np.degrees(np.arctan2(2 * (st["quat"][0] * st["quat"][1]
                                                      + st["quat"][2] * st["quat"][3]),
                                                 1 - 2 * (st["quat"][1] ** 2 + st["quat"][2] ** 2)))),
                     cosang))
        if term or trunc:
            break
    renderer.close()
    env.close()
    return frames, caps


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--policies", required=True,
                    help='"label=dir,label=dir,..." (left to right)')
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=float, default=0.25)
    ap.add_argument("--v-cmd", type=float, default=0.10, help="commanded forward speed [m/s]")
    ap.add_argument("--seconds", type=float, default=12.0)
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=420)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    entries = [p.split("=", 1) for p in args.policies.split(",")]
    steps = int(args.seconds * 50)
    runs = []
    for label, mdir in entries:
        print(f"[video] rolling out {label} ({mdir})")
        runs.append((label, *rollout(mdir, args.cap, args.v_cmd, steps,
                                     (args.width, args.height), args.seed)))

    n = min(len(f) for _, f, _ in runs)
    bar, head, panel = 74, 30, 172
    out = []
    for i in range(n):
        panes = []
        for label, frames, caps in runs:
            along, ori, null_pw, esc, hc, vc, f_cap, names, roll_deg, cosang = caps[i]
            im = Image.new("RGB", (args.width, head + args.height + panel + bar), (16, 16, 20))
            im.paste(Image.fromarray(frames[i]), (0, head))
            d = ImageDraw.Draw(im)
            d.text((10, 8), label, fill=(255, 255, 255))
            m_null = draw_thruster_panel(d, 0, head + args.height, args.width, panel,
                                         names, hc, vc, f_cap)
            y = head + args.height + panel + 6
            ratio = (f"   ({along / args.v_cmd * 100:4.0f} %)" if args.v_cmd > 1e-9
                     else "   (hold station)")
            d.text((10, y), f"speed {along:+.3f} / cmd {args.v_cmd:.2f} m/s{ratio}",
                   fill=(150, 200, 255))
            ali = ("  n/a" if cosang != cosang
                   else f"{np.degrees(np.arccos(np.clip(cosang, -1, 1))):5.1f} deg")
            d.text((10, y + 16), f"ori err {ori:.3f} rad   angle(v_cmd, v_act) {ali}",
                   fill=(230, 230, 230))
            d.text((10, y + 32), f"null share {null_pw:4.1f} %   null amp {m_null:+.2f} N   "
                                 f"mean |esc| {esc:.3f}   roll {roll_deg:+.1f} deg",
                   fill=(255, 170, 160))
            panes.append(np.asarray(im))
        out.append(np.concatenate(panes, axis=1))

    legend = ("3D: red = per-thruster force, green = commanded velocity, blue = actual velocity, "
              "yellow = target heading, white = actual heading   |   dials = each unit's thrust in its "
              "(tangent, up) plane, top view   |   bars = vertical-mode split (NULL = pure waste)")
    canvas = Image.new("RGB", (out[0].shape[1], out[0].shape[0] + 26), (16, 16, 20))
    ImageDraw.Draw(canvas).text((10, 6), legend, fill=(200, 200, 200))
    base = np.asarray(canvas)
    final = [np.concatenate([base[:26], f], axis=0) for f in out]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(args.out, final, fps=25, quality=8)
    print(f"[video] wrote {args.out}  ({len(final)} frames, {len(runs)} panes)")


if __name__ == "__main__":
    main()
