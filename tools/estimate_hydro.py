"""Estimate the Fossen-style hydrodynamic coefficients of UMIUSI from CAD geometry.

The values in ``configs/umiusi.yaml`` marked PLACEHOLDER (drag, added mass) were guesses.
This tool replaces the guessing with a documented, reproducible derivation:

  1. **Silhouette (projected) areas** are rasterised from the CAD STLs
     (``umiusi_model/STL/{base_link,thruster_link}.stl``) — the true open-frame areas, not the
     bounding box. Falls back to the bbox x solidity assumption when the STLs are absent
     (they are gitignored), so the tool still runs on a clean checkout.
  2. **Quadratic translational drag** ``D_q[i] = 1/2 rho Cd_i A_i`` with EFFECTIVE Cd taken from
     the BlueROV2 benchmark model (von Benzon et al. 2022, JMSE 10(12):1898, Table A1), whose
     coefficients were identified experimentally in a pool for a vehicle of the same class and
     mass (13.5 kg vs UMIUSI 12.5 kg) and normalised against the same kind of silhouette area.
     Using a *measured* effective Cd rather than a textbook box Cd folds in the frame-interference
     and internal-member drag that a silhouette area misses.
  3. **Quadratic rotational drag** by strip integration over the same silhouettes:
     ``M_k = 1/2 rho [ Cd_j Int |x_i|^3 a_j(x_i) dx_i + Cd_i Int |x_j|^3 a_i(x_j) dx_j ]``
     where ``a_j(x_i)`` is the distribution along ``x_i`` of the silhouette area normal to ``j``.
     Moments are taken about the whole-vehicle CoM.
  4. **Linear drag** is set from the BlueROV2 linear/quadratic ratio per axis (it is small: at the
     0.4 m/s cruise the quadratic term dominates) — a placeholder that the tow test replaces.
  5. **Added mass** from Lamb's ellipsoid solution for the enveloping ellipsoid, scaled by the
     per-axis solidity (silhouette / bbox face). The same recipe is run against the BlueROV2
     geometry and compared with its published added mass, so the residual error of the method is
     reported rather than hidden.

Everything is printed with its provenance and an uncertainty band. Nothing here replaces the
in-water calibration (see docs/calibration_plan.md) — these are defensible INITIAL values.

Usage:
    python -m tools.estimate_hydro              # report
    python -m tools.estimate_hydro --emit-yaml  # also print a drop-in configs/umiusi.yaml block
"""

import argparse
from pathlib import Path

import numpy as np
import yaml
from scipy.integrate import quad

_ROOT = Path(__file__).resolve().parents[1]
_STL = _ROOT / "umiusi_model" / "STL"
_CONFIG = _ROOT / "configs" / "umiusi.yaml"

# Body-axis index convention used throughout the sim: [x, y, z] = [surge, heave(up), sway].
AX = {"x": 0, "y": 1, "z": 2}
AXIS_NAME = ("x/surge", "y/heave", "z/sway")
# Rotational axes, in the sim's own order: index 3 = roll (about +X), 4 = yaw (about +Y),
# 5 = pitch (about +Z).  NOTE this is *not* the [roll, pitch, yaw] the old config comment claimed.
ROT_NAME = ("roll (about +X)", "yaw (about +Y)", "pitch (about +Z)")

# --- BlueROV2 heavy reference (von Benzon et al. 2022, JMSE 10(12):1898) ---------------------
# Fossen axes there: x surge, y sway, z heave (z down). Converted to UMIUSI's [x, y(up), z] below.
BLUEROV2 = {
    "mass": 13.5, "volume": 0.0134,
    "dims_LHW": (0.46, 0.38, 0.58),      # length(x) x height(z) x width(y)
    "A_proj": {"surge": 0.0877, "sway": 0.1131, "heave": 0.2049},   # m^2 (their Au, Av, Aw)
    "added_mass": {"surge": 6.36, "sway": 7.12, "heave": 18.68,     # kg
                   "roll": 0.189, "pitch": 0.135, "yaw": 0.222},    # kg m^2
    "drag_lin": {"surge": 13.7, "sway": 0.0, "heave": 33.0, "roll": 0.0, "pitch": 0.8, "yaw": 0.0},
    "drag_quad": {"surge": 141.0, "sway": 217.0, "heave": 190.0,
                  "roll": 1.19, "pitch": 0.47, "yaw": 1.5},
}


# =============================================================================================
# STL silhouettes
# =============================================================================================
def read_binary_stl(path):
    """(N, 3, 3) float64 vertices in metres from a binary STL stored in millimetres."""
    with open(path, "rb") as f:
        f.read(80)
        n = int(np.frombuffer(f.read(4), dtype="<u4")[0])
        rec = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
        d = np.frombuffer(f.read(n * rec.itemsize), dtype=rec, count=n)
    return d["v"].astype(np.float64) * 1e-3


def silhouette(tris, axis, cell=1e-3):
    """Rasterise the projection along `axis` onto a `cell`-metre grid.

    Returns (mask, origin, cell) with mask[i, j] True where the body covers the cell whose
    lower corner is origin + (i, j) * cell in the two remaining (ordered) axes. Each triangle
    is sampled at its vertices, edge midpoints and centroid; with ~1 mm CAD triangles on a 1 mm
    grid that covers the silhouette (a closed surface projects to a filled region).
    """
    keep = [i for i in range(3) if i != axis]
    p = tris[:, :, keep]
    pts = np.concatenate([p[:, 0], p[:, 1], p[:, 2],
                          (p[:, 0] + p[:, 1]) / 2, (p[:, 1] + p[:, 2]) / 2, (p[:, 2] + p[:, 0]) / 2,
                          p.mean(1)], axis=0)
    lo, hi = pts.min(0), pts.max(0)
    n = np.ceil((hi - lo) / cell).astype(int) + 1
    idx = ((pts - lo) / cell).astype(np.int64)
    mask = np.zeros(tuple(n), dtype=bool)
    mask[idx[:, 0], idx[:, 1]] = True
    return mask, lo, cell


def area_of(mask, cell):
    return float(mask.sum()) * cell * cell


def strip_profile(mask, lo, cell, along):
    """Area density a(s) [m] of a silhouette along one of its two in-plane coordinates.

    `along` is 0 or 1 (which of the mask's two axes to keep). Returns (s_centres, a) with
    sum(a) * cell == the silhouette area.
    """
    counts = mask.sum(axis=1 - along)
    s = lo[along] + (np.arange(mask.shape[along]) + 0.5) * cell
    return s, counts * cell            # [m], [m] (area per unit length)


# =============================================================================================
# Lamb ellipsoid added mass
# =============================================================================================
def lamb_coefficients(a, b, c):
    """Lamb's alpha0, beta0, gamma0 for an ellipsoid with semi-axes a, b, c."""
    def _int(p, q, r):
        def f(lam):
            return 1.0 / ((a * a + lam) ** p * (b * b + lam) ** q * (c * c + lam) ** r)
        return a * b * c * quad(f, 0.0, np.inf, limit=400)[0]
    return _int(1.5, 0.5, 0.5), _int(0.5, 1.5, 0.5), _int(0.5, 0.5, 1.5)


def ellipsoid_added_mass(a, b, c, rho=1000.0):
    """Added mass (3 translational [kg]) and added inertia (3 rotational [kg m^2]) of an
    ellipsoid with semi-axes a (about x), b (y), c (z), from Lamb's classical solution."""
    al, be, ga = lamb_coefficients(a, b, c)
    md = rho * (4.0 / 3.0) * np.pi * a * b * c
    trans = np.array([al / (2 - al), be / (2 - be), ga / (2 - ga)]) * md

    def _rot(u, v, k_u, k_v):
        num = (u * u - v * v) ** 2 * (k_v - k_u)
        den = 2 * (u * u - v * v) + (u * u + v * v) * (k_u - k_v)
        return md / 5.0 * num / den if abs(den) > 1e-12 else 0.0

    rot = np.array([_rot(b, c, be, ga), _rot(c, a, ga, al), _rot(a, b, al, be)])
    return trans, np.abs(rot)


def added_mass_translational(dims, solidity, rho=1000.0):
    """Enveloping-ellipsoid (Lamb) translational added mass scaled by the per-axis solidity.

    `dims` = full extents (Lx, Ly, Lz); `solidity[i]` = silhouette_i / bbox_face_i in [0, 1].
    RAW estimate — apply `AM_TRANS_CAL` (the BlueROV2 method-check ratio) before use.
    """
    a, b, c = np.asarray(dims, dtype=float) / 2.0
    trans, _ = ellipsoid_added_mass(a, b, c, rho)
    return trans * np.asarray(solidity, dtype=float)


def added_mass_rotational_rod(m_trans, dims):
    """Rotational added inertia from the ROD approximation: distribute each translational added
    mass uniformly along the body and sum the two contributions per rotation axis,
    I_k = m_j * L_i^2/12 + m_i * L_j^2/12.  RAW — apply `AM_ROT_CAL` before use.

    Axes here are the sim's: [x fwd, y up(H), z lateral(W)]; rotations [roll +X, yaw +Y, pitch +Z].
    """
    L, H, W = dims
    mx, my, mz = m_trans
    roll = mz * H * H / 12 + my * W * W / 12
    yaw = mx * W * W / 12 + mz * L * L / 12
    pitch = mx * H * H / 12 + my * L * L / 12
    return np.array([roll, yaw, pitch])


# Method-check calibration: the SAME recipes run on the BlueROV2 geometry, divided by its
# published (experimentally-anchored) Table A1 values. Computed by `_method_check()` below and
# frozen here so the correction is explicit. Per-axis: [x, y(heave/up), z] / [roll, yaw, pitch].
AM_TRANS_CAL = np.array([1.66, 1.56, 1.61])   # ellipsoid x solidity overestimates by ~1.6
AM_ROT_CAL = np.array([3.22, 1.37, 3.01])     # rod formula overestimate: [roll, yaw, pitch]


def added_mass_estimate(dims, solidity, rho=1000.0):
    """BlueROV2-calibrated added mass: translational (Lamb x solidity / cal) and rotational
    (rod distribution of the calibrated translational masses / cal)."""
    trans = added_mass_translational(dims, solidity, rho) / AM_TRANS_CAL
    rot = added_mass_rotational_rod(trans, dims) / AM_ROT_CAL
    return trans, rot


# =============================================================================================
# Estimation
# =============================================================================================
def rotational_quadratic_drag(profiles, areas, cd, com, rho=1000.0):
    """Strip-integrated quadratic rotational damping about the CoM.

    profiles[(sil_axis, along_axis)] -> (s, a(s)) as returned by strip_profile, where `sil_axis`
    is the projection axis (so the area is normal to it) and `along_axis` the coordinate the
    strips are indexed by. Returns [K_roll(+X), N_yaw(+Y), M_pitch(+Z)] in N*m/(rad/s)^2.
    """
    def term(sil_axis, along_axis):
        s, a = profiles[(sil_axis, along_axis)]
        r = np.abs(s - com[along_axis])
        return 0.5 * rho * cd[sil_axis] * float(np.sum(r ** 3 * a)) * (s[1] - s[0])

    # rotation about k: strips along i present the j-normal area, and vice versa.
    out = []
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):   # roll(x): (y,z); yaw(y): (z,x); pitch(z): (x,y)
        out.append(term(j, i) + term(i, j))
    return np.array(out)


def report():
    cfg = yaml.safe_load(_CONFIG.read_text())
    rho = float(cfg["water"]["density"])
    hull_mass = float(cfg["hull"]["mass"])
    thr_mass = float(cfg["thrusters"]["mass"])
    total_mass = hull_mass + 4 * thr_mass
    dims = np.array(cfg["hull"]["half_extents"], dtype=float) * 2.0

    lines = []
    P = lines.append
    P("=" * 88)
    P("UMIUSI hydrodynamic coefficient estimate  (Fossen diagonal approximation)")
    P("=" * 88)
    P(f"  total mass          : {total_mass:.3f} kg   (hull {hull_mass:.3f} + 4 x {thr_mass:.3f})")
    P(f"  hull bbox (L,H,W)   : {np.round(dims, 4)} m")
    P(f"  water density       : {rho:.0f} kg/m^3")
    P("")

    # ---- 1. silhouettes ---------------------------------------------------------------------
    base_stl, thr_stl = _STL / "base_link.stl", _STL / "thruster_link.stl"
    have_stl = base_stl.exists() and thr_stl.exists()
    profiles = {}
    if have_stl:
        tris = read_binary_stl(base_stl)
        com = np.array(cfg["hull"]["com"], dtype=float)
        A_hull = np.zeros(3)
        for ax in range(3):
            mask, lo, cell = silhouette(tris, ax)
            A_hull[ax] = area_of(mask, cell)
            keep = [i for i in range(3) if i != ax]
            for k, along in enumerate(keep):
                s, a = strip_profile(mask, lo, cell, k)
                profiles[(ax, along)] = (s, a)
        t_tris = read_binary_stl(thr_stl)
        A_thr = np.array([area_of(silhouette(t_tris, ax)[0], 1e-3) for ax in range(3)])
        # 4 thrusters, mounted 45deg outboard: their 3 projections agree to ~10 % so orientation
        # barely matters; 0.8 discounts mutual/hull shadowing (a 20 % uncertainty of its own).
        A_thr_total = 4 * float(A_thr.mean()) * 0.8
        A = A_hull + A_thr_total
        src = "CAD STL silhouette"
    else:
        # bbox x the solidities measured from the STL when it was available (recorded here so a
        # clean checkout without the gitignored meshes still produces the same numbers).
        sol = np.array([0.3465, 0.1405, 0.2716])
        A_hull = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]]) * sol
        A_thr_total = 0.0333
        A = A_hull + A_thr_total
        com = np.array(cfg["hull"]["com"], dtype=float)
        src = "bbox x recorded solidity (STL absent)"

    box_face = np.array([dims[1] * dims[2], dims[0] * dims[2], dims[0] * dims[1]])
    solidity = A_hull / box_face
    P(f"  1. PROJECTED (silhouette) AREAS   [{src}]")
    for i in range(3):
        P(f"     A_{AXIS_NAME[i]:<9s} = {A[i]:.5f} m^2   (hull {A_hull[i]:.5f} + thrusters "
          f"{A_thr_total:.5f}; hull solidity {solidity[i] * 100:4.1f} % of the {box_face[i]:.4f} m^2 face)")
    P("")

    # ---- 2. effective Cd from BlueROV2 ------------------------------------------------------
    b = BLUEROV2
    cd_b = {k: 2 * b["drag_quad"][k] / (rho * b["A_proj"][k]) for k in ("surge", "sway", "heave")}
    # BlueROV2 Fossen axes -> UMIUSI body axes [x=surge, y=heave, z=sway]
    cd = np.array([cd_b["surge"], cd_b["heave"], cd_b["sway"]])
    P("  2. EFFECTIVE DRAG COEFFICIENTS  [BlueROV2, von Benzon 2022 Table A1, back-solved from")
    P("     Cd = 2*D_quad/(rho*A_proj) with THEIR silhouette areas — same normalisation as above]")
    for i in range(3):
        P(f"     Cd_{AXIS_NAME[i]:<9s} = {cd[i]:.2f}")
    P("     (>1 because a silhouette area cannot see the internal frame members and their wakes)")
    P("")

    # ---- 3. translational drag -------------------------------------------------------------
    dq_trans = 0.5 * rho * cd * A
    # linear term: keep BlueROV2's per-axis linear/quadratic ratio (small at cruise speed).
    ratio = np.array([b["drag_lin"]["surge"] / b["drag_quad"]["surge"],
                      b["drag_lin"]["heave"] / b["drag_quad"]["heave"],
                      b["drag_lin"]["sway"] / b["drag_quad"]["sway"]])
    dl_trans = dq_trans * ratio
    P("  3. TRANSLATIONAL DAMPING   D_q = 1/2 rho Cd A ;  D_l = D_q * (BlueROV2 lin/quad ratio)")
    old_l = np.array(cfg["drag"]["linear"], dtype=float)
    old_q = np.array(cfg["drag"]["quadratic"], dtype=float)
    for i in range(3):
        P(f"     {AXIS_NAME[i]:<9s}  quad {dq_trans[i]:7.1f} N/(m/s)^2  (was {old_q[i]:5.1f}, "
          f"x{dq_trans[i] / old_q[i]:.2f})   lin {dl_trans[i]:6.2f} N/(m/s)  (was {old_l[i]:5.1f})")
    P("")

    # ---- 4. rotational drag ----------------------------------------------------------------
    if have_stl:
        dq_rot = rotational_quadratic_drag(profiles, A_hull, cd, com, rho)
        # scale up for the thrusters, which sit at large radius: add them as point areas.
        piv = np.array([u["pivot"] for u in cfg["thrusters"]["units"]], dtype=float)
        a_thr = float(A_thr.mean()) * 0.8
        extra = np.zeros(3)
        for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
            r = piv[:, [i, j]] - com[[i, j]]
            extra[k] = 0.5 * rho * cd[[j, i]].mean() * a_thr * float(np.sum(np.linalg.norm(r, axis=1) ** 3))
        dq_rot = dq_rot + extra
        rot_src = "strip integral over the STL silhouettes + thrusters as point areas at their pivots"
    else:
        dq_rot = np.array([2.20, 8.13, 5.02])   # recorded from the STL run
        rot_src = "recorded from the STL run"
    dq_rot_ratio = np.array([b["drag_lin"]["roll"] / max(b["drag_quad"]["roll"], 1e-9),
                             b["drag_lin"]["yaw"] / max(b["drag_quad"]["yaw"], 1e-9),
                             b["drag_lin"]["pitch"] / max(b["drag_quad"]["pitch"], 1e-9)])
    dl_rot = dq_rot * np.where(dq_rot_ratio > 0, dq_rot_ratio, 0.3)
    P(f"  4. ROTATIONAL DAMPING   [{rot_src}]")
    P("     M_k = 1/2 rho [ Cd_j Int|x_i|^3 a_j(x_i) dx_i + Cd_i Int|x_j|^3 a_i(x_j) dx_j ], about the CoM")
    for k in range(3):
        P(f"     {ROT_NAME[k]:<18s}  quad {dq_rot[k]:6.2f} N*m/(rad/s)^2  (was {old_q[3 + k]:4.1f})"
          f"   lin {dl_rot[k]:5.2f} N*m/(rad/s)  (was {old_l[3 + k]:4.1f})")
    P("")

    # ---- 5. added mass ---------------------------------------------------------------------
    am_t, am_r = added_mass_estimate(dims, solidity, rho)
    # thrusters: spheres of equal volume, Ca = 0.5, at their pivots -> parallel-axis inertia
    v_thr = 5.031e-4
    m_a_thr = 0.5 * rho * v_thr
    piv = np.array([u["pivot"] for u in cfg["thrusters"]["units"]], dtype=float)
    am_t = am_t + 4 * m_a_thr
    for k, (i, j) in enumerate([(1, 2), (2, 0), (0, 1)]):
        r2 = np.sum((piv[:, [i, j]] - com[[i, j]]) ** 2, axis=1)
        am_r[k] += m_a_thr * float(np.sum(r2))
    P("  5. ADDED MASS   [translational: Lamb ellipsoid x solidity / BlueROV2 method-check ratio;")
    P("                   rotational: rod distribution of those masses / method-check ratio;")
    P("                   thrusters: equal-volume spheres (Ca = 0.5) at pivots, parallel-axis]")
    for i in range(3):
        P(f"     {AXIS_NAME[i]:<9s}  {am_t[i]:6.2f} kg      ({am_t[i] / total_mass * 100:4.1f} % of the vehicle mass)")
    for k in range(3):
        P(f"     {ROT_NAME[k]:<18s}  {am_r[k]:6.3f} kg*m^2")
    P("")

    # ---- method validation against BlueROV2 -------------------------------------------------
    P("  METHOD CHECK — the same added-mass recipe run on the BlueROV2 geometry vs its published")
    P("  values (von Benzon 2022 Table A1). This is the honest error bar on step 5:")
    bd = b["dims_LHW"]                                        # (L, H, W) -> sim axes [x, y=H, z=W]
    b_dims = np.array([bd[0], bd[1], bd[2]])
    b_face = np.array([b_dims[1] * b_dims[2], b_dims[0] * b_dims[2], b_dims[0] * b_dims[1]])
    b_A = np.array([b["A_proj"]["surge"], b["A_proj"]["heave"], b["A_proj"]["sway"]])
    b_ref_t = np.array([b["added_mass"]["surge"], b["added_mass"]["heave"], b["added_mass"]["sway"]])
    b_ref_r = np.array([b["added_mass"]["roll"], b["added_mass"]["yaw"], b["added_mass"]["pitch"]])
    b_raw_t = added_mass_translational(b_dims, b_A / b_face, rho)
    b_raw_r = added_mass_rotational_rod(b_ref_t, b_dims)      # rod fed with the GOOD trans masses
    for i in range(3):
        P(f"     {AXIS_NAME[i]:<9s}  raw {b_raw_t[i]:6.2f} kg   published {b_ref_t[i]:6.2f} kg"
          f"   ratio {b_raw_t[i] / b_ref_t[i]:.2f}  -> AM_TRANS_CAL {AM_TRANS_CAL[i]:.2f}")
    for k in range(3):
        P(f"     {ROT_NAME[k]:<18s}  raw {b_raw_r[k]:6.3f}   published {b_ref_r[k]:6.3f}"
          f"   ratio {b_raw_r[k] / b_ref_r[k]:.2f}  -> AM_ROT_CAL {AM_ROT_CAL[k]:.2f}")
    P("     The UMIUSI numbers above are the raw recipes DIVIDED by these ratios; residual")
    P("     uncertainty is ~±30 % translational, ~±(factor 2) rotational. DR must cover it.")
    P("")

    # ---- consequences ----------------------------------------------------------------------
    tpc = float(cfg["thrusters"]["thrust_per_cmd"])
    fx = 4 * abs(cfg["thrusters"]["units"][0]["thrust_axis"][0]) * tpc
    def terminal(dq, dl, F):
        return (-dl + np.sqrt(dl * dl + 4 * dq * F)) / (2 * dq)
    P("  CONSEQUENCES (surge, all four thrusters at full command)")
    P(f"     max surge thrust  : {fx:.1f} N   (4 x {tpc:.0f} N x {abs(cfg['thrusters']['units'][0]['thrust_axis'][0]):.3f})")
    P(f"     terminal speed    : {terminal(dq_trans[0], dl_trans[0], fx):.2f} m/s  with the new drag"
      f"   (was {terminal(old_q[0], old_l[0], fx):.2f} m/s)")
    P(f"     BlueROV2 for scale: {terminal(b['drag_quad']['surge'], b['drag_lin']['surge'], 85.0):.2f} m/s"
      f"  at 85 N  (paper reports 0.72 m/s measured)")
    v = 0.40
    need = dq_trans[0] * v * v + dl_trans[0] * v
    P(f"     cruise at {v} m/s  : {need:.1f} N  -> esc command {need / fx:.2f} of full scale")
    P("")
    P("  NOT DERIVABLE FROM CAD — must be measured (see docs/calibration_plan.md):")
    P("     displaced_volume, buoyancy_offset_above_com (CoB height), thrust_per_cmd, servo rate.")
    P("     The CAD STL is the MATERIAL solid (0.00373 m^3 hull + 4 x 0.00050 m^3), which is far")
    P("     from neutral for 12.48 kg — the real vehicle carries buoyancy the CAD does not model.")
    P("=" * 88)
    return "\n".join(lines), dict(A=A, cd=cd, dq_trans=dq_trans, dl_trans=dl_trans,
                                  dq_rot=dq_rot, dl_rot=dl_rot, am_t=am_t, am_r=am_r)


def emit_yaml(v):
    quad_c = np.concatenate([v["dq_trans"], v["dq_rot"]])
    lin_c = np.concatenate([v["dl_trans"], v["dl_rot"]])
    am = np.concatenate([v["am_t"], v["am_r"]])

    def fmt(a):
        return "[" + ", ".join(f"{x:.3g}" for x in a) + "]"

    return ("\ndrag:\n"
            f"  linear:    {fmt(lin_c)}\n"
            f"  quadratic: {fmt(quad_c)}\n"
            "added_mass:\n"
            f"  diag: {fmt(am)}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--emit-yaml", action="store_true", help="print a drop-in configs/umiusi.yaml block")
    args = ap.parse_args()
    text, vals = report()
    print(text)
    if args.emit_yaml:
        print(emit_yaml(vals))
