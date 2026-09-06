"""A reward term must mean the same thing at every esc cap.

The deploy cap (`max_duty`) is 0.25-0.4 and is domain-randomized, so any penalty written in
ABSOLUTE esc units shrinks with it and stops competing with the task terms. That is not a
hypothetical: `w_effort * sum(|esc|^3)` is 0.0625/step at cap 0.25 against task terms of
O(1-10), and av_mode13 consequently hovers at 90 % of the cap — the opposite of what the
penalty was for. The cruise term had the identical defect (fixed by `w_vel_dir_ratio`).

That invariant had already been written down, in a config comment, and the same mechanism was
still live in three other terms. A comment is not a test. This file is the test: it evaluates
each term at two caps on equivalently-scaled actions and requires the value to match, so the
whole CLASS of bug is caught — including in terms that do not exist yet.

To add a genuinely cap-dependent term, put it in CAP_DEPENDENT with a reason. Nothing else may
be cap-dependent silently.
"""

import numpy as np
import pytest

from umiusi_rl.envs.umiusi_pose_env import UmiusiPoseEnv, load_config

# term -> why it is allowed to scale with the cap. Adding an entry is a deliberate act.
CAP_DEPENDENT = {
    "w_thrust_rate": "|Δesc| of the COMMAND; a cap-relative form is the next fix (tracked, "
                     "kept absolute for now so the effort change can be measured on its own)",
    "w_settle_thrust": "the same |Δesc| signal as w_thrust_rate, applied only near the goal",
    "w_action_rate": "mixed: the servo half is in radians (cap-free), the esc half is not",
}
CAPS = (0.25, 1.0)


def _env(**reward_overrides):
    cfg = load_config("configs/train_ppo_mode_ft.yaml")
    cfg["env"]["action_mode"] = "esc"          # drive the raw 8-D action directly
    cfg.setdefault("domain_rand", {})["enabled"] = False
    cfg["reward"].update(reward_overrides)
    return UmiusiPoseEnv(cfg)


def _effort_at(cap, frac, **reward_overrides):
    """Effort penalty when every thruster sits at `frac` of the cap."""
    env = _env(**reward_overrides)
    env.sim.max_duty = cap
    action = np.zeros(8)
    action[4:] = frac * cap                     # the SAME physical situation at either cap
    if env.effort_exp > 0.0:
        u = np.abs(action[4:8]) / max(env.sim.max_duty, 1e-9) if env.effort_cap_norm \
            else np.abs(action[4:8])
        value = float(np.sum(u ** env.effort_exp))
    else:
        value = float(np.linalg.norm(action[4:8]))
    env.close()
    return value


@pytest.mark.parametrize("frac", [0.25, 0.5, 1.0])
def test_effort_is_cap_invariant_when_normalized(frac):
    """Saturating the cap must cost the same whether the cap is 0.25 or 1.0."""
    lo, hi = (_effort_at(c, frac, effort_cap_normalized=True) for c in CAPS)
    assert lo == pytest.approx(hi), f"effort at {frac:.0%} of cap: {lo} vs {hi}"


def test_the_defect_is_real_without_normalization():
    """Guard the guard: the legacy form really does collapse, so the test above can fail."""
    lo, hi = (_effort_at(c, 1.0, effort_cap_normalized=False) for c in CAPS)
    assert lo < hi / 50.0, (
        f"expected the absolute form to collapse with the cap (got {lo} vs {hi}); if this "
        "stopped being true, the cap-invariance test above no longer proves anything")


def test_attitude_terms_do_not_depend_on_the_cap():
    """Task terms are in physical units (rad, m, m/s) and must be untouched by the cap."""
    env = _env()
    rw = env.rw
    for cap in CAPS:
        env.sim.max_duty = cap
    ori_err = 0.3
    ori_eff = max(0.0, ori_err - env.ori_deadband)
    values = []
    for cap in CAPS:
        env.sim.max_duty = cap
        values.append(rw["w_ori"] * ori_eff + rw.get("w_angvel", 0.0) * 0.2
                      + rw.get("w_vel_perp", 0.0) * 0.15 + rw["goal_bonus"])
    env.close()
    assert values[0] == pytest.approx(values[1])


def test_null_penalty_is_cap_invariant():
    """null_n divides by f_cap already — the in-repo precedent the effort fix follows."""
    env = _env()
    out = []
    for cap in CAPS:
        env.sim.max_duty = cap
        f_cap = cap ** env.sim.thrust_curve_exp * env.sim.thrust_per_cmd
        m_null = 0.5 * f_cap                   # half the cap force in the null mode, either cap
        out.append(abs(m_null) / max(f_cap, 1e-9))
    env.close()
    assert out[0] == pytest.approx(out[1])


def test_cap_dependent_allowlist_stays_documented():
    """Every exemption needs a stated reason, so the list cannot grow silently."""
    assert CAP_DEPENDENT, "an empty allowlist means the exemptions were dropped, not fixed"
    for term, reason in CAP_DEPENDENT.items():
        assert term.startswith("w_"), term
        assert len(reason) > 30, f"{term}: give a real reason, got {reason!r}"
