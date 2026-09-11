"""vel_cmd_cap_frac: a commanded speed the vehicle can actually hold.

The old contract here asserted `frac * 0.68 * max_duty` with both constants hardcoded, and its
docstring called that "the episode cap's reachable speed". Measured 2026-09-08, it is not:

  * 0.68 * max_duty is the OPEN-LOOP terminal surge speed — every thruster pointing forward, none
    left for attitude. Drag rises with v**2, so 0.8 of that terminal speed already costs ~64 % of
    the thrust; adding attitude control saturates the esc on 19.5 % of cruise steps at cap 0.25
    (30 % with singularity avoidance), and both attitude AND tracking get worse.
  * 0.68 is a NOMINAL constant, and domain_rand moves the plant far more than it moves the cap:
    the thrust exponent alone spans 1.5-2.8, so the true reachable speed spans 4.8x across
    episodes while that formula spans 1.75x. The command was unreachable in the thin tail and,
    much more often, far too easy — the median command was 0.34 of what the vehicle could do, so
    near-limit cruise was effectively absent from training.

So the assertion is now against the speed the EPISODE'S OWN plant can hold (thrust = drag, solved
from its randomized constants), which is what `vel_cmd_cap_from_plant` computes. A regression to
the nominal formula fails this test instead of passing it.
"""

import numpy as np

from umiusi_rl.envs.umiusi_pose_env import VEL_PER_CAP, UmiusiPoseEnv, load_config


def _cfg(**env_overrides):
    cfg = load_config("configs/train_ppo.yaml")
    cfg["env"]["task"] = "attitude_velocity"
    cfg["env"]["vel_cmd_zero_prob"] = 0.0
    cfg["env"].update(env_overrides)
    return cfg


def _terminal_speed(sim):
    """Surge speed at which this plant's full forward thrust balances its own drag [m/s].

    Recomputed here from the sim's attributes rather than called on the env, so the test is an
    independent statement of the contract and not a restatement of the implementation.
    """
    f = 4.0 * sim.thrust_per_cmd * sim.max_duty ** sim.thrust_curve_exp
    lin, quad = float(sim.drag_lin[0]), float(sim.drag_quad[0])
    return (-lin + np.sqrt(lin * lin + 4.0 * quad * f)) / (2.0 * quad)


def test_vel_cmd_stays_within_what_this_episode_can_reach():
    frac = 0.6
    cfg = _cfg(vel_cmd_cap_frac=frac, vel_cmd_cap_from_plant=True)
    cfg.setdefault("domain_rand", {})["enabled"] = True  # the plant, not just the cap, varies
    env = UmiusiPoseEnv(cfg)
    try:
        ratios = []
        for i in range(40):
            env.reset(seed=i)
            reach = _terminal_speed(env.sim)
            speed = float(np.linalg.norm(env.v_cmd))
            assert speed <= frac * reach + 1e-9, (
                f"seed {i}: commanded {speed:.3f} m/s against a reachable {reach:.3f}")
            ratios.append(speed / reach)
        # ...and the ceiling must TRACK the plant, not sit far below it: with a nominal-constant
        # ceiling most episodes command a third of capability and top-speed cruise never trains.
        assert max(ratios) > 0.5 * frac, f"ceiling never approached: max ratio {max(ratios):.2f}"
    finally:
        env.close()


def test_plant_based_ceiling_differs_from_the_nominal_one():
    """The two ceilings must actually disagree — otherwise the flag is decorative.

    Under DR they differ by design: the nominal formula follows only max_duty, the plant-based one
    also follows thrust gain, thrust exponent and drag.
    """
    cfg = _cfg(vel_cmd_cap_frac=0.6, vel_cmd_cap_from_plant=True)
    cfg.setdefault("domain_rand", {})["enabled"] = True
    env = UmiusiPoseEnv(cfg)
    try:
        rel = []
        for i in range(40):
            env.reset(seed=i)
            rel.append(_terminal_speed(env.sim) / (VEL_PER_CAP * env.sim.max_duty))
    finally:
        env.close()
    assert max(rel) / min(rel) > 2.0, (
        f"plant-based and nominal ceilings track each other too closely ({min(rel):.2f}-{max(rel):.2f}); "
        "domain_rand is no longer moving the plant, or the flag is not wired in")


def test_vel_cmd_cap_off_by_default():
    env = UmiusiPoseEnv(_cfg())
    try:
        speeds = [float(np.linalg.norm((env.reset(seed=i), env.v_cmd)[1])) for i in range(40)]
    finally:
        env.close()
    # legacy flat U(0, 0.4): commands above any capped ceiling must still occur
    assert max(speeds) > 0.30
