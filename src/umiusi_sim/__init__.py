"""umiusi_sim — the CORE simulator for the UMIUSI azimuth-thruster underwater robot.

Reusable, standalone MuJoCo simulation: the analytical hydrodynamics (``physics/``), the thruster
model, the robot description + scene composition (``description/``, incl. the MjSpec appearance
editor ``description.appearance``), the onboard-camera degradation forward model
(``rendering.underwater_sim``), the feed-forward allocation (``control``), and the viewer.

This package depends ONLY on mujoco + numpy + pyyaml — NO torch, NO ROS, NO RL. It imports neither
``umiusi_perception`` nor ``umiusi_rl``; both of those import IT. So the sim runs on its own:

    from umiusi_sim.simulator import UmiusiSimulator
    sim = UmiusiSimulator(); sim.reset(pos=(0, 0.5, 0)); sim.step(action)   # action = [servo x4, esc x4]
"""
