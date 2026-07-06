"""Compatibility shim — the behaviour FSM moved into the installable package.

The perception-driven balloon-popping FSM now lives in ``umiusi_sim.autonomy.behavior`` so the ROS
deploy nodes (``ros2_ws/src/umiusi_autonomy``) can ``import`` it without adding ``tools/`` to the
path. ``tools/autonomy_run`` and any external caller keep importing ``tools.behavior`` unchanged.
"""

from umiusi_sim.autonomy.behavior import *  # noqa: F401,F403
from umiusi_sim.autonomy.behavior import BalloonBehavior  # noqa: F401
