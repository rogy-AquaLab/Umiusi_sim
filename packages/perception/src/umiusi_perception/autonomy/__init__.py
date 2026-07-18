"""Autonomy: the ROS-free, sim-free balloon-popping behaviour FSM.

``BalloonBehavior`` consumes onboard ``Detection``s (colour / bearing / range / bbox) plus the
measured yaw rate and emits a simple drive command ({surge, heave, yaw}). It reads NO ground truth
and NO ROS — the SAME object drives both the in-sim autonomy run (``tools/autonomy_run``) and the
real robot's ``navigator_node`` (``ros2_ws/src/umiusi_autonomy``). Keeping it here (an installable
package, not ``tools/``) is what lets the deploy nodes ``import`` it directly.
"""

from umiusi_perception.autonomy.behavior import BalloonBehavior

__all__ = ["BalloonBehavior"]
