"""umiusi_perception — onboard balloon detection + tracking + the balloon-popping autonomy.

The vision-and-behaviour stack for the competition, one ROS-free library reused UNCHANGED in the
sim (``tools/autonomy_run``) and on the robot (``ros2_ws/src/umiusi_autonomy``). Depends on the
core ``umiusi_sim`` (install extra ``[perception]``); the core does not depend on it. Contents:

  * ``balloon_detector`` — classical HSV+connected-components detector. ``detect_balloons(rgb, ...)``
    -> per-balloon ``Detection``s (colour, bbox/centroid, bearing az/el, range). Ships two colour
    PROFILES: ``SIM_THRESHOLDS`` (clean sim renders) and ``REAL_THRESHOLDS`` (real underwater data).
  * ``learned_detector`` — the Pi-4-safe learned detector (``TinyBalloonNet``, CenterNet-lite);
    ``load_learned_detector(weights)`` returns the same ``rgb -> [Detection]`` interface.
  * ``tracker`` — multi-frame association / confirm-vote / persistence + near-colour re-confirmation.
  * ``underwater`` — underwater colour RESTORATION (inference preprocessing).
  * ``eval`` — the shared IoU evaluation harness (learned vs classical, per-colour P/R/F1).
  * ``autonomy`` — ``BalloonBehavior``, the rule-based search/approach/align/ram/confirm FSM.
"""

from umiusi_perception.balloon_detector import (
    REAL_THRESHOLDS,
    SIM_THRESHOLDS,
    Detection,
    detect_balloons,
)
from umiusi_perception.tracker import (
    Track,
    Tracker,
    confirm_colour,
    plausible_detections,
    sanitise_near_colours,
    size_consistent,
)

__all__ = [
    "Detection", "detect_balloons", "SIM_THRESHOLDS", "REAL_THRESHOLDS",
    "Track", "Tracker", "plausible_detections", "size_consistent",
    "confirm_colour", "sanitise_near_colours",
]
