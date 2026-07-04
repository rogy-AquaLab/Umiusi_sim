"""Perception (phase 5b): onboard-camera balloon detection for the competition.

``balloon_detector.detect_balloons(rgb, thresholds=...)`` turns a front_cam RGB frame into
per-balloon Detections (colour, image bbox/centroid, bearing az/el, range estimate). Classical
CV (HSV thresholding + connected components) — deliberately simple and replaceable.

Two colour PROFILES are shipped: ``SIM_THRESHOLDS`` (default; tuned to clean sim renders) and
``REAL_THRESHOLDS`` (data-driven from a labelled real underwater dataset). Real imagery also
enables ``reject_reflections=True`` to drop water-surface reflections. See ``balloon_detector``
for the method, the derivation of the real windows, and the tunable constants.
"""

from umiusi_sim.perception.balloon_detector import (
    REAL_THRESHOLDS,
    SIM_THRESHOLDS,
    Detection,
    detect_balloons,
)

__all__ = ["Detection", "detect_balloons", "SIM_THRESHOLDS", "REAL_THRESHOLDS"]
