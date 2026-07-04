"""Perception (phase 5b): onboard-camera balloon detection for the competition.

``balloon_detector.detect_balloons(rgb)`` turns a front_cam RGB frame into per-balloon
Detections (colour, image bbox/centroid, bearing az/el, range estimate). Classical CV
(HSV thresholding + connected components) — deliberately simple and replaceable for real
underwater imagery. See ``balloon_detector`` for the method and tunable constants.
"""

from umiusi_sim.perception.balloon_detector import Detection, detect_balloons

__all__ = ["Detection", "detect_balloons"]
