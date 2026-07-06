"""Multi-frame balloon TRACKER + detection sanitisers (ROS-free, sim-free perception library).

Why this exists
---------------
The learned detector flickers: a real balloon drops out for a frame or two (especially a large,
close one, or the tall 1.5 m yellow as it drifts near a frame edge), and spurious one-frame false
positives pop in and out. Feeding those raw per-frame detections straight to the behaviour FSM makes
it (a) re-acquire the SAME balloon over and over (losing the tall yellow's identity), (b) chase and
even commit to flickery false positives, and (c) risk popping a mislabelled blue up close. This
module stabilises detections BEFORE the FSM sees them, in one reusable place so both
``tools/autonomy_run`` (sim) and the future ``perception_node`` / ``navigator_node`` (robot) share it
(see ``docs/architecture.md``).

Three pieces, all pure Python over the classical/learned ``Detection`` dataclass:

1. ``Tracker`` — associates each frame's detections to persistent TRACKS by colour + bearing (+ range)
   gating, assigns stable integer track IDs, EMA-smooths bearing/range, CONFIRMS a track only after it
   is voted in over several perception frames (kills flickery false positives), and PERSISTS a track
   through a bounded run of missed frames (so a briefly-lost balloon keeps its identity instead of
   being re-acquired as a new target). This is the ONE mechanism that folds together what
   ``tools/behavior.py`` previously did with two separate ad-hoc structures (an acquisition-vote map
   for FP rejection + a single lock-hold track). The FSM now consumes CONFIRMED tracks.

2. ``size_consistent`` — a distance+size plausibility gate: a real balloon of known radius subtends a
   predictable pixel size / silhouette shape at its estimated range; a "detection" whose box shape or
   absolute size disagrees is rejected as a false positive.

3. ``confirm_colour`` / ``sanitise_near_colours`` — close-range red-vs-blue confirmation from the
   actual bbox pixels. Underwater the blue cast makes the learned red/blue labels risky, but the
   robot's lights help at close range, so before the FSM commits to popping a NEAR "red" we re-read
   the pixels; if they look blue we relabel it blue so the FSM avoids it (never pop blue, -10).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from umiusi_sim.perception.balloon_detector import (
    BALLOON_DIAMETER_M,
    COLOUR_POINTS,
    Detection,
    _pinhole,
    rgb_to_hsv,
)

# ======================================================================================
# 1. Temporal tracker
# ======================================================================================
# --- association gates (a detection joins an existing track if it clears ALL of these) --------
ASSOC_BEARING = math.radians(14.0)   # bearing gate: detection within this angle of the track [rad]
ASSOC_RANGE_FRAC = 2.0               # range gate: |r_det - r_trk| <= this * r_trk. DELIBERATELY loose
#                                      — the apparent-size range estimate is very noisy (a close
#                                      balloon's bbox flickers, swinging range ~2x frame-to-frame), so
#                                      this only rejects a >3x mismatch (clearly a different object) and
#                                      never a legitimate near-target dropout. Bearing + colour do the
#                                      real separating; a tighter range gate silently drops close
#                                      targets mid-approach and kills ram commits.
# --- confirmation (kill flickery false positives) ---------------------------------------------
# A track is CONFIRMED only once it has been VOTED IN over several perception frames. Votes rise by
# one on every fresh frame the track is seen and fall by one on a fresh frame it is missed (capped at
# VOTE_MAX so a long-lived track re-confirms fast yet can still decay away). A one-frame FP never
# accumulates CONFIRM_VOTES, so it can never become a target. Confirmation is STICKY: once a track
# has earned it, a later dip in votes does not un-confirm it — that is what lets a real balloon keep
# its identity through flicker instead of being dropped and re-acquired.
CONFIRM_VOTES = 3                    # fresh-frame votes required before a track is "confirmed"
VOTE_MAX = 6                         # vote cap (bounded re-confirm / decay)
# --- persistence (hold identity through dropouts) ---------------------------------------------
# A track survives up to PERSIST_FRAMES consecutive MISSED control steps before it is deleted; while
# missed it keeps its ID and last smoothed state so the FSM can bridge a detector dropout (commit +
# lunge through it) and, crucially, RE-ASSOCIATE the same balloon on the next hit rather than acquire
# a fresh target. Counted in control steps (like the FSM's own timers), so held frames between
# perception ticks age it too — matching the pre-refactor lock-hold exactly.
PERSIST_FRAMES = 12                  # consecutive missed control steps before a track is dropped
# --- smoothing (EMA on bearing/range) ---------------------------------------------------------
# Light EMA: bearing stays responsive (the close-range RAM needs a live az/el to stay head-on), range
# is smoothed harder because the apparent-size range estimate is the noisiest signal. alpha = weight
# of the NEW measurement (1.0 = no smoothing).
BEARING_EMA = 0.85
RANGE_EMA = 0.4


@dataclass
class Track:
    """A persistent balloon track (stable ``id`` across frames)."""

    id: int
    colour: str
    az: float
    el: float
    range_m: float
    bbox: tuple
    det: object = None            # latest associated Detection (for downstream pixel work)
    votes: int = 1               # fresh-frame confirmation votes
    confirmed: bool = False      # sticky once votes reach CONFIRM_VOTES
    misses: int = 0              # consecutive missed control steps (persistence counter)
    hits: int = 1               # total fresh frames this track was seen (stats)
    age: int = 0                # total control steps alive (stats)

    @property
    def bearing(self):
        return (self.az, self.el)


class Tracker:
    """Associates detections to persistent tracks; confirms + persists them.

    Call ``update(detections, fresh)`` once per CONTROL step. ``fresh`` is True on a genuine
    perception tick and False when the caller re-drives on HELD detections between detector ticks —
    confirmation votes and hit counts advance only on fresh frames, while the persistence miss-counter
    ages every step (mirroring realistic Pi-4 perception-vs-control timing). ``confirmed()`` returns
    the confirmed tracks the behaviour FSM should act on; ``get(id)`` fetches one by ID.
    """

    def __init__(self, assoc_bearing: float = ASSOC_BEARING, assoc_range_frac: float = ASSOC_RANGE_FRAC,
                 confirm_votes: int = CONFIRM_VOTES, persist_frames: int = PERSIST_FRAMES,
                 bearing_ema: float = BEARING_EMA, range_ema: float = RANGE_EMA):
        self.assoc_bearing = assoc_bearing
        self.assoc_range_frac = assoc_range_frac
        self.confirm_votes = confirm_votes
        self.persist_frames = persist_frames
        self.bearing_ema = bearing_ema
        self.range_ema = range_ema
        self.tracks: list[Track] = []
        self._next_id = 1

    def update(self, detections, fresh: bool = True) -> None:
        """Advance the tracker by one control step over ``detections`` (already range/size gated)."""
        for t in self.tracks:
            t.age += 1
        # Greedy nearest-bearing association, each detection to at most one track and vice-versa.
        unmatched = list(detections)
        matched_ids = set()
        # Sort tracks confirmed-first then by fewest misses so the most established track wins a
        # contested detection (keeps IDs stable through crossings).
        for t in sorted(self.tracks, key=lambda t: (not t.confirmed, t.misses)):
            best, best_d = None, self.assoc_bearing
            for d in unmatched:
                if d.colour != t.colour:
                    continue
                if not self._range_ok(d.range_m, t.range_m):
                    continue
                dd = math.hypot(d.bearing[0] - t.az, d.bearing[1] - t.el)
                if dd < best_d:
                    best, best_d = d, dd
            if best is not None:
                self._absorb(t, best, fresh)
                unmatched.remove(best)
                matched_ids.add(t.id)
        # Age / decay every track that was NOT matched this step.
        for t in self.tracks:
            if t.id in matched_ids:
                continue
            t.misses += 1
            if fresh:
                t.votes -= 1
        # Spawn new tentative tracks for leftover detections (only on fresh frames — held frames just
        # re-present the same detections and must not create phantom duplicates).
        if fresh:
            for d in unmatched:
                self.tracks.append(Track(id=self._next_id, colour=d.colour, az=d.bearing[0],
                                         el=d.bearing[1], range_m=d.range_m, bbox=d.bbox, det=d))
                self._next_id += 1
        # Drop dead tracks: lost too long, or an unconfirmed candidate whose votes decayed away.
        self.tracks = [t for t in self.tracks
                       if t.misses <= self.persist_frames and (t.confirmed or t.votes > 0)]

    def _absorb(self, t: Track, d, fresh: bool) -> None:
        """Fold a matched detection into track ``t`` (EMA-smooth, refresh votes/confirm)."""
        a_b, a_r = self.bearing_ema, self.range_ema
        t.az += a_b * (d.bearing[0] - t.az)
        t.el += a_b * (d.bearing[1] - t.el)
        if math.isfinite(d.range_m):
            t.range_m = d.range_m if not math.isfinite(t.range_m) else t.range_m + a_r * (d.range_m - t.range_m)
        t.bbox = d.bbox
        t.det = d
        t.misses = 0
        if fresh:
            t.votes = min(VOTE_MAX, t.votes + 1)
            t.hits += 1
            if t.votes >= self.confirm_votes:
                t.confirmed = True

    def _range_ok(self, r_det: float, r_trk: float) -> bool:
        if not (math.isfinite(r_det) and math.isfinite(r_trk)):
            return True
        return abs(r_det - r_trk) <= self.assoc_range_frac * max(r_trk, 1e-3)

    def confirmed(self) -> list[Track]:
        """Confirmed tracks (what the behaviour FSM selects/acts on)."""
        return [t for t in self.tracks if t.confirmed]

    def get(self, track_id) -> Track | None:
        for t in self.tracks:
            if t.id == track_id:
                return t
        return None


# ======================================================================================
# 2. Distance + size consistency filter
# ======================================================================================
# A real competition balloon is a BALLOON_RADIUS = 0.10 m sphere (0.20 m diameter), rendered as a
# vertical ellipsoid of aspect ~1.25 (height/width; see render_appearance.BALLOON_ASPECT). At range
# r its silhouette therefore subtends a PREDICTABLE pixel size given the camera's vertical FOV:
#
#     fx = (H/2) / tan(fovy/2)                 # focal length in px (square pixels)
#     apparent_diameter_px(r) = BALLOON_DIAMETER_M * fx / r
#
# The detectors DERIVE range from exactly this relation (range_m = D*fx / mean(w,h)), so the mean box
# size is consistent with the reported range by construction — the independent signal that a box is a
# false positive is therefore its SHAPE and ABSOLUTE size:
#   * aspect: width and height must each be consistent with one balloon range, so their ratio must sit
#     near the rendered ellipsoid aspect. A streak / colour-fringe / merged blob is far too elongated.
#   * absolute size: a box only a few px on a side is sub-pixel detector noise (its range estimate is
#     meaningless); a box implying a physically impossible near range is likewise rejected.
# All bands are deliberately loose (this rejects gross implausibilities, not borderline balloons).
ASPECT_LO = 0.55          # min box height/width for a balloon silhouette (reject horizontal streaks)
ASPECT_HI = 2.6           # max box height/width (rendered ~1.25; allow tilt/occlusion/blur headroom)
MIN_BBOX_PX = 6           # min(width, height) in px below which the box is sub-pixel noise
MIN_RANGE_M = 0.12        # a balloon closer than this would over-fill the frame — implausible range
SIZE_TOL = 0.6            # tolerance on apparent-vs-expected mean diameter (safety net; see note)


def size_consistent(det: Detection, frame_h: int = 240, frame_w: int = 320,
                    fovy_deg: float = 60.0, max_range_m: float = float("inf")) -> bool:
    """True if ``det``'s box shape + size is physically plausible for a balloon at its range.

    ``frame_h`` / ``fovy_deg`` set the pinhole focal length used for the (defensive) apparent-size
    check; ``max_range_m`` lets the caller fold its range gate in here too. See the module notes for
    the balloon-radius / aspect / tolerance assumptions.
    """
    u0, v0, u1, v1 = det.bbox
    w, h = (u1 - u0), (v1 - v0)
    if w < 1 or h < 1 or min(w, h) < MIN_BBOX_PX:
        return False
    aspect = h / w
    if not (ASPECT_LO <= aspect <= ASPECT_HI):
        return False
    r = det.range_m
    if not math.isfinite(r) or r < MIN_RANGE_M or r > max_range_m:
        return False
    # Defensive apparent-vs-expected size agreement. For the current detectors this is ~1.0 by
    # construction (range is derived from size); it only bites a future detector that reports range
    # independently of the box. Kept as a documented safety net, not the primary filter.
    fx, _fy, _cx, _cy = _pinhole(frame_h, frame_w, fovy_deg)
    expected = BALLOON_DIAMETER_M * fx / r
    apparent = 0.5 * (w + h)
    if expected > 0 and abs(apparent - expected) > SIZE_TOL * expected:
        return False
    return True


def plausible_detections(detections, frame_h: int = 240, frame_w: int = 320, fovy_deg: float = 60.0,
                         max_range_m: float = float("inf")):
    """Filter a detection list to physically plausible boxes (range gate + size consistency)."""
    return [d for d in detections
            if size_consistent(d, frame_h, frame_w, fovy_deg, max_range_m=max_range_m)]


# ======================================================================================
# 3. Close-range red / blue colour confirmation (from the actual pixels)
# ======================================================================================
# Underwater the water casts everything blue-green, so the learned detector's red/blue labels are
# least reliable exactly where a wrong call is most costly: a "red" that is really a BLUE decoy, up
# close, would be rammed for -10. At close range the robot's lights restore enough colour to check the
# label directly from the pixels. We sample the box INTERIOR (excludes the rim/tether/edge halo) and
# read robust hue statistics. Bands are wide to survive the residual cast; the decision only needs to
# separate genuine red from genuine blue, and it errs toward calling a patch blue (safety: never pop
# blue). Hue is in [0,360): red wraps 0, blue/cyan sits ~180-260.
COLOUR_CONFIRM_RANGE_M = 2.5    # only re-check labels this near (close-range colour-confirm band)
_INTERIOR_FRAC = 0.5            # central fraction of the box sampled (drop rim/tether/edge pixels)
_S_MIN = 0.12                  # ignore near-grey pixels (unlit / washed out) when voting on hue
_V_MIN = 0.10                  # ignore near-black rim pixels
_RED_HUE = ((0.0, 30.0), (330.0, 360.0))   # red wraps 0 deg
_BLUE_HUE = ((175.0, 275.0),)              # cyan .. blue (underwater blue reads cyan-ish)
_YELLOW_HUE = ((35.0, 170.0),)             # yellow reads yellow..green underwater
_DOMINANT_FRAC = 0.30          # a colour must claim at least this fraction of lit interior pixels


def _hue_frac(hue, sat, val, windows) -> float:
    lit = (sat >= _S_MIN) & (val >= _V_MIN)
    n = int(lit.sum())
    if n == 0:
        return 0.0
    m = np.zeros_like(hue, dtype=bool)
    for lo, hi in windows:
        m |= (hue >= lo) & (hue <= hi)
    return float((m & lit).sum()) / n


def confirm_colour(rgb: np.ndarray, bbox) -> str | None:
    """Dominant balloon colour ("red"/"yellow"/"blue") in the box interior, or None if inconclusive.

    Reusable, ROS/sim-free: pass the raw camera frame and a detection's bbox. Samples the central
    ``_INTERIOR_FRAC`` of the box in HSV and returns whichever of red/yellow/blue owns the most lit
    pixels (>= ``_DOMINANT_FRAC``); None when the patch is too washed out / mixed to call.
    """
    arr = np.asarray(rgb)
    if arr.ndim != 3 or arr.shape[2] < 3:
        return None
    H, W = arr.shape[:2]
    u0, v0, u1, v1 = (int(round(x)) for x in bbox)
    u0, u1 = max(0, min(u0, W)), max(0, min(u1, W))
    v0, v1 = max(0, min(v0, H)), max(0, min(v1, H))
    if u1 - u0 < 2 or v1 - v0 < 2:
        return None
    mu = ((1.0 - _INTERIOR_FRAC) / 2.0)
    iu0 = u0 + int((u1 - u0) * mu)
    iu1 = u1 - int((u1 - u0) * mu)
    iv0 = v0 + int((v1 - v0) * mu)
    iv1 = v1 - int((v1 - v0) * mu)
    patch = arr[max(v0, iv0):max(iv0 + 1, iv1), max(u0, iu0):max(iu0 + 1, iu1), :3]
    if patch.size == 0:
        return None
    hsv = rgb_to_hsv(patch)
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    fracs = {
        "red": _hue_frac(hue, sat, val, _RED_HUE),
        "yellow": _hue_frac(hue, sat, val, _YELLOW_HUE),
        "blue": _hue_frac(hue, sat, val, _BLUE_HUE),
    }
    colour = max(fracs, key=fracs.get)
    return colour if fracs[colour] >= _DOMINANT_FRAC else None


def sanitise_near_colours(rgb: np.ndarray, detections, near_range_m: float = COLOUR_CONFIRM_RANGE_M):
    """Re-check the colour of NEAR red/blue detections against their pixels before the FSM commits.

    For every detection within ``near_range_m`` that the detector called red or blue, re-read the box
    interior: if a claimed "red" actually looks blue, RELABEL it blue (points -10) so the FSM avoids
    instead of popping it (never pop blue). A blue that reads red up close is likewise corrected. Only
    a CONFIDENT contradicting pixel call flips a label; anything inconclusive is left untouched.
    Returns a new list (detections are replaced, not mutated in place). Yellow is left alone here (its
    green underwater cast is handled by the detector; this guards the costly red/blue confusion only).
    """
    out = []
    for d in detections:
        if d.colour in ("red", "blue") and math.isfinite(d.range_m) and d.range_m <= near_range_m:
            seen = confirm_colour(rgb, d.bbox)
            if seen in ("red", "blue") and seen != d.colour:
                out.append(_relabel(d, seen))
                continue
        out.append(d)
    return out


def _relabel(d: Detection, colour: str) -> Detection:
    """Copy of detection ``d`` with a corrected colour + points (bbox/geometry unchanged)."""
    from dataclasses import replace
    return replace(d, colour=colour, points=COLOUR_POINTS[colour])


__all__ = [
    "Track", "Tracker", "ASSOC_BEARING", "ASSOC_RANGE_FRAC", "CONFIRM_VOTES", "VOTE_MAX",
    "PERSIST_FRAMES", "BEARING_EMA", "RANGE_EMA",
    "size_consistent", "plausible_detections", "ASPECT_LO", "ASPECT_HI", "MIN_BBOX_PX",
    "confirm_colour", "sanitise_near_colours", "COLOUR_CONFIRM_RANGE_M",
]
