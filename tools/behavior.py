"""Perception-driven balloon-popping behaviour FSM (vision in the loop).

Consumes the onboard detector's output (a list of ``Detection`` with colour / bearing (az,el) /
range / bbox) plus the measured yaw rate, and emits a simple, robust drive command (``surge``
forward, ``heave`` up, ``yaw`` rate) that ``tools/autonomy_run`` feeds through the analytical
feed-forward allocation. NO ground truth: EVERY decision — which balloon to chase, whether a ram
popped it or missed — is made from the camera detections alone, exactly what would run on the real
robot behind the Pi-4 detector.

Camera-based pop confirmation
-----------------------------
The FSM never reads the physics/score. It keeps a temporal TRACK of the locked target (associates
detections across frames by colour + bearing proximity, holding it through brief flicker). After it
COMMITS to a ram, if the target's detection DISAPPEARS (no same-colour detection near its last
bearing) for CONFIRM_FRAMES frames -> it treats the balloon as popped (on a real pop the balloon is
hidden, so the camera sees it gone) and acquires the next target. If the target is STILL detected
after the ram -> it was a glancing MISS -> RECOVER (back off, re-align, retry). The game's authoritative
score is still judged by ``scn.popped`` in the driver; the ROBOT only ever acts on the camera.

States
------
SEARCH   : no red/yellow visible -> deliberate exploration. Full 360° in-place yaw SWEEP (integrating
           measured yaw), height-scanning (heave oscillation; balloons sit at 0.5/0.7/1.5 m); a sweep
           that finds nothing TRANSLATES to a fresh spot and sweeps again, alternating direction.
APPROACH : lock the nearest (largest-bbox) red/yellow; yaw onto its bearing + forward surge (throttled
           while mis-pointed), heave to centre it. AVOID blue.
ALIGN    : close (big bbox) -> SLOW down and precisely CENTRE the balloon (az,el -> ~0) before the ram
           so the pin hits head-on; if it can't get centred (persistently off-angle) -> REPOSITION.
RAM      : big bbox AND centred -> drive straight in at full surge to pop it head-on.
CONFIRM  : post-ram camera check -> back off while watching: target stays gone -> POP; reappears -> MISS.
RECOVER  : a miss/reposition -> back off, re-acquire, re-ALIGN, retry. A per-target attempt cap /
           pursuit timeout abandons a stubborn balloon and searches elsewhere.

Sign convention (matches ``tools/competition_run``): ``front_cam`` looks +X with image-right = body
+Z, so a target at +azimuth is toward body +Z and a POSITIVE yaw turns the nose that way. ``heave``
maps to +Y (up); a target ABOVE the optic axis (el>0) commands heave up.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

POSITIVE_COLOURS = ("red", "yellow")

# --- drive gains (feed-forward command convention; mirrors competition_run) -------------------
SPEED_CAP = 0.35          # max surge/heave command magnitude ("modest speed")
KP_YAW = 1.1              # yaw P gain: yaw command per radian of bearing error
KD_YAW = 0.15             # yaw D gain (damps the measured yaw rate)
KP_HEAVE = 1.3            # heave P gain: command per radian of target elevation error (get to depth)
FACE_TOL = math.radians(45.0)   # surge hard only once within this bearing error (APPROACH)
# --- SEARCH / exploration ---------------------------------------------------------------------
SEARCH_YAW = 0.5          # in-place yaw-sweep rate command
SEARCH_SURGE = 0.30       # forward speed while translating to a fresh spot between sweeps
SCAN_HEAVE = 0.15         # heave amplitude while sweeping (scan the different balloon heights)
SCAN_RATE = 1.5           # rad/s of the height-scan oscillation
TRANSLATE_STEPS = 50      # steps (~1 s @50 Hz) to translate before the next sweep
# --- closeness by BBOX (more reliable than the noisy range estimate) --------------------------
ALIGN_BBOX = 0.18         # bbox height / frame >= this -> close: start the slow centred ALIGN
RAM_COMMIT_BBOX = 0.26    # ...and >= this AND centred -> commit to the RAM
RAM_MAX_STEPS = 85        # committed and still visible this long (never popped) -> treat as a MISS
PASS_PEAK_BBOX = 0.45     # a "clearly passed it" MISS needs the balloon to have filled this much...
PASS_DROP_FRAC = 0.5      # ...then shrunk below this fraction of that peak while STILL visible
CENTRE_AZ = math.radians(8.0)   # "centred" = bearing within this...
CENTRE_EL = math.radians(11.0)  # ...and elevation within this (head-on enough for the 25° pin cone)
ALIGN_CREEP = 0.12        # small forward creep while aligning (keeps closing as it centres)
ALIGN_TIMEOUT = 90        # steps stuck in ALIGN un-centred -> REPOSITION (back off, new line)
# --- recover / confirm / give-up --------------------------------------------------------------
RECOVER_SURGE = 0.28      # reverse speed while backing off a missed ram
RECOVER_STEPS = 30        # steps (~0.6 s) to back off before re-aligning
CONFIRM_FRAMES = 26       # target gone this many frames while backing off a ram -> camera POP
CONFIRM_SURGE = 0.30      # firm reverse during CONFIRM: a merely-passed (not popped) balloon must
#                           come BACK into view (-> re-associates -> MISS); only a truly popped
#                           (hidden) one stays gone the whole back-off -> confirmed pop.
MAX_ATTEMPTS = 4          # ram attempts on one target before abandoning it
MAX_PURSUIT_STEPS = 500   # ~10 s pursuing one target before abandoning it
# --- blue avoidance ---------------------------------------------------------------------------
AVOID_AZ = math.radians(28.0)   # a blue within this bearing of dead-ahead is "in the way"
AVOID_RANGE = 1.6         # ...and closer than this -> steer around it
AVOID_YAW = 0.5           # avoidance yaw magnitude added away from the blue
# --- temporal track ---------------------------------------------------------------------------
LOCK_HOLD_STEPS = 8       # bridge detector dropouts: a track is "alive" until this many misses
ASSOC_BEARING = math.radians(14.0)  # associate a detection to the track if within this bearing


@dataclass
class _Track:
    """Temporal track of the locked target (associated across frames by colour + bearing)."""
    colour: str = ""
    az: float = 0.0
    el: float = 0.0
    range_m: float = float("inf")
    bbox_frac: float = 0.0          # bbox height / frame height (apparent closeness)
    peak_bbox: float = 0.0          # max bbox_frac seen this pursuit (for the miss test)
    misses: int = 999               # consecutive frames with no associated detection

    def alive(self):
        return bool(self.colour) and self.misses <= LOCK_HOLD_STEPS


@dataclass
class BalloonBehavior:
    """Stateful perception FSM. Call ``step(detections, yaw_rate, heading, dt)`` per control step."""

    frame_h: int = 240
    dt: float = 0.02
    state: str = "SEARCH"
    trk: _Track = field(default_factory=_Track)
    # search / sweep
    _swept: float = 0.0
    _sweep_dir: float = 1.0
    _scan_phase: float = 0.0
    _translating: int = 0
    _sweep_cycles: int = 0
    # pursuit / ram / recover / confirm
    _pursuit_steps: int = 0
    _attempts: int = 0
    _align_steps: int = 0
    _committed: bool = False        # have we committed to ramming this target?
    _ram_steps: int = 0             # steps since the current RAM commit
    _recover_steps: int = 0
    _confirm_frames: int = 0
    # stats (for the run report)
    n_ram: int = 0
    n_recover: int = 0
    n_abandon: int = 0
    n_confirmed_pop: int = 0        # camera-confirmed pops (FSM's own belief; NOT the GT score)
    n_miss: int = 0

    # -- detection selection / association -----------------------------------------------------
    def _pick_target(self, dets):
        """Acquire the nearest red/yellow = the LARGEST apparent (biggest bbox area)."""
        pos = [d for d in dets if d.colour in POSITIVE_COLOURS]
        return max(pos, key=lambda d: d.area_px) if pos else None

    def _associate(self, dets):
        """The detection matching the current track (same colour, closest bearing within gate)."""
        if not self.trk.colour:
            return None
        best, best_d = None, ASSOC_BEARING
        for d in dets:
            if d.colour != self.trk.colour:
                continue
            dd = math.hypot(d.bearing[0] - self.trk.az, d.bearing[1] - self.trk.el)
            if dd < best_d:
                best, best_d = d, dd
        return best

    def _set_track(self, d):
        f = self._bbox_frac(d)
        self.trk = _Track(colour=d.colour, az=d.bearing[0], el=d.bearing[1], range_m=d.range_m,
                          bbox_frac=f, peak_bbox=f, misses=0)

    def _update_track(self, d):
        self.trk.az, self.trk.el, self.trk.range_m = d.bearing[0], d.bearing[1], d.range_m
        self.trk.bbox_frac = self._bbox_frac(d)
        self.trk.peak_bbox = max(self.trk.peak_bbox, self.trk.bbox_frac)
        self.trk.misses = 0

    def _bbox_frac(self, d):
        u0, v0, u1, v1 = d.bbox
        return (v1 - v0) / max(1, self.frame_h)

    def _blue_avoidance(self, dets):
        """Yaw bias to steer around a blue that is roughly dead-ahead and close. 0 if clear."""
        threats = [d for d in dets
                   if d.colour == "blue" and abs(d.bearing[0]) < AVOID_AZ and d.range_m < AVOID_RANGE]
        if not threats:
            return 0.0, None
        b = min(threats, key=lambda d: d.range_m)
        side = -1.0 if b.bearing[0] >= 0 else 1.0  # turn away; dead-centre -> deterministic left
        gain = AVOID_YAW * (1.0 - min(1.0, b.range_m / AVOID_RANGE) * 0.5)
        return side * gain, b

    def _info(self, blue):
        return {"state": self.state, "target": self.trk.colour or None, "range": self.trk.range_m,
                "az": self.trk.az, "bbox": round(self.trk.bbox_frac, 2), "attempts": self._attempts,
                "held": self.trk.misses > 0, "blue_threat": blue is not None}

    # -- transitions ---------------------------------------------------------------------------
    def _reset_pursuit(self):
        self._pursuit_steps = 0
        self._attempts = 0
        self._align_steps = 0
        self._committed = False

    def _start_sweep(self):
        """(Re)start an in-place sweep, biased toward where the target was last seen."""
        self._swept = 0.0
        self._scan_phase = 0.0
        self._translating = 0
        self._sweep_dir = 1.0 if self.trk.az >= 0 else -1.0

    def _clear_track(self):
        self.trk = _Track()

    def _enter_recover(self):
        self.state = "RECOVER"
        self._recover_steps = RECOVER_STEPS
        self._align_steps = 0
        self.n_recover += 1

    def _enter_confirm(self):
        self.state = "CONFIRM"
        self._confirm_frames = 0

    def _abandon(self):
        """Give up a stubborn target: reset pursuit, search AWAY from it (flip + translate)."""
        self.n_abandon += 1
        self._reset_pursuit()
        self.state = "SEARCH"
        self._start_sweep()
        self._sweep_dir *= -1.0              # look the other way than the abandoned target
        self._translating = TRANSLATE_STEPS  # ...and move to a fresh area first
        self._clear_track()

    # -- main tick -----------------------------------------------------------------------------
    def step(self, detections, yaw_rate, heading=0.0, dt=None):
        """Return (command, info). command = {surge, heave, yaw}. Camera-only decisions."""
        dt = self.dt if dt is None else dt
        avoid_yaw, blue = self._blue_avoidance(detections)

        # Temporal track update: associate a detection to the current track (or count a miss).
        if self.trk.colour:
            m = self._associate(detections)
            if m is not None:
                self._update_track(m)
            else:
                self.trk.misses += 1

        # Post-ram camera confirmation (pop vs miss) — pure camera, no ground truth.
        if self.state == "CONFIRM":
            return self._confirm(yaw_rate, avoid_yaw, blue)
        if self.state == "RECOVER":
            return self._recover(yaw_rate, avoid_yaw, blue)

        # Acquire a target when searching.
        if self.state == "SEARCH" or not self.trk.colour:
            target = self._pick_target(detections)
            if target is None:
                self.state = "SEARCH"
                return self._search(yaw_rate, dt, avoid_yaw, blue)
            self._set_track(target)
            self._reset_pursuit()
            self.state = "APPROACH"

        # Lost the track while pursuing -> confirm (if committed) or recover.
        if not self.trk.alive():
            if self._committed:
                self._enter_confirm()
                return self._confirm(yaw_rate, avoid_yaw, blue)
            self._attempts += 1
            self._enter_recover()
            return self._recover(yaw_rate, avoid_yaw, blue)

        self._pursuit_steps += 1
        if self._pursuit_steps > MAX_PURSUIT_STEPS or self._attempts >= MAX_ATTEMPTS:
            self._abandon()
            return self._search(yaw_rate, dt, avoid_yaw, blue)

        az, el, bbox = self.trk.az, self.trk.el, self.trk.bbox_frac
        centred = abs(az) < CENTRE_AZ and abs(el) < CENTRE_EL

        # RAM in progress: drive straight in. A real pop makes the balloon vanish -> the track is
        # lost -> handled above as CONFIRM. Here we only catch the case where it stayed visible:
        # we CLEARLY passed a big balloon without popping (peak filled the frame, now shrunk), or a
        # RAM timeout (drove in but never popped). We do NOT miss on small bbox noise.
        if self.state == "RAM":
            self._ram_steps += 1
            passed = (self.trk.misses == 0 and self.trk.peak_bbox >= PASS_PEAK_BBOX
                      and bbox < PASS_DROP_FRAC * self.trk.peak_bbox)
            if passed or self._ram_steps > RAM_MAX_STEPS:
                self.n_miss += 1
                self._attempts += 1
                self._enter_recover()
                return self._recover(yaw_rate, avoid_yaw, blue)
            # Tight tracking through the ram (full gains) so it stays head-on to contact.
            yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate, -1, 1)
            heave = _clip(KP_HEAVE * el, -SPEED_CAP, SPEED_CAP)
            return {"surge": SPEED_CAP, "heave": heave, "yaw": yaw}, self._info(blue)

        # Commit to a RAM: close enough (bbox) AND pointed head-on.
        if bbox >= RAM_COMMIT_BBOX and centred:
            self.state = "RAM"
            self._committed = True
            self._align_steps = 0
            self._ram_steps = 0
            self.n_ram += 1
            yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate, -1, 1)
            heave = _clip(KP_HEAVE * el, -SPEED_CAP, SPEED_CAP)
            return {"surge": SPEED_CAP, "heave": heave, "yaw": yaw}, self._info(blue)

        # ALIGN: close but not yet committable -> slow, precisely centre the balloon first.
        if bbox >= ALIGN_BBOX:
            self.state = "ALIGN"
            self._align_steps += 1
            # We are in the pop zone (close): from here on a DISAPPEARANCE is treated as a pop to
            # confirm (the pin can contact during the creep, before a formal RAM commit).
            self._committed = True
            if self._align_steps > ALIGN_TIMEOUT and not centred:
                # Persistently off-angle at close range -> can't get a clean head-on line here.
                # REPOSITION: back off (RECOVER) and re-approach on a fresh line rather than ram
                # glancing. (This vehicle can't strafe — sway makes a yaw couple — so backing off +
                # re-aiming is how it arcs to a new approach line.)
                self._align_steps = 0
                self._enter_recover()
                return self._recover(yaw_rate, avoid_yaw, blue)
            yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate + avoid_yaw, -1, 1)
            heave = _clip(KP_HEAVE * el, -SPEED_CAP, SPEED_CAP)
            # Creep in unless badly off-bearing (then turn first) or avoiding a blue — keeps closing
            # (and driving heave toward the balloon's depth) while it refines the centre.
            surge = 0.0 if (abs(az) > FACE_TOL or blue is not None) else ALIGN_CREEP
            return {"surge": surge, "heave": heave, "yaw": yaw}, self._info(blue)

        # APPROACH (far): yaw onto the bearing, surge (throttled while mis-pointed), heave to centre.
        self.state = "APPROACH"
        self._align_steps = 0
        yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate + avoid_yaw, -1, 1)
        heave = _clip(KP_HEAVE * el, -SPEED_CAP, SPEED_CAP)
        surge = SPEED_CAP * max(0.0, math.cos(az))
        if abs(az) > FACE_TOL:
            surge *= 0.3
        if blue is not None:
            surge *= 0.5
        return {"surge": surge, "heave": heave, "yaw": yaw}, self._info(blue)

    # -- post-ram confirmation (CAMERA-based pop vs miss) --------------------------------------
    def _confirm(self, yaw_rate, avoid_yaw, blue):
        """Back off gently and watch: the target reappearing = MISS; gone CONFIRM_FRAMES = POP."""
        self._confirm_frames += 1
        if self.trk.misses == 0:  # a matching detection came back -> it did NOT pop -> MISS
            self.n_miss += 1
            self._attempts += 1
            self._enter_recover()
            return self._recover(yaw_rate, avoid_yaw, blue)
        if self._confirm_frames >= CONFIRM_FRAMES:  # gone long enough -> camera-confirmed POP
            self.n_confirmed_pop += 1
            self._clear_track()
            self._reset_pursuit()
            self.state = "SEARCH"
            self._start_sweep()
            return self._search(yaw_rate, self.dt, avoid_yaw, blue)
        # Firmly reverse (holding the last bearing) so a merely-passed balloon comes back into view.
        yaw = _clip(0.6 * KP_YAW * self.trk.az + KD_YAW * yaw_rate, -1, 1)
        return {"surge": -CONFIRM_SURGE, "heave": 0.0, "yaw": yaw}, self._info(blue)

    # -- recover (back off + re-align, or search if the target is gone) ------------------------
    def _recover(self, yaw_rate, avoid_yaw, blue):
        self._recover_steps -= 1
        if self.trk.alive():
            yaw = _clip(0.8 * KP_YAW * self.trk.az + KD_YAW * yaw_rate + avoid_yaw, -1, 1)
            heave = _clip(KP_HEAVE * self.trk.el, -SPEED_CAP, SPEED_CAP)
        else:
            yaw, heave = self._sweep_dir * SEARCH_YAW, 0.0
        if self._recover_steps <= 0:
            if self.trk.alive():
                self.state = "ALIGN"       # re-align and retry
                self._committed = False
            else:
                self.state = "SEARCH"
                self._clear_track()
                self._start_sweep()
        return {"surge": -RECOVER_SURGE, "heave": heave, "yaw": yaw}, self._info(blue)

    # -- SEARCH handler ------------------------------------------------------------------------
    def _search(self, yaw_rate, dt, avoid_yaw, blue):
        """Deliberate exploration: full 360° in-place sweep (height-scanning), then translate."""
        if self._translating > 0:
            self._translating -= 1
            return ({"surge": SEARCH_SURGE, "heave": 0.0, "yaw": 0.4 * avoid_yaw}, self._info(blue))
        self._swept += abs(yaw_rate) * dt
        self._scan_phase += SCAN_RATE * dt
        heave = SCAN_HEAVE * math.sin(self._scan_phase)  # scan the different balloon heights
        yaw = self._sweep_dir * SEARCH_YAW + 0.4 * avoid_yaw
        if self._swept >= 2.0 * math.pi:  # a full sweep found nothing -> go somewhere new
            self._sweep_cycles += 1
            self._sweep_dir *= -1.0       # alternate direction (cover new ground)
            self._swept = 0.0
            self._translating = TRANSLATE_STEPS
        return {"surge": 0.0, "heave": heave, "yaw": yaw}, self._info(blue)


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x
