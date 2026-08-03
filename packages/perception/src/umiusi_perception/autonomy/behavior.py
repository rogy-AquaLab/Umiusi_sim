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
from dataclasses import dataclass, field, replace

from umiusi_perception.tracker import ASSOC_BEARING, Tracker, plausible_detections

POSITIVE_COLOURS = ("red", "yellow")

# --- detection range gate + target selection (rule-based, NOT a reward) ------------------------
# Detections beyond MAX_TARGET_RANGE are DROPPED: they are mostly false positives (the detector's
# far recall is poor) and unreachable anyway — gating them cuts the noise and focuses the FSM on
# balloons it can actually reach. Selection is then simply "the NEAREST reachable NON-BLUE track"
# (see ``_pick_target``): on the new ~19-balloon field, maximising COUNT/THROUGHPUT (clear the field
# near-to-far) beats chasing a far red past several near yellows — the 3-5 min budget rewards popping
# whatever is closest. Popping near-to-far also CLEARS the field as we advance, so far fewer un-popped
# balloons are left ahead to under-pass (the wire-entanglement lever; see the path guard below). Blue
# is strictly avoided (never a target, -10).
MAX_TARGET_RANGE = 4.5    # m; ignore any detection beyond this (far FP noise + unreachable)
# --- wire-avoidance path guard (competition-critical: tethers tangle if we under-pass a balloon) --
# Passing UNDER an un-popped balloon risks snagging its vertical tether wire (esp. the tall 1.5 m
# yellows). Two FSM mechanisms keep us off the wires:
#   * PREEMPT: while approaching, always re-lock onto the NEAREST non-blue track — so an un-popped
#     balloon that lies closer (i.e. in the straight path) gets popped FIRST, clearing the wire.
#   * DETOUR: steer laterally around any close, dead-ahead un-popped balloon that is NOT our current
#     ram target (a blue decoy, or a balloon we've given up on) so we arc around its wire, not under it.
# PATH_CLEAR_RADIUS is the horizontal keep-out around a wire we plan the detour against; it is the
# scenario's TETHER_RADIUS (0.20 m capture zone) plus a hull-half-width margin so the vehicle body,
# not just its centre, clears the wire. Kept local so the FSM stays ROS/scenario-decoupled.
PATH_CLEAR_RADIUS = 0.40  # m; horizontal keep-out corridor around an un-popped balloon's tether
PREEMPT_MARGIN = 0.30     # m; re-lock to a nearer non-blue target only if it is at least this closer
#                           (hysteresis — stops lock thrashing between two similar-range balloons)

# --- drive gains (feed-forward command convention; mirrors competition_run) -------------------
SPEED_CAP = 0.35          # max surge/heave command magnitude ("modest speed")
KP_YAW = 1.1              # yaw P gain: yaw command per radian of bearing error
KD_YAW = 0.15             # yaw D gain (damps the measured yaw rate)
KP_HEAVE = 1.5            # heave P gain: command per radian of target elevation error (get to depth).
#                           A touch stronger than the old 1.4 so the vehicle wins the DEPTH race —
#                           reaches the balloon's height (esp. the tall 1.5 m yellows) before closing.
FACE_TOL = math.radians(45.0)   # surge hard only once within this bearing error (APPROACH)
# --- SEARCH / exploration ---------------------------------------------------------------------
SEARCH_YAW = 0.5          # in-place yaw-sweep rate command
SEARCH_SURGE = 0.30       # forward speed while translating to a fresh spot between sweeps
SCAN_HEAVE = 0.15         # heave amplitude while sweeping (scan the different balloon heights)
SCAN_RATE = 1.5           # rad/s of the height-scan oscillation
TRANSLATE_STEPS = 50      # steps (~1 s @50 Hz) to translate before the next sweep
# --- closeness by BBOX (more reliable than the noisy range estimate) --------------------------
ALIGN_BBOX = 0.18         # bbox height / frame >= this -> close: start the slow centred ALIGN
RAM_COMMIT_BBOX = 0.26    # ...and >= this AND centred (and settled) -> commit to the RAM
RAM_MAX_STEPS = 85        # committed and still visible this long (never popped) -> treat as a MISS
PASS_PEAK_BBOX = 0.45     # a "clearly passed it" MISS needs the balloon to have filled this much...
PASS_DROP_FRAC = 0.5      # ...then shrunk below this fraction of that peak while STILL visible
CENTRE_AZ = math.radians(6.0)   # "centred" = bearing within this (tighter -> more head-on hits)...
CENTRE_EL = math.radians(6.0)   # ...and elevation within this. Tight because the pin sits ~0.3 m
#                           ahead of the camera, so a small el at commit grows into a vertical MISS
#                           over the blind lunge; camera + pin are near co-located (both y~0.1 m), so
#                           a truly el~0 approach puts the pin ON the balloon.
COMMIT_EL = math.radians(12.0)  # elevation band to COMMIT to the ram: looser than CENTRE_EL because
#                           the el-scaled surge (below) already refuses to close fast while off-depth,
#                           so committing early is safe and lets the drive-THROUGH lunge fire even on
#                           the tall (1.5 m) yellows that rarely settle inside CENTRE_EL.
ALIGN_CREEP = 0.12        # small forward creep while aligning (keeps closing as it centres)
ALIGN_TIMEOUT = 110       # steps stuck in ALIGN un-centred -> REPOSITION (back off, new line)
# --- final ALIGN -> settle -> lunge (raise the head-on physical-hit rate) ----------------------
# The pin tip sits ~0.3 m ahead of the camera, so a balloon "popped" (pin within ~0.13 m of centre)
# is reached only AFTER the camera range drops well below the RAM-commit range — by which point the
# balloon has filled / left the frame and the detector drops it. So once centred + close we (a)
# SETTLE briefly (near-stationary, drive az/el to ~0) to kill approach overshoot, then (b) LUNGE
# straight in at full surge and DRIVE THROUGH the sight-loss for LUNGE_STEPS to actually make
# contact BEFORE backing off to judge pop vs miss. Without the drive-through the pin never lands.
# Forward surge is SCALED DOWN when off-depth so the vehicle matches the balloon's HEIGHT first
# (climb/descend toward it) and only then closes level — the pin then reaches the balloon's front
# face head-on instead of passing under/over it while still changing depth (the single biggest
# head-on-hit lever for the tall 1.5 m yellows). Full surge at |el| <= EL_FULL_SURGE, tapering to
# EL_MIN_SURGE by EL_ZERO_SURGE. Because heave (KP_HEAVE) is far stronger than the floored surge,
# depth still converges first even at the floor — the floor just keeps closing so throughput (esp.
# on the low reds, which approach fine) is not sacrificed. Smooth taper, never a hard stall.
# DEPTH-FIRST gate (the tall-yellow lever): the surge is measured against the AIM-point elevation error
# (el + aim bias), so the vehicle must be near the balloon's HEIGHT before it closes the last of the
# horizontal gap. The floor is LOWER than the old 0.35 (climb/descend more before closing) but NOT
# near-zero — a near-zero floor stalls the approach so long the vehicle arrives high/slow and either
# passes OVER the balloon or never builds the closing speed a pop needs (measured: it cratered the pop
# rate). 0.22 keeps it approaching, just biased toward matching depth first; heave (KP_HEAVE, stronger)
# still wins the depth race. Full surge once |el_err| <= EL_FULL_SURGE — level onto the front face.
EL_FULL_SURGE = math.radians(5.0)
EL_ZERO_SURGE = math.radians(15.0)
EL_MIN_SURGE = 0.22       # surge floor when far off-depth — lower than old 0.35 (climb more), not a stall
SETTLE_STEPS = 3          # hold centred + near-stationary this long before the lunge (kill overshoot);
#                           kept short so we commit before the close-range detector flicker aborts us
LUNGE_STEPS = 26          # after commit, keep driving straight in (even blind) this long to reach contact
RAM_SURGE = 0.26          # committed-approach surge: LESS than SPEED_CAP so heave has time to drive
#                           el->0 (get the pin to the balloon's HEIGHT) BEFORE the pin reaches the
#                           balloon plane — a full-surge lunge overshoots in x and the pin passes
#                           UNDER a tall balloon (a vertical/glancing miss, not a head-on pop).
# --- aim slightly from ABOVE (the tall-1.5 m-yellow fix) --------------------------------------
# Bias the aim a touch ABOVE the balloon centre so the vehicle sits slightly high and DESCENDS onto
# the front/upper face — the pin then meets a TALL (1.5 m) yellow instead of passing under it (the
# pin sits at hull mid-height y~0.1 m; a level approach to a 1.5 m balloon tends to arrive low as
# depth still settles). The heave loop targets el = -AIM_EL_BIAS (balloon a hair below the optic axis
# at contact => pin above centre), and the centred/commit elevation tests use the SAME biased error
# so the FSM commits when it is centred on the AIM point, not the raw centre.
#
# The bias is applied ONLY to yellows (``_aim_bias``): they are the tall balloons that need to be met
# from above. The reds sit low (0.5 m) and are already approached from the start height coming DOWN,
# so a level, dead-centre aim keeps the pin-axis error small at contact — adding an above-bias there
# would only shift the tip a couple cm high at close range, which the near-frontal cone reads as extra
# angle and can push a good ram past POP_ANGLE_TOL_DEG. So: yellows get the above-bias, reds/others
# aim at the centre. 5 deg is small (~3 cm vertical at the ~0.3 m contact range — well inside the
# 0.13 m pop sphere), enough to catch the upper face of a tall yellow without sitting so high that the
# pin passes OVER it or the near-frontal cone (POP_ANGLE_TOL_DEG=20) rejects a downward-angled hit.
# (An aggressive 8 deg over-corrected — the vehicle rammed from too far above and stopped popping;
# measured pop rate cratered. 5 deg is the modest nudge up from the old 4 that still helps.)
RAM_AIM_ABOVE_DEG = 5.0
AIM_EL_BIAS = math.radians(RAM_AIM_ABOVE_DEG)
AIM_ABOVE_COLOURS = ("yellow",)  # colours the above-bias applies to (the tall balloons)
# --- recover / confirm / give-up --------------------------------------------------------------
RECOVER_SURGE = 0.28      # reverse speed while backing off a missed ram
RECOVER_STEPS = 30        # steps (~0.6 s) to back off before re-aligning
CONFIRM_FRAMES = 55       # target gone this many frames while backing off a ram -> camera POP. Long
#                           (~0.8 m back-off) so a merely out-of-frame / passed balloon RE-ENTERS the
#                           view (-> MISS); only a truly popped (hidden) one stays gone -> POP. A
#                           longer back-off only converts FALSE pops to honest misses, never the
#                           reverse, so it strictly improves camera-confirmed-pop accuracy.
CONFIRM_SURGE = 0.34      # firm reverse during CONFIRM: a merely-passed (not popped) balloon must
#                           come BACK into view (-> re-associates -> MISS); only a truly popped
#                           (hidden) one stays gone the whole back-off -> confirmed pop.
CONFIRM_MIN_PEAK = RAM_COMMIT_BBOX  # only believe a POP if we actually got to ram range (else the
#                           disappearance is an out-of-frame / detector dropout, not a real pop).
CONFIRM_EDGE = math.radians(18.0)   # ...and only if the target was near-CENTRED (not drifting to a
#                           frame edge) when it vanished: a real head-on pop disappears from the
#                           CENTRE; a tall balloon we passed under leaves toward the TOP edge (large
#                           |el|) -> that is a MISS, not a pop. Kills false "popped a tall yellow".
MAX_ATTEMPTS = 6          # ram attempts on one target before abandoning it (retry a near one more)
MAX_PURSUIT_STEPS = 500   # ~10 s pursuing one target before abandoning it
# --- give-up memory (don't get TRAPPED on an unreachable colour) -------------------------------
# ``_attempts`` resets whenever we briefly lose then re-acquire a target, so on its own it can never
# abandon a balloon we keep re-locking (e.g. a near tall yellow the pin can't reach — bird in the
# hand that turns out to be a stone). A per-COLOUR failure count PERSISTS across re-acquires: after
# GIVEUP_FAILS misses on a colour we SUPPRESS it for SUPPRESS_STEPS so selection falls through to a
# reachable alternative (usually the red), then re-allow it. We still TRY the near bird-in-hand
# first — this only stops us looping on it forever when it won't pop.
GIVEUP_FAILS = 4          # misses on one colour (across re-acquires) before we look elsewhere
SUPPRESS_STEPS = 350      # ~7 s to prefer other colours after giving up on one
# --- blue avoidance ---------------------------------------------------------------------------
AVOID_AZ = math.radians(28.0)   # a blue within this bearing of dead-ahead is "in the way"
AVOID_RANGE = 1.6         # ...and closer than this -> steer around it
AVOID_YAW = 0.5           # avoidance yaw magnitude added away from the blue
# --- temporal track (the ONE mechanism: see umiusi_perception.tracker) ---------------------
# Multi-frame association, acquisition-vote FP rejection AND lock-hold persistence now all live in
# ``perception.tracker.Tracker`` (ROS/sim-free, reused by the robot nodes). The FSM consumes CONFIRMED
# tracks from it. The tuned thresholds live there: association gate (ASSOC_BEARING/ASSOC_RANGE_FRAC),
# confirmation (CONFIRM_VOTES / VOTE_MAX — folds in the old acquisition vote) and persistence
# (PERSIST_FRAMES — folds in the old LOCK_HOLD_STEPS). ``_Track`` below is now just the FSM's pursuit
# SNAPSHOT of the locked track (updated from the tracker each step, retained through a drop so the
# post-ram CONFIRM can keep watching), NOT a second tracker.


@dataclass
class _Track:
    """Pursuit snapshot of the locked target (synced from the tracker; see the note above)."""
    colour: str = ""
    az: float = 0.0
    el: float = 0.0
    range_m: float = float("inf")
    bbox_frac: float = 0.0          # bbox height / frame height (apparent closeness)
    peak_bbox: float = 0.0          # max bbox_frac seen this pursuit (for the miss test)
    misses: int = 999               # consecutive frames with no associated detection


@dataclass
class BalloonBehavior:
    """Stateful perception FSM. Call ``step(detections, yaw_rate, heading, dt)`` per control step."""

    frame_h: int = 240
    frame_w: int = 320
    fovy_deg: float = 60.0
    dt: float = 0.02
    # PIN-AWARE aiming: the calibrated body-frame offset (dx, dy, dz) = pin_tip - camera. When set, every
    # detection's bearing is re-expressed FROM THE PIN before tracking/control, so the FSM drives the PIN
    # (not the camera optic axis) onto the balloon — decoupling where the pin is mounted (choose it for
    # camera FOV) from how it is aimed. None = legacy camera-centring (unchanged behaviour).
    pin_offset: tuple | None = None
    state: str = "SEARCH"
    trk: _Track = field(default_factory=_Track)
    tracker: Tracker = field(default_factory=Tracker)  # the ONE multi-frame tracker
    _target_id: object = None                   # tracker id of the locked target (None = unlocked)
    _alive: bool = False                        # is the locked target currently tracked/persisting?
    fails: dict = field(default_factory=dict)   # per-colour persistent miss count (give-up memory)
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
    _settle_steps: int = 0          # steps held centred+still before the lunge (overshoot kill)
    _committed: bool = False        # have we committed to ramming this target?
    _ram_steps: int = 0             # steps since the current RAM commit (align-loss lunge counts too)
    _recover_steps: int = 0
    _confirm_frames: int = 0
    _suppress_colour: str = ""      # colour currently suppressed by the give-up memory
    _suppress_steps: int = 0        # steps left on that suppression
    # stats (for the run report)
    n_ram: int = 0
    n_recover: int = 0
    n_abandon: int = 0
    n_confirmed_pop: int = 0        # camera-confirmed pops (FSM's own belief; NOT the GT score)
    n_miss: int = 0

    # -- target selection / locked-track sync (over the tracker's CONFIRMED tracks) -------------
    def _pick_target(self):
        """Selection over the tracker's CONFIRMED tracks: the NEAREST reachable NON-BLUE track.

        On the ~19-balloon field, throughput wins — pop whatever is closest (near-to-far) rather than
        chase a far red past several near yellows. This also clears the field as we advance, so fewer
        un-popped balloons are left ahead to under-pass (wire avoidance). Only POSITIVE_COLOURS are
        selectable (blue is tracked solely for avoidance, never a target). Returns a Track or None.
        Consuming CONFIRMED tracks (not raw detections) keeps a flickery one-frame false positive from
        ever steering selection."""
        eligible = [t for t in self.tracker.confirmed() if t.colour in POSITIVE_COLOURS]
        if self._suppress_steps > 0 and self._suppress_colour:  # give-up memory: try other colours
            eligible = [t for t in eligible if t.colour != self._suppress_colour]
        if not eligible:
            return None
        return min(eligible, key=lambda t: t.range_m)           # nearest reachable non-blue

    def _lock(self, track):
        """Lock the FSM onto a tracker track: seed the pursuit snapshot and start the approach."""
        self._target_id = track.id
        f = self._bbox_frac(track.bbox)
        self.trk = _Track(colour=track.colour, az=track.az, el=track.el, range_m=track.range_m,
                          bbox_frac=f, peak_bbox=f, misses=track.misses)
        self._alive = True
        self._reset_pursuit()
        self.state = "APPROACH"

    def _sync_track(self):
        """Refresh the pursuit snapshot from the tracker's locked track. If the tracker has DROPPED
        the id, re-adopt any current track matching the snapshot (same colour, nearest bearing in the
        gate) — this bridges a >persist dropout so the post-ram CONFIRM stays honest. If nothing
        matches, keep the frozen snapshot and count the miss (so CONFIRM's long watch can complete)."""
        if not self.trk.colour:
            self._alive = False
            return
        t = self.tracker.get(self._target_id)
        if t is None:
            t = self._match_locked()
        if t is not None:
            self._target_id = t.id
            self.trk.colour = t.colour
            self.trk.az, self.trk.el, self.trk.range_m = t.az, t.el, t.range_m
            self.trk.bbox_frac = self._bbox_frac(t.bbox)
            self.trk.peak_bbox = max(self.trk.peak_bbox, self.trk.bbox_frac)
            self.trk.misses = t.misses
            self._alive = True
        else:
            self.trk.misses += 1
            self._alive = False

    def _match_locked(self):
        """A current tracker track matching the locked snapshot (same colour, closest bearing in gate)."""
        best, best_d = None, ASSOC_BEARING
        for t in self.tracker.tracks:
            if t.colour != self.trk.colour:
                continue
            dd = math.hypot(t.az - self.trk.az, t.el - self.trk.el)
            if dd < best_d:
                best, best_d = t, dd
        return best

    def _bbox_frac(self, bbox):
        u0, v0, u1, v1 = bbox
        return (v1 - v0) / max(1, self.frame_h)

    def _blue_avoidance(self):
        """Yaw bias to steer around a CONFIRMED blue that is roughly dead-ahead and close. 0 if clear.
        Uses confirmed blue tracks (not raw detections) so a one-frame blue FP can't jerk the heading,
        while a real blue decoy — seen over many frames as we near it — is reliably avoided."""
        threats = [t for t in self.tracker.confirmed()
                   if t.colour == "blue" and abs(t.az) < AVOID_AZ and t.range_m < AVOID_RANGE]
        if not threats:
            return 0.0, None
        b = min(threats, key=lambda t: t.range_m)
        side = -1.0 if b.az >= 0 else 1.0  # turn away; dead-centre -> deterministic left
        gain = AVOID_YAW * (1.0 - min(1.0, b.range_m / AVOID_RANGE) * 0.5)
        return side * gain, b

    def _wire_avoidance(self):
        """Yaw bias to arc AROUND (not under) any close, dead-ahead un-popped balloon that is NOT the
        current ram target — a balloon in our path whose vertical tether we would otherwise snag by
        under-passing it (the tall yellows especially). The current target is excluded: we ram it
        head-on and pop it, which removes its wire. Uses the lateral offset of each confirmed track
        from dead-ahead: a track within PATH_CLEAR_RADIUS of the path AND closer than AVOID_RANGE is a
        wire to dodge. Blue decoys are handled by ``_blue_avoidance``; this catches the un-poppable
        (suppressed / off-bearing) balloons we would otherwise drive straight under. 0 if the path is
        clear."""
        threats = []
        for t in self.tracker.confirmed():
            if t.id == self._target_id or t.colour == "blue":
                continue                       # our target (popped head-on) / blues (own handler)
            if not (math.isfinite(t.range_m) and t.range_m < AVOID_RANGE):
                continue
            lateral = t.range_m * math.sin(abs(t.az))   # horizontal offset from dead-ahead
            if abs(t.az) < AVOID_AZ and lateral < PATH_CLEAR_RADIUS:
                threats.append(t)
        if not threats:
            return 0.0
        b = min(threats, key=lambda t: t.range_m)
        side = -1.0 if b.az >= 0 else 1.0      # turn away from it; dead-centre -> deterministic left
        return side * AVOID_YAW * (1.0 - min(1.0, b.range_m / AVOID_RANGE) * 0.5)

    def _info(self, blue):
        return {"state": self.state, "target": self.trk.colour or None, "range": self.trk.range_m,
                "az": self.trk.az, "bbox": round(self.trk.bbox_frac, 2), "attempts": self._attempts,
                "held": self.trk.misses > 0, "blue_threat": blue is not None}

    # -- transitions ---------------------------------------------------------------------------
    def _reset_pursuit(self):
        self._pursuit_steps = 0
        self._attempts = 0
        self._align_steps = 0
        self._settle_steps = 0
        self._committed = False

    def _register_fail(self):
        """Record a miss/abandon on the current target's COLOUR (persists across re-acquires). After
        GIVEUP_FAILS on a colour, suppress it briefly so selection tries a reachable alternative."""
        c = self.trk.colour
        if not c:
            return
        self.fails[c] = self.fails.get(c, 0) + 1
        if self.fails[c] >= GIVEUP_FAILS:
            self._suppress_colour = c
            self._suppress_steps = SUPPRESS_STEPS
            self.fails[c] = 0

    def _start_sweep(self):
        """(Re)start an in-place sweep, biased toward where the target was last seen."""
        self._swept = 0.0
        self._scan_phase = 0.0
        self._translating = 0
        self._sweep_dir = 1.0 if self.trk.az >= 0 else -1.0

    def _clear_track(self):
        self.trk = _Track()
        self._target_id = None
        self._alive = False

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
        self._register_fail()                # abandoning also feeds the give-up memory for its colour
        self._reset_pursuit()
        self.state = "SEARCH"
        self._start_sweep()
        self._sweep_dir *= -1.0              # look the other way than the abandoned target
        self._translating = TRANSLATE_STEPS  # ...and move to a fresh area first
        self._clear_track()

    # -- main tick -----------------------------------------------------------------------------
    def step(self, detections, yaw_rate, heading=0.0, dt=None, fresh=True):
        """Return (command, info). command = {surge, heave, yaw}. Camera-only decisions.

        ``fresh`` is True on a fresh perception frame and False when the caller is re-driving on the
        HELD detections between detector ticks — confirmation votes advance only on fresh frames."""
        dt = self.dt if dt is None else dt
        # PIN-AWARE aiming (optional): re-express each detection's bearing FROM THE PIN before anything
        # else, so tracking + every control decision drives the PIN onto the balloon (see pin_offset).
        if self.pin_offset is not None:
            detections = [_pin_relative(d, self.pin_offset) for d in detections]
        # RANGE GATE + SIZE-CONSISTENCY FILTER: drop far detections (mostly false positives +
        # unreachable) AND boxes whose shape/size is implausible for a balloon at their range, up
        # front, so the tracker (and every downstream decision) only sees reachable, plausible boxes.
        detections = plausible_detections(detections, frame_h=self.frame_h, frame_w=self.frame_w,
                                          fovy_deg=self.fovy_deg, max_range_m=MAX_TARGET_RANGE)
        # ONE multi-frame mechanism: associate -> confirm -> persist. Votes advance only on fresh
        # frames; the persistence miss-counter ages every control step (held frames included).
        self.tracker.update(detections, fresh=fresh)
        avoid_yaw, blue = self._blue_avoidance()
        if self._suppress_steps > 0:
            self._suppress_steps -= 1       # give-up memory decays -> the colour becomes eligible again

        # Refresh the locked-target snapshot from the tracker (holds identity through dropouts).
        self._sync_track()

        # Post-ram camera confirmation (pop vs miss) — pure camera, no ground truth.
        if self.state == "CONFIRM":
            return self._confirm(yaw_rate, avoid_yaw, blue)
        if self.state == "RECOVER":
            return self._recover(yaw_rate, avoid_yaw, blue)

        # Acquire a target when searching (rule-based, over CONFIRMED tracks only).
        if self.state == "SEARCH" or not self.trk.colour:
            target = self._pick_target()
            if target is None:
                self.state = "SEARCH"
                return self._search(yaw_rate, dt, avoid_yaw, blue)
            self._lock(target)

        # Lost the track while pursuing. If we had COMMITTED to a ram, keep driving straight in
        # (blind, on the frozen last bearing) until the lunge window is spent — the pin only reaches
        # the balloon AFTER it leaves the frame — THEN back off to confirm. Otherwise recover.
        if not self._alive:
            if self._committed:
                if self.state == "RAM" and self._ram_steps < LUNGE_STEPS:
                    self._ram_steps += 1  # DRIVE THROUGH the sight-loss to make physical contact
                    yaw = _clip(KP_YAW * self.trk.az + KD_YAW * yaw_rate, -1, 1)
                    heave = _heave_cmd(self.trk.el, self.trk.colour)
                    return {"surge": RAM_SURGE, "heave": heave, "yaw": yaw}, self._info(blue)
                self._enter_confirm()
                return self._confirm(yaw_rate, avoid_yaw, blue)
            self._attempts += 1
            self._enter_recover()
            return self._recover(yaw_rate, avoid_yaw, blue)

        self._pursuit_steps += 1
        if self._pursuit_steps > MAX_PURSUIT_STEPS or self._attempts >= MAX_ATTEMPTS:
            self._abandon()
            return self._search(yaw_rate, dt, avoid_yaw, blue)

        # PREEMPT (nearest-first + wire avoidance): while still approaching (not close / not committed
        # to a ram), if a DIFFERENT non-blue track is meaningfully NEARER than the locked target,
        # switch to it. Popping the nearest first clears balloons — especially any lying in our path —
        # as we advance, instead of driving under them toward a farther target and snagging a tether.
        if not self._committed and self.trk.bbox_frac < ALIGN_BBOX:
            cand = self._pick_target()
            if (cand is not None and cand.id != self._target_id
                    and cand.range_m < self.trk.range_m - PREEMPT_MARGIN):
                self._lock(cand)

        az, el, bbox = self.trk.az, self.trk.el, self.trk.bbox_frac
        # Elevation is measured against the AIM point (biased AIM_EL_BIAS above centre), so the FSM
        # commits/centres when pointed at the aim, not the raw centre — descend onto the upper face.
        el_err = el + _aim_bias(self.trk.colour)
        centred = abs(az) < CENTRE_AZ and abs(el_err) < CENTRE_EL   # head-on enough for a clean pop
        commit_ok = abs(az) < CENTRE_AZ and abs(el_err) < COMMIT_EL  # good enough to commit the ram

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
                self._register_fail()
                self._enter_recover()
                return self._recover(yaw_rate, avoid_yaw, blue)
            # Tight tracking through the ram (full gains) so it stays head-on to contact; surge is
            # eased off if depth drifts (el grows) so the pin stays level to the balloon's front face.
            yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate, -1, 1)
            heave = _heave_cmd(el, self.trk.colour)
            return {"surge": RAM_SURGE * _el_surge_scale(el, self.trk.colour), "heave": heave,
                    "yaw": yaw}, self._info(blue)

        # Close enough (bbox) AND pointed head-on: SETTLE briefly (near-stationary, drive az/el to
        # ~0) to kill approach overshoot, then COMMIT to the RAM lunge. If centring slips during the
        # settle, fall through to ALIGN and re-earn it.
        if bbox >= RAM_COMMIT_BBOX and commit_ok:
            self._settle_steps += 1
            self._committed = True
            if self._settle_steps < SETTLE_STEPS:
                self.state = "ALIGN"
                yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate, -1, 1)
                heave = _heave_cmd(el, self.trk.colour)
                return {"surge": 0.5 * ALIGN_CREEP, "heave": heave, "yaw": yaw}, self._info(blue)
            self.state = "RAM"
            self._align_steps = 0
            self._ram_steps = 0
            self.n_ram += 1
            yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate, -1, 1)
            heave = _heave_cmd(el, self.trk.colour)
            return {"surge": RAM_SURGE, "heave": heave, "yaw": yaw}, self._info(blue)
        self._settle_steps = 0  # lost closeness/centring -> restart the settle next time

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
            heave = _heave_cmd(el, self.trk.colour)
            # Creep in, but scaled down by depth error — level onto the balloon's height first, then
            # close. Hold entirely if badly off-bearing (turn first) or avoiding a blue.
            surge = (0.0 if (abs(az) > FACE_TOL or blue is not None)
                     else ALIGN_CREEP * _el_surge_scale(el, self.trk.colour))
            return {"surge": surge, "heave": heave, "yaw": yaw}, self._info(blue)

        # APPROACH (far): yaw onto the bearing, heave to the target's depth, and surge in — but scale
        # the surge by depth error too, so a tall/low balloon is met by CLIMBING/DESCENDING to its
        # height first and only then closing level (otherwise we arrive at its range still off-depth
        # and the pin passes under/over it). Throttle further while mis-pointed or avoiding a blue.
        self.state = "APPROACH"
        self._align_steps = 0
        # Steer around (not under) any un-popped balloon in our path to the target — its tether would
        # otherwise snag as we under-pass it (blue decoys handled separately by ``avoid_yaw``).
        wire_yaw = self._wire_avoidance()
        yaw = _clip(KP_YAW * az + KD_YAW * yaw_rate + avoid_yaw + wire_yaw, -1, 1)
        heave = _heave_cmd(el, self.trk.colour)
        surge = SPEED_CAP * max(0.0, math.cos(az)) * _el_surge_scale(el, self.trk.colour)
        if abs(az) > FACE_TOL:
            surge *= 0.3
        if blue is not None or wire_yaw != 0.0:
            surge *= 0.5    # ease off while arcing around a blue / an in-path wire
        return {"surge": surge, "heave": heave, "yaw": yaw}, self._info(blue)

    # -- post-ram confirmation (CAMERA-based pop vs miss) --------------------------------------
    def _confirm(self, yaw_rate, avoid_yaw, blue):
        """Back off and watch: the target reappearing = MISS; gone CONFIRM_FRAMES = POP.

        A pop is only BELIEVED if we actually reached ram range (peak_bbox >= CONFIRM_MIN_PEAK). A
        disappearance from a target we never got close to is far more likely an out-of-frame /
        detector dropout than a real pop, so it is treated as a MISS and retried, not scored — this
        keeps the camera-confirmed-pop count honest. While backing off we also re-drive heave toward
        the last elevation so a still-present balloon that left the top of the frame comes back."""
        self._confirm_frames += 1
        # A credible pop: we reached ram range AND the target vanished from the CENTRE (head-on),
        # not by drifting to a frame edge. Anything else that "stayed gone" is an out-of-frame /
        # detector dropout -> retry as a MISS rather than claim a false pop.
        credible = (self.trk.peak_bbox >= CONFIRM_MIN_PEAK
                    and abs(self.trk.el) <= CONFIRM_EDGE and abs(self.trk.az) <= CONFIRM_EDGE)
        if self.trk.misses == 0 or (not credible and self._confirm_frames >= CONFIRM_FRAMES):
            # matching detection came back (MISS), or it wasn't a credible pop and stayed gone.
            self.n_miss += 1
            self._attempts += 1
            self._register_fail()
            self._enter_recover()
            return self._recover(yaw_rate, avoid_yaw, blue)
        if self._confirm_frames >= CONFIRM_FRAMES:  # credible AND gone long enough -> camera POP
            self.n_confirmed_pop += 1
            self.fails.pop(self.trk.colour, None)   # success clears that colour's give-up memory
            self._clear_track()
            self._reset_pursuit()
            self.state = "SEARCH"
            self._start_sweep()
            return self._search(yaw_rate, self.dt, avoid_yaw, blue)
        # Firmly reverse (holding the last bearing + elevation) so a merely-passed balloon that left
        # the frame comes back into view (-> re-associates -> MISS).
        yaw = _clip(0.6 * KP_YAW * self.trk.az + KD_YAW * yaw_rate, -1, 1)
        heave = _heave_cmd(self.trk.el, self.trk.colour)
        return {"surge": -CONFIRM_SURGE, "heave": heave, "yaw": yaw}, self._info(blue)

    # -- recover (back off + re-align, or search if the target is gone) ------------------------
    def _recover(self, yaw_rate, avoid_yaw, blue):
        self._recover_steps -= 1
        if self._alive:
            yaw = _clip(0.8 * KP_YAW * self.trk.az + KD_YAW * yaw_rate + avoid_yaw, -1, 1)
            heave = _heave_cmd(self.trk.el, self.trk.colour)
        else:
            yaw, heave = self._sweep_dir * SEARCH_YAW, 0.0
        if self._recover_steps <= 0:
            if self._alive:
                self.state = "ALIGN"       # re-align and retry
                self._committed = False
            else:
                self.state = "SEARCH"
                self._clear_track()
                self._start_sweep()
        return {"surge": -RECOVER_SURGE, "heave": heave, "yaw": yaw}, self._info(blue)

    # -- SEARCH handler ------------------------------------------------------------------------
    def _search(self, yaw_rate, dt, avoid_yaw, blue):
        """Deliberate exploration: full 360° in-place sweep (height-scanning), then translate. While
        TRANSLATING (the only forward motion in SEARCH) it steers around un-popped balloons in its
        path so relocating across the field does not drive under their tethers."""
        if self._translating > 0:
            self._translating -= 1
            wire_yaw = self._wire_avoidance()
            surge = SEARCH_SURGE * (0.5 if wire_yaw != 0.0 else 1.0)
            return ({"surge": surge, "heave": 0.0, "yaw": 0.4 * avoid_yaw + wire_yaw},
                    self._info(blue))
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


def _pin_relative(det, offset):
    """A copy of ``det`` with its bearing re-expressed FROM THE PIN. Reconstruct the balloon's 3-D
    position from the camera bearing (az, el) + range, subtract the body-frame ``offset`` = pin_tip -
    camera (camera axes: x forward, y up, z right), and re-bear. The bbox/range are unchanged (apparent
    SIZE is viewpoint-independent to first order; only the DIRECTION shifts). Infinite range -> no-op.

    NOTE: the parallax reconstruction AMPLIFIES bearing noise at close range, so pin-aware aiming needs
    an accurate detector bearing (<=~2 deg); fading the correction near contact was measured to re-open
    the under-pass, so it is intentionally NOT done. Gate pin-aware on the detector's bearing accuracy."""
    az, el = det.bearing
    r = det.range_m
    if not math.isfinite(r):
        return det
    ce = math.cos(el)
    x = r * ce * math.cos(az) - offset[0]
    y = r * math.sin(el) - offset[1]
    z = r * ce * math.sin(az) - offset[2]
    return replace(det, bearing=(math.atan2(z, x), math.atan2(y, math.hypot(x, z))))


def _clip(x, lo, hi):
    return lo if x < lo else hi if x > hi else x


def _aim_bias(colour):
    """Elevation aim bias for the target colour: AIM_EL_BIAS above centre for the tall balloons
    (yellows), 0 for the low reds/others (aim dead centre — keeps the pin-axis error small)."""
    return AIM_EL_BIAS if colour in AIM_ABOVE_COLOURS else 0.0


def _heave_cmd(el, colour):
    """Heave command driving the target toward the AIM point — biased ``_aim_bias(colour)`` ABOVE the
    balloon centre. Equilibrium at el = -bias: for a tall yellow the vehicle holds a touch high and
    descends onto its upper face (the tall-yellow fix); for a low red it aims level on the centre."""
    return _clip(KP_HEAVE * (el + _aim_bias(colour)), -SPEED_CAP, SPEED_CAP)


def _el_surge_scale(el, colour=""):
    """Forward-surge multiplier in [EL_MIN_SURGE, 1] against the AIM-point elevation error (el plus the
    colour's aim bias): full (1.0) within EL_FULL_SURGE of the aim depth, tapering to the near-zero
    floor by EL_ZERO_SURGE. So the vehicle CLIMBS/DESCENDS to the balloon's height first (esp. the tall
    yellows, whose aim sits a hair above centre) and only then closes the last horizontal gap level,
    meeting the front face head-on instead of passing under it while still changing depth."""
    err = abs(el + _aim_bias(colour))
    return _clip((EL_ZERO_SURGE - err) / (EL_ZERO_SURGE - EL_FULL_SURGE), EL_MIN_SURGE, 1.0)
