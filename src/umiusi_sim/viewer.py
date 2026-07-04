"""Shared live-viewing module — one consistent MuJoCo GUI experience everywhere.

Every live viewer in the project (``tools/drive.py``, ``UmiusiPoseEnv.render()`` for
``eval --render``, ``tools/competition_run.py --render``) goes through this module, so the camera
behaviour, the +Y-up handling and the on-screen controls are identical no matter which tool you launch.

Why a shared module: this model is +Y-up (CAD frame), but MuJoCo's FREE camera assumes
+Z-up, so the free camera looks tilted and is easy to get lost in. The fixed MJCF cameras
(``iso`` / ``top`` / ``track``) are framed correctly for +Y-up; ``track`` additionally
follows the vehicle CoM so it never drifts out of frame. The default here is therefore the
``track`` camera — the vehicle is always framed and followed.

Typical use (real-time loop driven by us)::

    from umiusi_sim.viewer import UmiusiViewer
    with UmiusiViewer(sim.model, sim.data, base_id=sim.base_id,
                      control_rate_hz=sim.cfg["sim"]["control_rate_hz"]) as v:
        v.run(lambda: sim.step(action))          # paces to real time, syncs each step

Use when an outer loop already drives stepping (e.g. SB3 eval)::

    v = UmiusiViewer(model, data, base_id=base_id).launch()
    ...
    v.sync()                                     # call once per rendered frame

The GUI itself cannot be exercised headless (no display); this module keeps all the
camera/pacing logic in one place so it is verified by review + the user running it, while
callers stay tiny.
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer

# Fixed MJCF cameras framed for +Y-up. "track" follows the vehicle CoM (default).
FIXED_CAMS = ("track", "iso", "top")
DEFAULT_CAM = "track"


class UmiusiViewer:
    """Thin, consistent wrapper around ``mujoco.viewer.launch_passive``.

    Parameters
    ----------
    model, data : the MuJoCo model/data to view.
    base_id : body id used to frame the free camera on the whole-vehicle CoM
        (``data.subtree_com[base_id]``). Defaults to the ``base_link`` body.
    cam : initial camera — a fixed-camera name (``"track"`` default, or ``"iso"``/``"top"``/
        any MJCF camera) or ``"free"`` for the mouse-controllable free camera.
    control_rate_hz : real-time pacing rate for :meth:`run` (defaults to 50 Hz).
    key_callback : optional ``fn(keycode)`` for tool-specific keys (e.g. drive.py steering).
    extra_keys : optional ``{"key": "what it does"}`` mapping, shown in the console legend.
    quiet : suppress the console legend (used by env.render to keep its output clean).
    """

    def __init__(self, model, data, *, base_id=None, cam=DEFAULT_CAM,
                 control_rate_hz=50.0, key_callback=None, extra_keys=None, quiet=False):
        self.model = model
        self.data = data
        self.base_id = base_id if base_id is not None else model.body("base_link").id
        self.cam = cam
        self.control_rate_hz = float(control_rate_hz)
        self._key_callback = key_callback
        self.extra_keys = dict(extra_keys or {})
        self.quiet = quiet
        self.viewer = None

    # -- setup -----------------------------------------------------------------
    def launch(self):
        """Open the passive viewer, apply the initial camera, print the legend. Returns self."""
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )
        self.set_camera(self.cam)
        if not self.quiet:
            self._print_legend()
        return self

    def set_camera(self, cam):
        """Point the viewer at a fixed MJCF camera by name, or ``"free"`` (framed on the CoM)."""
        self.cam = cam
        c = self.viewer.cam
        if cam == "free":
            # Free camera: mouse-controllable but +Z-up, so frame it on the whole-vehicle CoM
            # with a sane azimuth/elevation. It will look tilted (the model is +Y-up).
            c.type = mujoco.mjtCamera.mjCAMERA_FREE
            c.lookat[:] = self.data.subtree_com[self.base_id]
            c.distance = 1.5
            c.azimuth, c.elevation = 135.0, -20.0
        else:
            c.type = mujoco.mjtCamera.mjCAMERA_FIXED
            c.fixedcamid = self.model.camera(cam).id

    def _print_legend(self):
        """One-time console legend, identical across every tool."""
        lines = ["viewer controls:"]
        if self.cam == "free":
            lines.append("  NOTE: model is +Y-up; the FREE camera assumes +Z-up so it looks "
                         "tilted — the fixed 'track'/'iso'/'top' cameras are upright.")
            lines.append("  mouse: left-drag orbit, right-drag pan, scroll zoom (free camera)")
        else:
            lines.append(f"  camera: fixed '{self.cam}' (upright, +Y-up framed"
                         + ("; follows the vehicle)" if self.cam == "track" else ")"))
            lines.append("  mouse: drives the FREE camera (press Tab/[ / ] to switch to it first)")
        lines.append("  [  /  ] : cycle cameras (free <-> fixed MJCF cameras)")
        for key, desc in self.extra_keys.items():
            lines.append(f"  {key} : {desc}")
        print("\n".join(lines), flush=True)

    # -- loop ------------------------------------------------------------------
    def run(self, step_fn):
        """Real-time loop: each control step call ``step_fn()``, ``sync()``, then sleep to pace.

        ``step_fn`` advances one control step (e.g. ``lambda: sim.step(action)``). Blocks until
        the window is closed. The viewer is auto-launched if not already open.
        """
        if self.viewer is None:
            self.launch()
        control_dt = 1.0 / self.control_rate_hz
        while self.viewer.is_running():
            tic = time.time()
            step_fn()
            self.viewer.sync()
            time.sleep(max(0.0, control_dt - (time.time() - tic)))

    # -- passthrough -----------------------------------------------------------
    def sync(self):
        self.viewer.sync()

    def is_running(self):
        return self.viewer is not None and self.viewer.is_running()

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def __enter__(self):
        return self.launch()

    def __exit__(self, *exc):
        self.close()
        return False
