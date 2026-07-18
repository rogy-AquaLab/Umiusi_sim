"""Rendering: the onboard-camera DEGRADATION forward model (what the underwater camera sees).

Physically-based (Jaffe-McGlamery) underwater image formation — depth-based colour attenuation,
backscatter haze, turbidity, surface reflection — applied to a clean rendered frame by
``UmiusiSimulator.render_camera(degrade=True)``. This is a SIM concern (it models the sensor), not a
perception-inference one, so it lives in the sim package; ``umiusi_perception`` consumes the frames
it produces but does not own the model.
"""
