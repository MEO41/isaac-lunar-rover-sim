"""Physics steppers and synchronous rendering.

ExplicitPhysicsStepper (headless): ``control_period_s * physics_rate_hz`` explicit PhysX steps per control step, no
rendering. ApplicationUpdateStepper (window): one ``app.update()`` per control step = 1/60 s = 2 physics steps at
120 Hz, renders the viewport. Both call every registered listener once per physics step with (step_dt_s, time_s).
"""
from __future__ import annotations

from typing import Callable

PhysicsStepListener = Callable[[float, float], None]


class PhysicsStepper:
    """Common interface of the two steppers."""

    def __init__(self, application, physics_rate_hz: int):
        """Store the app and rate; ``start`` plays the timeline."""
        self.application = application
        self.physics_dt_s = 1.0 / physics_rate_hz
        self.time_s = 0.0
        self.listeners: list[PhysicsStepListener] = []
        self.control_period_s = self.physics_dt_s

    def add_listener(self, listener: PhysicsStepListener) -> None:
        """Call ``listener(dt_s, time_s)`` after every physics step."""
        self.listeners.append(listener)

    def start(self) -> None:
        """Play the timeline and run one application update (tensor views need it)."""
        import omni.timeline
        omni.timeline.get_timeline_interface().play()
        self.application.update()

    def _notify(self, steps: int) -> None:
        """Advance the clock by ``steps`` physics steps and call the listeners for each."""
        for _ in range(steps):
            self.time_s += self.physics_dt_s
            for listener in self.listeners:
                listener(self.physics_dt_s, self.time_s)

    def advance_control_period(self) -> None:
        """Advance physics by one control period."""
        raise NotImplementedError


class ExplicitPhysicsStepper(PhysicsStepper):
    """Headless: explicit simulate/fetch_results, ``steps_per_control`` steps per control period, no render."""

    def __init__(self, application, physics_rate_hz: int, control_rate_hz: int):
        """Steps per control period = physics_rate_hz / control_rate_hz (must divide)."""
        super().__init__(application, physics_rate_hz)
        if physics_rate_hz % control_rate_hz:
            raise ValueError(f"physics rate {physics_rate_hz} Hz is not a multiple of control rate {control_rate_hz} Hz")
        self.steps_per_control = physics_rate_hz // control_rate_hz
        self.control_period_s = 1.0 / control_rate_hz
        self._physx = None

    def advance_physics_steps(self, steps: int) -> None:
        """Run ``steps`` explicit PhysX steps (listeners called after each)."""
        if self._physx is None:
            import omni.physx
            self._physx = omni.physx.get_physx_simulation_interface()
        for _ in range(steps):
            self._physx.simulate(self.physics_dt_s, self.time_s)
            self._physx.fetch_results()
            self._notify(1)

    def advance_control_period(self) -> None:
        """One control period of explicit physics."""
        self.advance_physics_steps(self.steps_per_control)


class ApplicationUpdateStepper(PhysicsStepper):
    """Window: one app.update() per control period (1/60 s, two 120 Hz physics steps, renders the viewport)."""

    def __init__(self, application, physics_rate_hz: int):
        """Control period = 1/60 s."""
        super().__init__(application, physics_rate_hz)
        self.control_period_s = 1.0 / 60.0
        self.steps_per_control = max(1, round(self.control_period_s / self.physics_dt_s))

    def advance_physics_steps(self, steps: int) -> None:
        """Advance roughly ``steps`` physics steps with application updates."""
        for _ in range(max(1, round(steps / self.steps_per_control))):
            self.advance_control_period()

    def advance_control_period(self) -> None:
        """One application update."""
        self.application.update()
        self._notify(self.steps_per_control)


_REPLICATOR_INITIALIZED = False


def render_synchronously(subframes: int) -> None:
    """Render all render products now without advancing simulation time; annotator data is then current.

    Measured on this machine: annotators deliver nothing until replicator has stepped once, and the first step only
    succeeds with ``pause_timeline=True``; later steps work with the timeline left playing. The rendered rover pose
    follows explicit PhysX stepping (pixel centroid within a few pixels of its projection). The timeline is resumed
    after the first step so application-update stepping keeps advancing physics."""
    global _REPLICATOR_INITIALIZED
    import omni.replicator.core as replicator
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    was_playing = timeline.is_playing()
    replicator.orchestrator.step(rt_subframes=max(1, int(subframes)), delta_time=0.0,
                                 pause_timeline=not _REPLICATOR_INITIALIZED)
    _REPLICATOR_INITIALIZED = True
    if was_playing and not timeline.is_playing():
        timeline.play()
