"""Keyboard -> twist command for the window launcher (carb input events of the Isaac app window).

Held keys: W / Up = forward, S / Down = backward, A / Left = turn left, D / Right = turn right, Shift = x2 speed.
One-shot keys are collected in ``pending_actions``: X = stop (Space is Kit's play/pause hotkey), R = reset to spawn (clears tracks),
T = clear tracks only, C = cycle viewport camera, Escape = quit.
"""
from __future__ import annotations


class KeyboardTwistDevice:
    """Tracks held keys and turns them into [forward_speed_m_s, yaw_rate_rad_s]."""

    def __init__(self, forward_speed_m_s: float, yaw_rate_rad_s: float):
        """Subscribe to the default app window's keyboard."""
        import carb.input
        import omni.appwindow
        self.forward_speed_m_s = forward_speed_m_s
        self.yaw_rate_rad_s = yaw_rate_rad_s
        self.held: set[str] = set()
        self.pending_actions: list[str] = []
        self._carb_input = carb.input
        self._input_interface = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._subscription = self._input_interface.subscribe_to_keyboard_events(self._keyboard, self._on_event)
        keys = carb.input.KeyboardInput
        self._held_keys = {keys.W: "forward", keys.UP: "forward", keys.S: "backward", keys.DOWN: "backward",
                           keys.A: "left", keys.LEFT: "left", keys.D: "right", keys.RIGHT: "right",
                           keys.LEFT_SHIFT: "fast", keys.RIGHT_SHIFT: "fast"}
        self._action_keys = {keys.X: "stop", keys.R: "reset", keys.T: "clear_tracks", keys.C: "camera",
                             keys.ESCAPE: "quit"}

    def _on_event(self, event, *arguments) -> bool:
        """carb keyboard callback: update held keys and queue one-shot actions."""
        event_type = self._carb_input.KeyboardEventType
        if event.type == event_type.KEY_PRESS:
            if event.input in self._held_keys:
                self.held.add(self._held_keys[event.input])
            elif event.input in self._action_keys:
                self.pending_actions.append(self._action_keys[event.input])
                if self._action_keys[event.input] == "stop":
                    self.held.clear()
        elif event.type == event_type.KEY_RELEASE and event.input in self._held_keys:
            self.held.discard(self._held_keys[event.input])
        return True

    def twist(self) -> tuple[float, float]:
        """Current [forward_speed_m_s, yaw_rate_rad_s] from the held keys."""
        scale = 2.0 if "fast" in self.held else 1.0
        forward = (("forward" in self.held) - ("backward" in self.held)) * self.forward_speed_m_s * scale
        yaw = (("left" in self.held) - ("right" in self.held)) * self.yaw_rate_rad_s * scale
        return float(forward), float(yaw)

    def take_actions(self) -> list[str]:
        """Return and clear the queued one-shot actions."""
        actions, self.pending_actions = self.pending_actions, []
        return actions

    def close(self) -> None:
        """Unsubscribe from the keyboard."""
        if self._subscription is not None:
            self._input_interface.unsubscribe_to_keyboard_events(self._keyboard, self._subscription)
            self._subscription = None
