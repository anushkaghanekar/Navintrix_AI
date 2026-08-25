"""Emergency controller: handles the EMERGENCY_DETECTED -> ... ->
ADAPTIVE_CONTROL state sequence from the spec. Overrides the adaptive
controller's choice but still goes through the same SafetyStateMachine —
it never sets phases directly.

This is the coordinator that turns an approaching emergency vehicle into a
priority request the safety state machine will accept. It is not itself a
signal authority: every action it takes is a call to
``state_machine.request_phase_change(..., emergency=True)``, so the FSM's
green->yellow->all-red->green sequence is always respected (emergency
only bypasses the min-green floor, never the safety transition).
"""

from __future__ import annotations

from enum import Enum, auto


class EmergencyMode(Enum):
    VERIFY_APPROACH = auto()
    SELECT_REQUIRED_MOVEMENT = auto()
    SAFE_TRANSITION = auto()
    EMERGENCY_GREEN = auto()
    EMERGENCY_PASSED = auto()


class EmergencyController:
    def __init__(self, state_machine, priority_module, trajectory_module):
        self.state_machine = state_machine
        self.priority_module = priority_module           # emergency/priority.py
        self.trajectory_module = trajectory_module       # emergency/trajectory.py
        self.mode = EmergencyMode.EMERGENCY_PASSED
        self.active_vehicle_id: int | None = None
        self.movement_history: list[str] = []

    def handle(self, active_emergencies: list, timestamp: float) -> bool:
        """Called every control loop iteration when emergency vehicles are
        present.

        Select the priority vehicle (emergency/priority.py), verify it is on
        an active approach, determine its required movement
        (emergency/trajectory.py), and request that road through the state
        machine with ``emergency=True``. Returns True while an emergency is
        being handled (so the loop keeps the adaptive controller paused).
        """
        selected = self.priority_module.select_priority_emergency(
            active_emergencies, self.state_machine.phase
        )
        if selected is None:
            self.mode = EmergencyMode.EMERGENCY_PASSED
            return False

        # Served vehicle changed -> reset per-vehicle trajectory history.
        if self.active_vehicle_id != selected.track_id:
            self.active_vehicle_id = selected.track_id
            self.movement_history = []

        # Record road history + request green only while genuinely approaching.
        if selected.approaching_intersection:
            if not self.movement_history or self.movement_history[-1] != selected.road:
                self.movement_history.append(selected.road)
            self.movement = self.trajectory_module.required_movement(
                selected, self.movement_history
            )
            self.mode = EmergencyMode.EMERGENCY_GREEN
        else:
            self.movement = None
            self.mode = EmergencyMode.VERIFY_APPROACH

        # Emergency priority always asks through the safety FSM: emergency=True
        # skips only min-green; the transition (yellow/all-red) is preserved.
        self.state_machine.request_phase_change(
            selected.road, timestamp, emergency=True
        )
        return True
