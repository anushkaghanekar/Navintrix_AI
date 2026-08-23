"""Emergency controller: handles the EMERGENCY_DETECTED -> ... ->
ADAPTIVE_CONTROL state sequence from the spec. Overrides the adaptive
controller's choice but still goes through the same SafetyStateMachine —
it never sets phases directly.
"""

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
        self.priority_module = priority_module      # emergency/priority.py
        self.trajectory_module = trajectory_module   # emergency/trajectory.py

    def handle(self, active_emergencies: list, timestamp: float) -> bool:
        """Called every control loop iteration when emergency vehicles are
        present.

        TODO: verify approach (emergency/detector.py), select the priority
        vehicle if there are multiple (emergency/priority.py), determine its
        required movement (emergency/trajectory.py), and request that road
        via self.state_machine.request_phase_change(..., emergency=True).
        Track EMERGENCY_PASSED (via emergency/tracker.py's `cleared` flag)
        to know when to hand control back to the adaptive controller.
        Return True while an emergency is actively being handled.
        """
        raise NotImplementedError
