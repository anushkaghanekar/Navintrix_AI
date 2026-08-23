"""Adaptive controller: picks the highest-priority safe road, normal (non-
emergency) traffic only. Emergency overrides live in emergency_controller.py
and always go through the same state_machine.SafetyStateMachine.
"""


def priority_score(density: float, queue_length: float, waiting_time: float,
                    flow: float, coeffs: dict) -> float:
    """TODO: coeffs['alpha']*density + coeffs['beta']*queue_length
    + coeffs['gamma']*waiting_time + coeffs['delta']*flow
    (coefficients come from configs/signal.yaml's `controller:` block).
    """
    raise NotImplementedError


class AdaptiveController:
    def __init__(self, state_machine, controller_config: dict, fairness):
        self.state_machine = state_machine
        self.controller_config = controller_config
        self.fairness = fairness   # controller/fairness.py's anti-starvation tracker

    def choose_next_road(self, road_metrics: dict) -> str:
        """road_metrics: {road: {density, queue_length, waiting_time, flow}}.

        TODO: compute priority_score per road, apply the fairness override
        if any road has been waiting past max_wait_before_forced_green_seconds,
        then request the winning road via self.state_machine.request_phase_change.
        """
        raise NotImplementedError
