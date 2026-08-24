"""Adaptive controller: picks the highest-priority safe road, normal (non-
emergency) traffic only. Emergency overrides live in emergency_controller.py
and always go through the same state_machine.SafetyStateMachine.

Critically (see ROADMAP), this module never sets a signal phase directly. It
computes which road *should* get green and asks the safety state machine via
request_phase_change; the machine decides whether the timing is safe and
otherwise ignores the request.
"""

from __future__ import annotations

DEFAULT_ROADS = ("north", "east", "south", "west")


def priority_score(
    density: float,
    queue_length: float,
    waiting_time: float,
    flow: float,
    coeffs: dict,
) -> float:
    """Weighted priority score for one road's current state.

    score = alpha*density + beta*queue_length + gamma*waiting_time
            + delta*traffic_flow
    Coefficients come from configs/signal.yaml's ``controller`` block.
    Missing coefficients default to 1.0 if absent (but configs should
    always provide them).
    """
    return (
        float(coeffs.get("alpha", 1.0)) * float(density)
        + float(coeffs.get("beta", 1.0)) * float(queue_length)
        + float(coeffs.get("gamma", 1.0)) * float(waiting_time)
        + float(coeffs.get("delta", 1.0)) * float(flow)
    )


class AdaptiveController:
    def __init__(self, state_machine, controller_config: dict, fairness):
        self.state_machine = state_machine
        self.controller_config = controller_config
        self.fairness = fairness   # controller/fairness.py's anti-starvation tracker
        self._now = 0.0

    def set_time(self, timestamp: float) -> None:
        """Set the controller's current simulation/realtime clock.

        The controller uses this value both to ask fairness how long roads
        have waited and to timestamp its phase-change requests. Without it,
        decisions would be evaluated against t=0 (nonsense in a live loop).
        """
        self._now = float(timestamp)

    def choose_next_road(self, road_metrics: dict) -> str:
        """Decide which road to request green for (normal traffic).

        ``road_metrics``: {road: {density, queue_length, waiting_time, flow}}.

        Courses (first match wins):
          * A road that has waited past the fairness ceiling is FORCED
            (worst offender) regardless of its raw score.
          * Otherwise the road with the highest priority_score is chosen.
          * With no metrics at all, nothing is requested and the current
            green road is returned (a no-op).

        The chosen road is requested via self.state_machine.request_phase_change
        (never set directly); if the machine refuses (e.g. min-green not yet
        met), we return the intended road — the machine simply ignores it.
        """
        coeffs = self.controller_config.get("controller", self.controller_config)

        # No metrics -> no decision.
        if not road_metrics:
            return self.state_machine.current_green_road()

        now = self._now
        forced = self.fairness.road_forcing_green(now)
        if forced is not None and forced in road_metrics:
            self.state_machine.request_phase_change(forced, now)
            return forced

        best_road = None
        best_score = float("-inf")
        for road, metrics in road_metrics.items():
            metrics = metrics or {}
            score = priority_score(
                metrics.get("density", 0.0),
                metrics.get("queue_length", 0.0),
                metrics.get("waiting_time", 0.0),
                metrics.get("flow", 0.0),
                coeffs,
            )
            if score > best_score:
                best_score = score
                best_road = road

        if best_road is not None:
            self.state_machine.request_phase_change(best_road, now)
            return best_road

        return self.state_machine.current_green_road()
