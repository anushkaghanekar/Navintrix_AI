"""Bridges controller/ (SafetyStateMachine + AdaptiveController +
EmergencyController) to a running SUMO simulation via TraCI.

This is the piece that turns "we have a controller" into "we have a
controller that actually drives traffic lights in a simulated
intersection" — budget real time for it, TraCI's API takes some getting
used to.
"""

import traci


def run_simulation(sumocfg_path: str, state_machine, adaptive_controller,
                    emergency_controller, max_steps: int) -> dict:
    """TODO:
    1. traci.start(["sumo", "-c", sumocfg_path])
    2. loop traci.simulationStep() up to max_steps
    3. each step: pull vehicle positions/classes from TraCI, feed them
       through the same detection-shaped pipeline (or a simulation-native
       equivalent) to get per-road metrics
    4. call adaptive_controller / emergency_controller as appropriate
    5. push the resulting phase to SUMO's traffic light via
       traci.trafficlight.setRedYellowGreenState(...)
    6. collect and return the metrics evaluation/metrics.py needs
       (waiting time, queue length, throughput, etc.)
    7. traci.close()
    """
    raise NotImplementedError
