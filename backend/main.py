"""FastAPI backend entrypoint.

This wiring (routes, WebSocket connection handling) is real and runnable
as-is against placeholder data — useful for the frontend team to build
against immediately. Swap the placeholder return values for calls into
controller/ and evaluation/ once those are implemented.

Run: uvicorn backend.main:app --reload
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

app = FastAPI(title="Adaptive Traffic Signal Control API")

# TODO: replace with the actual live controller/analytics state once
# controller/ and analytics/ are implemented.
_PLACEHOLDER_STATE = {
    "roads": {
        "north": {"vehicles": 0, "queue": 0, "waiting_seconds": 0, "signal": "RED"},
        "south": {"vehicles": 0, "queue": 0, "waiting_seconds": 0, "signal": "GREEN"},
        "east": {"vehicles": 0, "queue": 0, "waiting_seconds": 0, "signal": "RED"},
        "west": {"vehicles": 0, "queue": 0, "waiting_seconds": 0, "signal": "RED"},
    },
    "controller_mode": "ADAPTIVE",
    "emergency": None,
}


@app.get("/api/intersection")
def get_intersection():
    """TODO: return static intersection/config metadata for the dashboard to render."""
    raise NotImplementedError


@app.get("/api/traffic")
def get_traffic():
    return _PLACEHOLDER_STATE["roads"]


@app.get("/api/vehicles")
def get_vehicles():
    """TODO: return currently tracked vehicles (from tracking/bytetrack.py state)."""
    raise NotImplementedError


@app.get("/api/signals")
def get_signals():
    """TODO: return current phase/road/remaining time from
    controller/state_machine.SafetyStateMachine."""
    raise NotImplementedError


@app.get("/api/emergency")
def get_emergency():
    return _PLACEHOLDER_STATE["emergency"]


@app.get("/api/metrics")
def get_metrics():
    """TODO: return live density/queue/waiting-time/flow per road."""
    raise NotImplementedError


@app.get("/api/controller/status")
def controller_status():
    return {"mode": _PLACEHOLDER_STATE["controller_mode"]}


@app.post("/api/controller/start")
def controller_start():
    """TODO: start the control loop (see simulation/traci_controller.py
    or a live-camera equivalent) in a background task."""
    raise NotImplementedError


@app.post("/api/controller/stop")
def controller_stop():
    raise NotImplementedError


@app.post("/api/controller/mode")
def controller_mode(mode: str):
    """TODO: switch between e.g. ADAPTIVE / FIXED_TIME / DENSITY_ONLY for
    live demo comparisons."""
    raise NotImplementedError


@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """Streams live vehicle counts, signal changes, emergency detections,
    and metric updates. Wiring is real; the payload is placeholder.
    """
    await websocket.accept()
    try:
        while True:
            # TODO: replace with real state pushed from the control loop
            # (e.g. via an asyncio.Queue fed by simulation/traci_controller.py)
            await websocket.receive_text()
            await websocket.send_json(_PLACEHOLDER_STATE)
    except WebSocketDisconnect:
        pass
