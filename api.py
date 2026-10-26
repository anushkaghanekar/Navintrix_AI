from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from app import DIRECTIONS, adaptive_green_times, analyze_direction, signal_phase_order


app = FastAPI(title="Signal Sense API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        origin.strip()
        for origin in os.getenv(
            "CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image(image) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def serialize_result(result) -> dict:
    return {
        "direction": result.direction,
        "counts": result.counts,
        "vehicleCount": result.vehicle_count,
        "queueVehicleCount": result.queue_vehicle_count,
        "queueMeters": result.queue_meters,
        "emergencyCount": result.emergency_count,
        "vehiclePoints": result.vehicle_points,
        "queuePoints": result.queue_points,
        "emergencyPoints": result.emergency_points,
        "demandPoints": result.demand_points,
        "annotatedImage": f"data:image/png;base64,{encode_image(result.annotated)}",
    }


async def read_upload(upload: UploadFile) -> Image.Image:
    return Image.open(BytesIO(await upload.read())).convert("RGB")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": os.getenv("MODEL_PATH", "yolo26n.pt")}


@app.post("/api/analyze")
async def analyze(
    north: UploadFile = File(...),
    east: UploadFile = File(...),
    south: UploadFile = File(...),
    west: UploadFile = File(...),
    confidence: float = Form(0.35),
    emergency_north: int = Form(0),
    emergency_east: int = Form(0),
    emergency_south: int = Form(0),
    emergency_west: int = Form(0),
) -> dict:
    uploads = {"North": north, "East": east, "South": south, "West": west}
    emergency_overrides = {
        "North": emergency_north,
        "East": emergency_east,
        "South": emergency_south,
        "West": emergency_west,
    }
    results = []
    for direction in DIRECTIONS:
        image = await read_upload(uploads[direction])
        results.append(analyze_direction(direction, image, emergency_overrides[direction], confidence))

    green_times = adaptive_green_times(results)
    phase_order = signal_phase_order(results)
    serialized = [serialize_result(result) | {"greenSeconds": green_times[result.direction]} for result in results]
    total_points = sum(result.demand_points for result in results)
    priority = max(results, key=lambda result: (result.emergency_count, result.demand_points))
    return {
        "results": serialized,
        "phaseOrder": phase_order,
        "cycleSeconds": 120,
        "changeoverSeconds": 6,
        "priorityDirection": priority.direction,
        "totalVehicles": sum(result.vehicle_count for result in results),
        "totalPoints": total_points,
        "totalEmergency": sum(result.emergency_count for result in results),
    }
