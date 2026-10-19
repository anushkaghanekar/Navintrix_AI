# Signal Sense

A local Streamlit demo for turning four intersection camera images into a traffic-light recommendation with YOLO26m.

## Run the React app

```bash
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

In a second terminal:

```bash
npm install
npm run dev
```

Open http://localhost:5173, upload one image for each approach, and select **Analyze junction**. The React dashboard calls the local FastAPI service, which runs YOLO26m and returns the annotated evidence plus the adaptive signal plan.

The original Streamlit prototype remains available with `streamlit run app.py`.

## Model note

The included `yolo26m.pt` checkpoint is a COCO-trained detector. It recognizes common classes such as `car`, `motorcycle`, `bus`, and `truck`, but COCO does not contain ambulance, police, or fire-truck classes. The app reports emergency classes if a custom checkpoint supplies them and includes a per-approach emergency override for demonstrating priority handling.

The signal planner uses a transparent point model for the demo:

- Bicycle: 1 point
- Motorcycle: 2 points
- Car: 3 points
- Truck: 5 points
- Bus: 6 points
- Queue: 1 point per 5 meters
- Emergency vehicle: 30 points

The available green time is distributed proportionally across the approaches inside a 120-second cycle. Each approach receives at least 15 seconds, with 6 seconds reserved per approach for yellow/all-red changeover. Replace these calibration values with traffic-engineering data before field use.

Queue length is estimated from the still image by counting detected vehicles whose bounding-box centers fall in the lower 42% of the frame, treating that area as the near-road queue zone. Vehicle class lengths plus a 2-meter spacing allowance are summed into an approximate queue length. This is useful for a demo, but camera calibration or video-based speed/stop analysis is needed for field accuracy.

The live signal simulation replays the computed cycle with automatic green, yellow, and all-red phases. It demonstrates the controller using uploaded images; it does not require a camera or live feed.