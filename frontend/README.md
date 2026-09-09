# Frontend

Not scaffolded here — initialize it directly so you get an up-to-date,
non-stale Vite setup:

```bash
npm create vite@latest . -- --template react-ts
npm install
npm install -D tailwindcss postcss autoprefixer
npx tailwindcss init -p
npm install recharts
```

## Panels to build (see ROADMAP.md core scope)

- **Video panel** — stream/replay with bounding boxes, track IDs, counting
  lines, ROIs overlaid (draw on a `<canvas>` over the `<video>`, don't burn
  boxes into the video server-side)
- **Four-way overview** — current signal state per approach
- **Road metric cards** — vehicles / queue / waiting time / density / signal,
  one per road (see backend's `GET /api/traffic`)
- **Signal state panel** — current phase, remaining green time, controller mode
- **Emergency panel** — only visible when `GET /api/emergency` is non-null;
  should visually read as clearly distinct from normal adaptive mode

Connect to the backend's `/ws/live` WebSocket for live updates rather than
polling the REST endpoints in a loop.

## Live YOLO input contract

The trained checkpoint is loaded lazily from `models/best.pt` using
`configs/model.yaml`. The frontend can send one encoded JPEG/PNG/WebP frame as
a binary WebSocket message to `/ws/live`; the response is the normal dashboard
snapshot plus:

- `detections`: class, confidence, and `bbox` (`x1`, `y1`, `x2`, `y2`)
- `vehicles`: ByteTrack IDs, tracked bounding boxes, centers, road assignment,
  and emergency flags
- `emergency`: the verified approaching emergency vehicle, when present
- `inference_ms` and `frame_size`

Text WebSocket messages still request the latest snapshot without running a
new inference. Call `POST /api/inference/reset` when switching to a new video
or camera source. For a one-shot frame, send the raw image bytes to
`POST /api/inference/image`; `GET /api/inference/status` reports model and
processing health.
