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
