# SLD Multi-Model Inference Viewer

Interactive web app for running D-FINE and RF-DETR object-detection models on SLD (single-line diagram) images with tiled inference, overlapping bounding-box visualization, and live class/threshold editing.

## Architecture

```
Browser  ──HTTP──▶  Nginx (:80)
                      │
         /api/* ──▶  FastAPI + Uvicorn (:8000)  ──▶  PyTorch (GPU)
         /*     ──▶  static files (Vite build)
```

## Quick start (GPU box — Ubuntu)

### 1. Clone & set up backend

```bash
# Clone this repo to your GPU instance
cd /opt
git clone <your-repo-url> sld-inference
cd sld-inference

# Clone D-FINE (needed for the model architecture)
git clone --depth 1 https://github.com/Peterande/D-FINE.git /opt/D-FINE

# Set up Python virtualenv
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Drop your checkpoint files

```
backend/weights/dfine/best_stg1.pth
backend/weights/rfdetr/checkpoint_best_regular.pth
```

The `manifest.json` files next to them are already pre-configured.

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env: set DFINE_REPO_PATH=/opt/D-FINE, WEIGHTS_DIR=weights
```

### 4. Start the API server

```bash
cd /opt/sld-inference/backend
bash start.sh
# or directly:
# uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Test: `curl http://localhost:8000/health` → `{"status":"ok"}`

### 5. Build the frontend

```bash
cd /opt/sld-inference/frontend
npm install
npm run build      # outputs to frontend/dist/
```

### 6. Set up Nginx

```bash
sudo apt install nginx -y
sudo cp nginx/sld-inference.conf /etc/nginx/sites-available/sld-inference
sudo ln -s /etc/nginx/sites-available/sld-inference /etc/nginx/sites-enabled/
sudo rm -f /etc/nginx/sites-enabled/default
sudo nginx -t && sudo systemctl reload nginx
```

### 7. Open in browser

Navigate to `http://<your-gpu-instance-ip>/` — you should see the viewer.

> **Alibaba Cloud Security Group**: Make sure port **80** is open for inbound TCP traffic.

---

## Local development

Run backend and frontend in two terminals:

```bash
# Terminal 1 — backend
cd backend
source .venv/bin/activate
uvicorn app.main:app --reload --port 8000

# Terminal 2 — frontend (Vite dev server with /api proxy)
cd frontend
npm install
npm run dev
```

Vite auto-proxies `/api/*` to `http://localhost:8000` via `vite.config.ts`.

---

## Adding a new checkpoint

1. Create a subfolder under `backend/weights/`, e.g. `weights/dfine_v2/`
2. Drop the `.pth` file inside
3. Create `manifest.json` with the required fields:

```json
{
  "arch": "dfine",
  "model_id": "dfine_v2",
  "display_name": "D-FINE SLD v2",
  "weights_file": "my_checkpoint.pth",
  "num_classes": 30,
  "resolution": 640,
  "class_names": ["..."],
  "confidence_default": 0.20,
  "grid_size": 4,
  "overlap": 0.20,
  "iou_threshold": 0.50
}
```

4. Restart the API server (or hit `GET /api/models` — it rescans on each call)

---

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/api/models` | List all discovered models + config |
| `POST` | `/api/models/{id}/load` | Pre-load a model onto GPU |
| `PUT` | `/api/models/{id}/config` | Update class names / threshold |
| `POST` | `/api/infer` | Run inference (multipart: image + JSON body) |

---

## Key technical notes

- **num_classes is immutable** — it's baked into the checkpoint's output layer shape. The UI shows it as read-only. Only class *names* and threshold are live-editable.
- **Score floor** — the API returns all detections with score ≥ 0.05 (configurable via `MIN_SCORE_FLOOR` in `.env`). The threshold slider in the UI filters client-side for instant response.
- **Tiling** — large SLD images are split into an edge-anchored N×N grid with configurable overlap, inference runs on each tile, then results are merged with per-class NMS. Adaptive D-FINE tiling must use the post-scale median symbol size from the training pipeline; the included 30-class adaptive manifest uses `estimated_symbol_px=210` for approximately 14,044-pixel-wide sheets.
- **Training metadata** — keep the processed COCO JSON category order next to the checkpoint and copy it into `class_names`; do not use a guessed or background-prefixed list. When `STOP_EPOCH` is used, upload the stage checkpoint that actually achieved the validation metric (`best_stg2.pth` after the stage transition when applicable).
- **Sequential inference** — models run one at a time to avoid GPU OOM. For multi-GPU setups, enable concurrent execution in the infer router.
