#!/usr/bin/env bash
# Deploy the complete SLD Inference Viewer on an Ubuntu NVIDIA GPU VM.
# Run from the checked-out repository: sudo bash setup_and_run.sh
set -Eeuo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
log_info() { echo -e "${CYAN}[INFO]${NC} $*"; }
log_ok() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
trap 'log_error "Setup failed at line $LINENO. Review the command above and rerun after fixing it."' ERR

if [[ ${EUID} -ne 0 ]]; then
    log_error "Run with sudo: sudo bash setup_and_run.sh"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
APP_DIR="${APP_DIR:-$SCRIPT_DIR}"
DFINE_DIR="${DFINE_DIR:-$APP_DIR/D-FINE}"
APP_USER="${APP_USER:-sldinference}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"

for required_dir in backend frontend nginx; do
    [[ -d "$APP_DIR/$required_dir" ]] || {
        log_error "APP_DIR='$APP_DIR' is not the SLD-Inference repository (missing $required_dir/)."
        exit 1
    }
done

log_info "Deploying from $APP_DIR"
log_info "Installing Ubuntu packages..."
export DEBIAN_FRONTEND=noninteractive
apt-get update -y
apt-get install -y python3 python3-venv python3-pip python3-dev build-essential \
    git nginx curl ca-certificates libgl1 libglib2.0-0 net-tools procps

if ! command -v nvidia-smi >/dev/null 2>&1; then
    log_error "No NVIDIA driver is available. Use an NVIDIA GPU VM/image, install its driver, reboot, then rerun."
    exit 1
fi
log_ok "NVIDIA driver detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ' -)"

if ! command -v node >/dev/null 2>&1 || [[ "$(node -v | sed 's/^v//' | cut -d. -f1)" -lt 18 ]]; then
    log_info "Installing Node.js 20 LTS..."
    curl --fail --show-error --location --retry 3 https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
log_ok "Node.js $(node -v), npm $(npm -v)"

if [[ ! -d "$DFINE_DIR" ]]; then
    log_info "D-FINE is not bundled; cloning it to $DFINE_DIR..."
    git clone --depth 1 https://github.com/Peterande/D-FINE.git "$DFINE_DIR"
fi
[[ -f "$DFINE_DIR/requirements.txt" ]] || { log_error "Invalid D-FINE directory: $DFINE_DIR"; exit 1; }

if ! id -u "$APP_USER" >/dev/null 2>&1; then
    useradd --system --create-home --home-dir "/var/lib/$APP_USER" --shell /usr/sbin/nologin "$APP_USER"
fi
for gpu_group in video render; do
    if getent group "$gpu_group" >/dev/null; then
        usermod -aG "$gpu_group" "$APP_USER"
    fi
done
chown -R "$APP_USER:$APP_USER" "$APP_DIR"

log_info "Creating Python environment and installing CUDA-enabled PyTorch..."
cd "$APP_DIR/backend"
if [[ ! -d .venv ]]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip wheel setuptools
# Install PyTorch first from the CUDA wheel repository. The index can be
# overridden for a VM with a different supported CUDA driver, for example:
# TORCH_INDEX_URL=https://download.pytorch.org/whl/cu121 sudo bash setup_and_run.sh
python -m pip install --upgrade torch torchvision --index-url "$TORCH_INDEX_URL"
python -m pip install -r requirements.txt
python -m pip install -r "$DFINE_DIR/requirements.txt"
python - <<'PY'
import torch
assert torch.cuda.is_available(), (
    "PyTorch cannot access CUDA. Check the NVIDIA driver and set TORCH_INDEX_URL "
    "to a wheel compatible with that driver before rerunning setup."
)
print(f"PyTorch {torch.__version__} using CUDA {torch.version.cuda} on {torch.cuda.get_device_name(0)}")
PY
log_ok "Python dependencies and CUDA verified."

ENV_FILE="$APP_DIR/backend/.env"
touch "$ENV_FILE"
set_env() {
    local key="$1" value="$2"
    if grep -qE "^${key}=" "$ENV_FILE"; then
        sed -i "s|^${key}=.*|${key}=${value}|" "$ENV_FILE"
    else
        printf '%s=%s\n' "$key" "$value" >> "$ENV_FILE"
    fi
}
set_env WEIGHTS_DIR "$APP_DIR/backend/weights"
set_env DFINE_REPO_PATH "$DFINE_DIR"
set_env MIN_SCORE_FLOOR "0.05"
set_env CORS_ORIGINS '["*"]'
chown "$APP_USER:$APP_USER" "$ENV_FILE"

for manifest in "$APP_DIR"/backend/weights/*/manifest.json; do
    [[ -f "$manifest" ]] || continue
    weight_file="$(python -c "import json; print(json.load(open('$manifest'))['weights_file'])")"
    weight_path="$(dirname "$manifest")/$weight_file"
    if [[ -s "$weight_path" ]]; then
        log_ok "Found checkpoint: $weight_path"
    else
        log_warn "Checkpoint missing: $weight_path"
    fi
done

log_info "Building the React frontend..."
cd "$APP_DIR/frontend"
npm ci --no-audit --no-fund
npm run build
chown -R "$APP_USER:$APP_USER" "$APP_DIR/frontend/dist"

log_info "Installing Nginx and systemd configuration..."
sed "s|__APP_DIR__|$APP_DIR|g" "$APP_DIR/nginx/sld-inference.conf" \
    > /etc/nginx/sites-available/sld-inference
ln -sfn /etc/nginx/sites-available/sld-inference /etc/nginx/sites-enabled/sld-inference
rm -f /etc/nginx/sites-enabled/default
nginx -t

cat > /etc/systemd/system/sld-inference.service <<EOF
[Unit]
Description=SLD GPU Inference API
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_DIR/backend
EnvironmentFile=-$APP_DIR/backend/.env
Environment=PYTHONUNBUFFERED=1
Environment=PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ExecStart=$APP_DIR/backend/.venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8000 --workers 1 --timeout-keep-alive 300 --log-level info
Restart=always
RestartSec=5
TimeoutStartSec=0
TimeoutStopSec=90
LimitNOFILE=65536

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now sld-inference
systemctl enable --now nginx

log_info "Waiting for API health check..."
for attempt in {1..30}; do
    if curl --fail --silent --show-error --max-time 5 http://127.0.0.1:8000/health | grep -q '"status":"ok"'; then
        log_ok "API is healthy."
        break
    fi
    if [[ "$attempt" -eq 30 ]]; then
        systemctl --no-pager --full status sld-inference || true
        journalctl -u sld-inference -n 100 --no-pager || true
        log_error "API did not become healthy."
        exit 1
    fi
    sleep 2
done

systemctl reload nginx
PUBLIC_IP="$(curl --fail --silent --show-error --connect-timeout 3 --max-time 5 https://ifconfig.me 2>/dev/null || true)"
PUBLIC_IP="${PUBLIC_IP:-YOUR-SERVER-IP}"
log_ok "Deployment complete. Open: http://$PUBLIC_IP/"
echo "Logs: journalctl -u sld-inference -f"
echo "The proxy allows one hour for a single tiled inference request; the browser client has no shorter timeout."
