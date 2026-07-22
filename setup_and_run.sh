#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# SLD Multi-Model Inference Viewer — One-Click Automated Setup & Deployment Script
# ─────────────────────────────────────────────────────────────────────────────
# Run this script on your Alibaba Cloud Ubuntu GPU instance:
#   cd /opt/SLD-Inference
#   bash setup_and_run.sh
# ─────────────────────────────────────────────────────────────────────────────
set -e

# Color helpers
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warn()    { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

echo -e "${BLUE}"
echo "=========================================================================="
echo "          SLD MULTI-MODEL INFERENCE VIEWER - AUTOMATED SETUP             "
echo "=========================================================================="
echo -e "${NC}"

# 1. Root permission check
if [ "$EUID" -ne 0 ]; then
  log_error "Please run as root (or with sudo)."
  exit 1
fi

APP_DIR="/opt/SLD-Inference"
DFINE_DIR="/opt/D-FINE"

# Ensure we are in the application root
if [ ! -d "$APP_DIR" ]; then
    log_info "Creating application directory at $APP_DIR ..."
    mkdir -p "$APP_DIR"
fi
cd "$APP_DIR"

# 2. System dependencies
log_info "1/7 Updating system packages and installing required OS tools..."
apt-get update -y
apt-get install -y python3-venv python3-pip git nginx curl net-tools procps

# 3. Node.js & npm installation check
log_info "2/7 Checking Node.js & npm..."
if ! command -v node &> /dev/null || [ $(node -v | cut -d'.' -f1 | tr -d 'v') -lt 18 ]; then
    log_info "Node.js 18+ not detected. Installing Node.js 20 LTS from NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
    apt-get install -y nodejs
fi
log_success "Node.js $(node -v) & npm $(npm -v) ready."

# 4. Clone D-FINE repository if missing
log_info "3/7 Checking D-FINE architecture repository..."
if [ ! -d "$DFINE_DIR" ]; then
    log_info "Cloning D-FINE repository to $DFINE_DIR ..."
    git clone --depth 1 https://github.com/Peterande/D-FINE.git "$DFINE_DIR"
    log_success "D-FINE repository cloned."
else
    log_success "D-FINE repository already present at $DFINE_DIR."
fi

# 5. Python Environment & Backend Setup
log_info "4/7 Setting up Python backend virtual environment & dependencies..."
cd "$APP_DIR/backend"

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate

pip install --upgrade pip -q
pip install -r requirements.txt -q
log_success "Backend Python dependencies installed."

# Configure .env file
log_info "Configuring backend .env settings..."
cat <<EOF > .env
WEIGHTS_DIR=$APP_DIR/backend/weights
DFINE_REPO_PATH=$DFINE_DIR
MIN_SCORE_FLOOR=0.05
CORS_ORIGINS=["*"]
EOF
log_success "Created/updated $APP_DIR/backend/.env"

# Verify Weight Files
echo ""
echo "------------------- WEIGHT FILES AUDIT -------------------"
DFINE_WEIGHT="$APP_DIR/backend/weights/dfine/best_stg1.pth"
RFDETR_WEIGHT="$APP_DIR/backend/weights/rfdetr/checkpoint_best_regular.pth"

if [ -f "$DFINE_WEIGHT" ]; then
    log_success "D-FINE weights found: $DFINE_WEIGHT"
else
    log_warn "D-FINE weights MISSING at: $DFINE_WEIGHT"
    log_warn "Please upload your best_stg1.pth to $APP_DIR/backend/weights/dfine/"
fi

if [ -f "$RFDETR_WEIGHT" ]; then
    log_success "RF-DETR weights found: $RFDETR_WEIGHT"
else
    log_warn "RF-DETR weights MISSING at: $RFDETR_WEIGHT"
    log_warn "Please upload your checkpoint_best_regular.pth to $APP_DIR/backend/weights/rfdetr/"
fi
echo "----------------------------------------------------------"
echo ""

# 6. Frontend Build
log_info "5/7 Installing frontend dependencies and building static bundle..."
cd "$APP_DIR/frontend"
npm install --silent
npm run build
log_success "Frontend built successfully at $APP_DIR/frontend/dist"

# 7. Configure Nginx
log_info "6/7 Configuring Nginx web server..."
cp "$APP_DIR/nginx/sld-inference.conf" /etc/nginx/sites-available/sld-inference
ln -sf /etc/nginx/sites-available/sld-inference /etc/nginx/sites-enabled/sld-inference
rm -f /etc/nginx/sites-enabled/default

nginx -t
systemctl reload nginx
log_success "Nginx configured and reloaded."

# 8. Start Backend Service
log_info "7/7 Starting FastAPI Uvicorn backend..."
cd "$APP_DIR/backend"

# Terminate existing uvicorn instances if any
pkill -f "uvicorn app.main:app" || true
sleep 1

# Start server in background
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 > "$APP_DIR/backend/backend.log" 2>&1 &

sleep 3

# Health check test
if curl -s http://127.0.0.1:8000/health | grep -q "ok"; then
    log_success "FastAPI Backend is running and HEALTHY on port 8000!"
else
    log_error "FastAPI Backend failed to respond to health check. Check logs at: $APP_DIR/backend/backend.log"
fi

# Detect Public IP
PUBLIC_IP=$(curl -s ifconfig.me || curl -s icanhazip.com || echo "YOUR-SERVER-IP")

echo ""
echo -e "${GREEN}=========================================================================="
echo "                  DEPLOYMENT COMPLETE & READY TO USE!                     "
echo "=========================================================================="
echo -e "${NC}"
echo -e "Access your viewer at:  ${CYAN}http://${PUBLIC_IP}/${NC}"
echo -e "Backend health status: ${CYAN}http://${PUBLIC_IP}/health${NC}"
echo -e "Backend log file:      ${CYAN}$APP_DIR/backend/backend.log${NC}"
echo ""
echo -e "${YELLOW}Important Reminder:${NC} Make sure Port 80 (HTTP) is allowed in your Alibaba Cloud ECS Security Group Rules."
echo ""
