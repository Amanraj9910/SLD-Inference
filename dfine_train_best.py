#!/usr/bin/env python3
"""
D-FINE-X SLD Training & Preprocessing (Unified Production Pipeline)
===================================================================

Combines full end-to-end preprocessing, MSAL Client Credentials Microsoft Graph API
authentication (headless, no rclone/browser/OAuth), dataset validation, annotation-safe
auto-cropping, scale normalization, adaptive tiling (resized to 640x640 model resolution),
instance-aware class balancing with minority augmentation, real log.txt (JSONL) monitoring,
clean epoch-boundary early stopping, top-5 checkpoint retention, automatic resume from remote,
and background Microsoft Graph synchronization.

Order of Execution:
  1. GPU & PyTorch Check
  2. Dependencies & det_solver Early-Stop Patching
  3. PyTorch GPU Check & VRAM Detection
  4. Entra ID MSAL Client Credentials Authentication Setup
  5. Microsoft Graph Connection & Drive Verification
  6. Pipeline & Run Configuration
  7. Pull Raw Dataset from Microsoft Graph
  8. Dataset Validation & Statistics (Raw Export)
  9. Scale Normalization (Circuit Breaker -> CT -> Fuse -> Fallback)
 10. Auto Crop White Margins (Bounded by Annotation Envelope)
 11. Adaptive Tiling & Resizing to 640x640 (Empty Tile Removal)
 12. Post-Preprocessing Dataset Validation & Statistics
 13. Instance-Aware Class Balancing + Minority Augmentation (Saved Locally Only)
 14. Pretrained Checkpoint Download (Objects365+COCO D-FINE-X)
 15. Dynamic Batch Size & LR Scaling based on VRAM
 16. Custom Training YAML Config Generation
 17. Verify D-FINE Imports
 18. Background Threads: Graph Sync + log.txt Monitoring + Top-5 Retention
 19. Launch D-FINE Training (Fresh Fine-Tune / Automatic Resume from Graph)
 20. Training Metrics Display
 21. Save Run Config & Final Microsoft Graph Sync
"""

import os
import sys
import json
import copy
import hashlib
import math
import random
import statistics
import shutil
import subprocess
import threading
import time
import urllib.request
import ast
import re
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import msal
import requests
from tqdm import tqdm

# ══════════════════════════════════════════════════════════════
# HELPER CLASSES — MSAL & Microsoft Graph Isolation Layer
# ══════════════════════════════════════════════════════════════

class TokenManager:
    """Manages Microsoft Entra ID authentication and token acquisition using Client Credentials Flow."""
    def __init__(self, client_id, client_secret, tenant_id):
        self.client_id = client_id.strip()
        self.client_secret = client_secret.strip()
        self.tenant_id = tenant_id.strip()
        self.authority = f"https://login.microsoftonline.com/{self.tenant_id}"
        self.scope = ["https://graph.microsoft.com/.default"]
        self.app = msal.ConfidentialClientApplication(
            self.client_id,
            client_credential=self.client_secret,
            authority=self.authority
        )

    def get_access_token(self):
        result = self.app.acquire_token_for_client(scopes=self.scope)
        if "access_token" in result:
            return result["access_token"]
        else:
            err = result.get("error_description", result.get("error"))
            raise RuntimeError(f"MSAL Token Acquisition Failed: {err}")


class GraphClient:
    """Handles HTTP communication with Microsoft Graph API using connection pooling & retries."""
    def __init__(self, token_manager, user_upn):
        self.token_manager = token_manager
        self.user_upn = user_upn.strip()
        self.base_url = f"https://graph.microsoft.com/v1.0/users/{self.user_upn}"
        self.session = requests.Session()
        self.drive_id = None

    def request(self, method, endpoint, headers=None, json=None, data=None, stream=False, timeout=60, max_retries=5):
        if headers is None:
            headers = {}

        url = endpoint if endpoint.startswith("https://") else f"{self.base_url}/{endpoint.lstrip('/')}"

        for attempt in range(max_retries):
            token = self.token_manager.get_access_token()
            req_headers = copy.deepcopy(headers)
            req_headers["Authorization"] = f"Bearer {token}"

            try:
                response = self.session.request(
                    method=method,
                    url=url,
                    headers=req_headers,
                    json=json,
                    data=data,
                    stream=stream,
                    timeout=timeout
                )

                if response.status_code in [200, 201, 202, 204, 206]:
                    return response
                elif response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", 2 ** attempt))
                    print(f"⚠️  Graph API Rate Limited (429). Retrying in {retry_after}s...")
                    time.sleep(retry_after)
                elif response.status_code in [500, 502, 503, 504]:
                    print(f"⚠️  Graph API Server Error ({response.status_code}). Retry {attempt + 1}/{max_retries}...")
                    time.sleep(2 ** attempt)
                elif response.status_code == 401:
                    print(f"⚠️  Graph API Token Unauthorized (401). Refreshing token and retrying...")
                    time.sleep(1)
                else:
                    response.raise_for_status()
            except requests.exceptions.RequestException as e:
                if attempt == max_retries - 1:
                    raise e
                print(f"⚠️  Network request failed ({e}). Retry {attempt + 1}/{max_retries}...")
                time.sleep(2 ** attempt)

        raise RuntimeError(f"Request failed after {max_retries} retries: {method} {url}")

    def verify_connection(self, dataset_folder, output_folder):
        """Verify connection to user drive and ensure SLD-Dataset and SLD-Training-Output exist."""
        print(f"Connecting to Microsoft Graph API for user {self.user_upn}...")
        resp = self.request("GET", "drive")
        drive_data = resp.json()
        self.drive_id = drive_data["id"]

        print("Connected successfully")
        print(f"Drive ID: {self.drive_id}")

        ds_item = self.get_item_by_path(dataset_folder)
        if not ds_item:
            raise FileNotFoundError(f"Dataset folder '{dataset_folder}' not found in {self.user_upn}'s OneDrive.")
        print(f"Dataset found: {dataset_folder}")

        out_item = self.get_item_by_path(output_folder)
        if not out_item:
            print(f"Output folder '{output_folder}' missing. Creating remote folder...")
            self.ensure_remote_folder(output_folder)
            print(f"Output folder created: {output_folder}")
        else:
            print(f"Output folder found: {output_folder}")

    def get_item_by_path(self, item_path):
        clean_path = item_path.strip("/")
        endpoint = "drive/root" if not clean_path else f"drive/root:/{clean_path}"
        try:
            resp = self.request("GET", endpoint)
            return resp.json()
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 404:
                return None
            raise e

    def ensure_remote_folder(self, folder_path):
        parts = [p for p in folder_path.strip("/").split("/") if p]
        current = ""
        for part in parts:
            parent = current
            current = f"{current}/{part}" if current else part
            if not self.get_item_by_path(current):
                parent_endpoint = f"drive/root:/{parent}:/children" if parent else "drive/root/children"
                body = {
                    "name": part,
                    "folder": {},
                    "@microsoft.graph.conflictBehavior": "fail"
                }
                self.request("POST", parent_endpoint, json=body)


class GraphDownloader:
    """Handles recursive downloading from Microsoft Graph with streaming and progress bars."""
    def __init__(self, graph_client):
        self.client = graph_client

    def list_children(self, item_path):
        clean_path = item_path.strip("/")
        endpoint = f"drive/root:/{clean_path}:/children" if clean_path else "drive/root/children"
        items = []
        next_url = endpoint
        while next_url:
            resp = self.client.request("GET", next_url)
            data = resp.json()
            items.extend(data.get("value", []))
            next_url = data.get("@odata.nextLink", None)
        return items

    def download_file(self, item, local_path):
        remote_size = item["size"]
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        if os.path.exists(local_path) and os.path.getsize(local_path) == remote_size:
            return

        download_url = item.get("@microsoft.graph.downloadUrl")
        if not download_url:
            download_url = f"{self.client.base_url}/drive/items/{item['id']}/content"

        resp = self.client.request("GET", download_url, stream=True)
        temp_path = f"{local_path}.tmp"

        with open(temp_path, "wb") as f, tqdm(
            total=remote_size, unit="B", unit_scale=True, desc=os.path.basename(local_path), leave=False
        ) as bar:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        if os.path.getsize(temp_path) != remote_size:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            raise IOError(f"Downloaded file size ({os.path.getsize(temp_path)}) does not match remote size ({remote_size}) for {local_path}")

        if os.path.exists(local_path):
            os.remove(local_path)
        os.rename(temp_path, local_path)

    def download_folder_recursive(self, remote_path, local_dir):
        os.makedirs(local_dir, exist_ok=True)
        items = self.list_children(remote_path)
        for item in items:
            item_name = item["name"]
            item_remote_path = f"{remote_path.rstrip('/')}/{item_name}"
            item_local_path = os.path.join(local_dir, item_name)

            if "folder" in item:
                self.download_folder_recursive(item_remote_path, item_local_path)
            else:
                ext = os.path.splitext(item_name)[1].lower()
                if ext in (".jpg", ".jpeg", ".png", ".json", ".txt", ".xml", ".pth", ".csv", ".yml", ".yaml"):
                    self.download_file(item, item_local_path)


class GraphUploader:
    """Handles recursive uploading to Microsoft Graph with chunked session uploads for large files."""
    def __init__(self, graph_client):
        self.client = graph_client

    def upload_file(self, local_path, remote_path):
        if not os.path.exists(local_path):
            return

        file_size = os.path.getsize(local_path)
        clean_remote_path = remote_path.strip("/")

        existing_item = self.client.get_item_by_path(clean_remote_path)
        if existing_item and existing_item.get("size") == file_size:
            return

        parent_dir = "/".join(clean_remote_path.split("/")[:-1])
        if parent_dir:
            self.client.ensure_remote_folder(parent_dir)

        if file_size < 4 * 1024 * 1024:
            endpoint = f"drive/root:/{clean_remote_path}:/content"
            headers = {"Content-Type": "application/octet-stream"}
            with open(local_path, "rb") as f:
                data = f.read()
            self.client.request("PUT", endpoint, headers=headers, data=data)
        else:
            create_session_endpoint = f"drive/root:/{clean_remote_path}:/createUploadSession"
            session_body = {
                "item": {
                    "@microsoft.graph.conflictBehavior": "replace",
                    "name": os.path.basename(local_path)
                }
            }
            resp = self.client.request("POST", create_session_endpoint, json=session_body)
            upload_url = resp.json()["uploadUrl"]

            chunk_size = 10 * 1024 * 1024
            with open(local_path, "rb") as f:
                start_byte = 0
                while start_byte < file_size:
                    chunk = f.read(chunk_size)
                    chunk_length = len(chunk)
                    end_byte = start_byte + chunk_length - 1

                    headers = {
                        "Content-Length": str(chunk_length),
                        "Content-Range": f"bytes {start_byte}-{end_byte}/{file_size}"
                    }

                    chunk_uploaded = False
                    for attempt in range(5):
                        try:
                            put_resp = self.client.session.put(upload_url, headers=headers, data=chunk, timeout=120)
                            if put_resp.status_code in [200, 201, 202]:
                                chunk_uploaded = True
                                break
                            elif put_resp.status_code in [429, 500, 502, 503, 504]:
                                time.sleep(2 ** attempt)
                            else:
                                put_resp.raise_for_status()
                        except Exception as e:
                            if attempt == 4:
                                raise e
                            time.sleep(2 ** attempt)

                    if not chunk_uploaded:
                        raise IOError(f"Failed to upload chunk {start_byte}-{end_byte} for {local_path}")

                    start_byte += chunk_length

    def upload_folder_recursive(self, local_dir, remote_folder):
        if not os.path.exists(local_dir):
            return 0

        uploaded_count = 0
        for root, dirs, files in os.walk(local_dir):
            rel_dir = os.path.relpath(root, local_dir)
            if rel_dir == ".":
                remote_subfolder = remote_folder
            else:
                remote_subfolder = f"{remote_folder.rstrip('/')}/{rel_dir.replace('\\', '/')}"

            for f in files:
                local_file_path = os.path.join(root, f)
                remote_file_path = f"{remote_subfolder.rstrip('/')}/{f}"
                self.upload_file(local_file_path, remote_file_path)
                uploaded_count += 1
        return uploaded_count

# ══════════════════════════════════════════════════════════════
# STEP 1 — GPU Check
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 1 — GPU Check")
print("=" * 60)
os.system("nvidia-smi")
print()
print("Check GPU count and per-GPU memory above before continuing.")
print()

# ══════════════════════════════════════════════════════════════
# STEP 2 — Install Dependencies & Patch det_solver.py
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 2 — Dependencies & Repository Setup")
print("=" * 60)
CUDA_TAG = "cu121"

os.system("sudo apt-get update -y -q")
os.system("sudo apt-get install -y -q git tmux unzip curl")
os.system(f"pip install --quiet torch torchvision --index-url https://download.pytorch.org/whl/{CUDA_TAG}")
os.system("pip install --quiet requests pandas pyyaml Pillow numpy msal tqdm matplotlib")

HOME = os.path.expanduser("~")
WORKSPACE = f"{HOME}/workspace"
DFINE_REPO = f"{WORKSPACE}/D-FINE"
os.makedirs(WORKSPACE, exist_ok=True)

if not os.path.exists(DFINE_REPO):
    os.system(f"git clone https://github.com/Peterande/D-FINE.git {DFINE_REPO}")
else:
    print("✅ D-FINE repo already exists, skipping clone")

os.system(f"pip install --quiet -r {DFINE_REPO}/requirements.txt")

# ---- Apply the Early-Stop Patch to D-FINE's det_solver.py ----
solver_path = f"{DFINE_REPO}/src/solver/det_solver.py"
with open(solver_path) as f:
    solver_src = f.read()

PATCH_MARKER = "STOP_REQUESTED"
if PATCH_MARKER not in solver_src:
    old_code = "        for epoch in range(start_epoch, args.epochs):\n            self.train_dataloader.set_epoch(epoch)"
    new_code = (
        "        for epoch in range(start_epoch, args.epochs):\n"
        '            if self.output_dir and (self.output_dir / "STOP_REQUESTED").exists():\n'
        '                print(f"STOP_REQUESTED found -- stopping cleanly before epoch {epoch} begins.")\n'
        "                break\n"
        "            self.train_dataloader.set_epoch(epoch)"
    )
    assert old_code in solver_src, (
        "det_solver.py's training loop pattern not found. The repository code may have changed."
    )
    solver_src = solver_src.replace(old_code, new_code)
    with open(solver_path, "w") as f:
        f.write(solver_src)
    print("✅ Early-stop patch successfully applied to det_solver.py")
else:
    print("✅ Early-stop patch already present in det_solver.py")

ast.parse(solver_src)
print("✅ Patched det_solver.py verified valid Python syntax")
print("✅ Step 2 complete\n")

# ══════════════════════════════════════════════════════════════
# STEP 3 — PyTorch GPU Check & VRAM Detection
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 3 — PyTorch GPU Check")
print("=" * 60)
import torch

NUM_GPUS = torch.cuda.device_count()
if NUM_GPUS == 0:
    raise RuntimeError("No GPU visible to PyTorch. Check drivers and instance type.")

print(f"=== {NUM_GPUS} GPU(s) visible ===")
gpu_vram_gb = []
for i in range(NUM_GPUS):
    total_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
    gpu_vram_gb.append(total_gb)
    print(f"GPU {i}: {torch.cuda.get_device_name(i)} — {total_gb:.1f} GB")

MIN_VRAM_GB = min(gpu_vram_gb)
if MIN_VRAM_GB < 8:
    print("⚠️  Under 8GB VRAM on at least one GPU. Lower batch size to prevent OOM.")
print()

# ══════════════════════════════════════════════════════════════
# STEP 4 — Entra ID Credentials & MSAL Setup
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 4 — Entra ID MSAL Credentials Setup")
print("=" * 60)

CLIENT_ID = "YOUR_CLIENT_ID"
CLIENT_SECRET = "YOUR_CLIENT_SECRET"
TENANT_ID = "YOUR_TENANT_ID"
USER_UPN = "amanr@hoshodigital.com"
TARGET_DATASET_NAME = "SLD-Dataset"
TARGET_OUTPUT_NAME = "SLD-Training-Output"

token_manager = TokenManager(CLIENT_ID, CLIENT_SECRET, TENANT_ID)
graph_client = GraphClient(token_manager, USER_UPN)
graph_downloader = GraphDownloader(graph_client)
graph_uploader = GraphUploader(graph_client)

print("✅ MSAL TokenManager & GraphClient initialized\n")

# ══════════════════════════════════════════════════════════════
# STEP 5 — Test Microsoft Graph Connection & Verify Folders
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 5 — Test Microsoft Graph Connection & Verify Folders")
print("=" * 60)

graph_client.verify_connection(TARGET_DATASET_NAME, TARGET_OUTPUT_NAME)
print()

# ══════════════════════════════════════════════════════════════
# STEP 6 — Run Name & Pipeline Configuration
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 6 — Run & Pipeline Configuration")
print("=" * 60)

RUN_NAME = "sld_dfinex_unified_v1"

CLOUD_DATASET_REMOTE = TARGET_DATASET_NAME
CLOUD_OUTPUT_REMOTE_BASE = f"{TARGET_OUTPUT_NAME}/exp_{RUN_NAME}"

RAW_DIR = f"{HOME}/data/sld_raw"
PROCESSED_DIR = f"{HOME}/data/sld_processed"
OUTPUT_DIR_BASE = f"{HOME}/workspace/dfine_sld/exp_{RUN_NAME}"

# OUTPUT_DIR and CLOUD_OUTPUT_REMOTE finalized after class-hash computation (Step 7B)
for d in [RAW_DIR, PROCESSED_DIR]:
    os.makedirs(d, exist_ok=True)

PIPELINE_CONFIG = {
    # Feature Toggles
    "ENABLE_SCALE_NORMALIZATION": True,
    "ENABLE_AUTO_CROP": True,
    "ENABLE_ADAPTIVE_TILING": True,
    "ENABLE_REMOVE_EMPTY_TILES": True,
    "ENABLE_CLASS_BALANCING": True,

    # Scale Normalization
    "REFERENCE_CLASS_PRIORITY": ["Circuit Breaker", "Current Transformer", "Fuse"],
    "TARGET_REFERENCE_HEIGHT": 60.0,  # px target for reference class symbol height
    "MIN_PLAUSIBLE_REF_HEIGHT_PX": 8.0,  # reference annotations smaller than this are treated as mislabeled, not real symbols
    "ENABLE_TILE_QA_SAMPLES": True,  # render sample tiles with boxes drawn, for visual sanity-checking
    "QA_SAMPLES_PER_SPLIT": 12,
    "MAX_RESIZED_PIXELS": 150_000_000,  # safety ceiling on scale-norm output size (Fix: OOM guard)

    # Adaptive Tiling & Resizing
    "MODEL_INPUT_SIZE": 640,           # Native D-FINE tile resolution (640x640)
    "TARGET_SYMBOL_PX": 48,            # Aim for median symbol size to be ~48px in 640x640 tile
    "TILE_OVERLAP": 0.20,              # 20% overlap between tiles
    "MIN_VISIBLE_AREA": 0.50,          # Keep cropped annotation if >=50% area visible

    # Auto Crop Safety
    "WHITE_THRESHOLD": 240,
    "BLANK_ROW_FRACTION": 0.985,
    "MIN_MARGIN_PX": 5,
    "ANNOTATION_SAFETY_MARGIN_PX": 8,  # Envelope buffer: minimum px guard
    "ANNOTATION_SAFETY_MARGIN_PCT": 0.03,  # 3% of image dimension as per-image safety buffer (Fix 2.4)

    # Class Balancing
    "MAX_REPEAT_FACTOR": 8.0,
    "DENSITY_WEIGHT": 0.5,
}

print(f"Run name          : {RUN_NAME}")
print(f"Raw Dataset       : {CLOUD_DATASET_REMOTE}  ->  {RAW_DIR}")
print(f"Processed Dataset : {PROCESSED_DIR}")
print(f"Output Base       : {CLOUD_OUTPUT_REMOTE_BASE}  <-  {OUTPUT_DIR_BASE} (hash-versioned after dataset pull)\n")
print("Pipeline Configuration:")
for k, v in PIPELINE_CONFIG.items():
    print(f"  {k:32s} = {v}")
print()

# ══════════════════════════════════════════════════════════════
# STEP 7 — Pull Raw Dataset from Microsoft Graph
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7 — Pull Raw Dataset from Microsoft Graph")
print("=" * 60)

print(f"Syncing raw dataset from Graph path '{CLOUD_DATASET_REMOTE}' ...")
graph_downloader.download_folder_recursive(CLOUD_DATASET_REMOTE, RAW_DIR)

for split in ["train", "valid", "test"]:
    split_dir = os.path.join(RAW_DIR, split)
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Missing split folder: {split_dir}")
    n_imgs = len([f for f in os.listdir(split_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    print(f"✅ {split}: {n_imgs} images")
print()

# ══════════════════════════════════════════════════════════════
# STEP 7B — Compute Class Hash & Finalize Output Directory
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 7B — Compute Class Hash & Finalize Output Directory")
print("=" * 60)

# Read categories from ALL raw splits to get the full class list
_raw_categories_for_hash = {}
for split in ["train", "valid", "test"]:
    _anno = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    if os.path.exists(_anno):
        with open(_anno) as f:
            _c = json.load(f)
        for cat in _c.get("categories", []):
            _raw_categories_for_hash[cat["id"]] = cat

_sorted_class_names = sorted(cat["name"] for cat in _raw_categories_for_hash.values())
CLASS_HASH = hashlib.md5("_".join(_sorted_class_names).encode()).hexdigest()[:8]

OUTPUT_DIR = f"{OUTPUT_DIR_BASE}/cls_{CLASS_HASH}"
CLOUD_OUTPUT_REMOTE = f"{CLOUD_OUTPUT_REMOTE_BASE}/cls_{CLASS_HASH}"
CLOUD_PROCESSED_REMOTE = None  # Processed dataset kept locally only (OneDrive upload disabled)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Classes found in raw dataset: {len(_raw_categories_for_hash)}")
for _cid, _cat in sorted(_raw_categories_for_hash.items()):
    print(f"  id={_cid:>3}  {_cat['name']}")
print(f"Class hash      : {CLASS_HASH}")
print(f"Output dir      : {OUTPUT_DIR}")
print(f"Cloud remote    : {CLOUD_OUTPUT_REMOTE}")
print(f"Processed dataset: {PROCESSED_DIR} (Stored locally only, not saved to OneDrive)")
print()

# ══════════════════════════════════════════════════════════════
# RESUME CHECK — Check Remote Graph for Existing Checkpoint
# ══════════════════════════════════════════════════════════════
remote_last_item = graph_client.get_item_by_path(f"{CLOUD_OUTPUT_REMOTE}/last.pth")
if remote_last_item:
    print(f"🔄 Remote checkpoint 'last.pth' found in '{CLOUD_OUTPUT_REMOTE}'. Syncing experiment state to resume training...")
    graph_downloader.download_folder_recursive(CLOUD_OUTPUT_REMOTE, OUTPUT_DIR)
    print(f"✅ Resuming checkpoint state downloaded to {OUTPUT_DIR}\n")

# ══════════════════════════════════════════════════════════════
# STEP 8 — Dataset Validation & Statistics (Raw Export)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 8 — Dataset Validation & Statistics")
print("=" * 60)

def validate_coco_dataset(anno_path, img_dir, split_name):
    """Validate a single COCO annotation file + image directory."""
    errors, warnings = [], []
    if not os.path.exists(anno_path):
        errors.append(f"[{split_name}] Missing annotation file: {anno_path}")
        return errors, warnings

    with open(anno_path) as f:
        try:
            coco = json.load(f)
        except json.JSONDecodeError as e:
            errors.append(f"[{split_name}] Invalid JSON in {anno_path}: {e}")
            return errors, warnings

    for key in ["images", "annotations", "categories"]:
        if key not in coco:
            errors.append(f"[{split_name}] Missing top-level key: '{key}'")
            return errors, warnings

    if len(coco["images"]) == 0:
        errors.append(f"[{split_name}] No images in annotation file")
    if len(coco["annotations"]) == 0:
        warnings.append(f"[{split_name}] No annotations in annotation file")

    valid_cat_ids = {c["id"] for c in coco["categories"]}
    image_ids = [img["id"] for img in coco["images"]]
    image_id_set = set(image_ids)
    anno_ids = [a["id"] for a in coco["annotations"]]

    if len(image_ids) != len(image_id_set):
        errors.append(f"[{split_name}] Duplicate image IDs found")

    if len(anno_ids) != len(set(anno_ids)):
        errors.append(f"[{split_name}] Duplicate annotation IDs found")

    existing_files = set(os.listdir(img_dir)) if os.path.exists(img_dir) else set()
    missing_images = sum(1 for img in coco["images"] if img["file_name"] not in existing_files)
    if missing_images > 0:
        errors.append(f"[{split_name}] {missing_images} referenced images missing on disk")

    image_dims = {img["id"]: (img.get("width"), img.get("height")) for img in coco["images"]}

    invalid_boxes = 0
    nonfinite_boxes = 0
    oob_boxes = 0
    invalid_cat_refs = 0
    orphan_annos = 0
    for ann in coco["annotations"]:
        if ann["category_id"] not in valid_cat_ids:
            invalid_cat_refs += 1
        if ann["image_id"] not in image_id_set:
            orphan_annos += 1
        if "bbox" in ann:
            x, y, w, h = ann["bbox"]
            if any(not math.isfinite(v) for v in (x, y, w, h)):
                nonfinite_boxes += 1
                continue
            if w <= 0 or h <= 0 or x < 0 or y < 0:
                invalid_boxes += 1
                continue
            img_w, img_h = image_dims.get(ann["image_id"], (None, None))
            if img_w and img_h and (x + w > img_w + 0.5 or y + h > img_h + 0.5):
                oob_boxes += 1

    if invalid_boxes > 0:
        errors.append(f"[{split_name}] {invalid_boxes} invalid bounding boxes (negative/zero dimensions)")
    if nonfinite_boxes > 0:
        errors.append(f"[{split_name}] {nonfinite_boxes} bounding boxes contain NaN/Inf values")
    if oob_boxes > 0:
        errors.append(f"[{split_name}] {oob_boxes} bounding boxes extend outside their image's width/height")
    if invalid_cat_refs > 0:
        errors.append(f"[{split_name}] {invalid_cat_refs} invalid category ID references")
    if orphan_annos > 0:
        errors.append(f"[{split_name}] {orphan_annos} orphan annotations (non-existent image_id)")

    return errors, warnings


def validate_or_die(anno_path: str, img_dir: str, split_name: str, stage_label: str) -> None:
    errs, warns = validate_coco_dataset(anno_path, img_dir, split_name)
    for w in warns:
        print(f"⚠️  [{stage_label}] {w}")
    for e in errs:
        print(f"❌ [{stage_label}] {e}")
    if errs:
        raise ValueError(f"[{stage_label}] Validation failed for '{split_name}'. Halting execution.\n" + "\n".join(errs))


def compute_dataset_statistics(coco: dict, label: str) -> dict:
    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = {c["id"]: c["name"] for c in coco.get("categories", [])}

    per_class_count = defaultdict(int)
    widths, heights, areas = [], [], []
    anns_per_image = defaultdict(int)
    for ann in annotations:
        per_class_count[ann["category_id"]] += 1
        x, y, w, h = ann["bbox"]
        widths.append(w)
        heights.append(h)
        areas.append(ann.get("area", w * h))
        anns_per_image[ann["image_id"]] += 1

    def _median(vals):
        if not vals:
            return 0
        s = sorted(vals)
        return s[len(s) // 2]

    stats = {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_classes": len(categories),
        "median_bbox_width": _median(widths),
        "median_bbox_height": _median(heights),
        "avg_bbox_area": (sum(areas) / len(areas)) if areas else 0,
        "avg_anns_per_image": (sum(anns_per_image.values()) / len(images)) if images else 0,
    }

    print(f"--- Dataset Statistics [{label}] ---")
    print(f"  Images             : {stats['num_images']}")
    print(f"  Annotations        : {stats['num_annotations']}")
    print(f"  Classes            : {stats['num_classes']}")
    print(f"  Median bbox width  : {stats['median_bbox_width']:.1f}px")
    print(f"  Median bbox height : {stats['median_bbox_height']:.1f}px")
    print(f"  Avg bbox area      : {stats['avg_bbox_area']:.1f}px²")
    print(f"  Avg anns/image     : {stats['avg_anns_per_image']:.2f}")
    print()
    return stats


for split in ["train", "valid", "test"]:
    anno_path = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    img_dir = os.path.join(RAW_DIR, split)
    validate_or_die(anno_path, img_dir, split, "Raw Pre-Check")
print("✅ Raw dataset validation passed across all splits")

# Strip known-junk categories (e.g. Roboflow-auto-generated placeholder classes that
# were never intentionally labeled) before anything downstream can see or count them.
# Add any other confirmed-junk class names to this set as they're found.
EXCLUDED_CLASSES = {"SLD-ANNOTATION"}
for split in ["train", "valid", "test"]:
    _anno_path = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    if not os.path.exists(_anno_path):
        continue
    with open(_anno_path) as f:
        _c = json.load(f)
    _drop_ids = {cat["id"] for cat in _c["categories"] if cat["name"] in EXCLUDED_CLASSES}
    if _drop_ids:
        _before = len(_c["annotations"])
        _c["categories"] = [cat for cat in _c["categories"] if cat["id"] not in _drop_ids]
        _c["annotations"] = [a for a in _c["annotations"] if a["category_id"] not in _drop_ids]
        print(f"🧹 [{split}] Dropped {_before - len(_c['annotations'])} annotation(s) "
              f"for excluded class(es): {sorted(n for n in EXCLUDED_CLASSES)}")
        with open(_anno_path, "w") as f:
            json.dump(_c, f)
print()

# ══════════════════════════════════════════════════════════════
# STEP 8B — Canonical Class-ID Remap  (fixes CUDA "index out of bounds")
# ══════════════════════════════════════════════════════════════
# ROOT CAUSE THIS SECTION FIXES:
#   Raw COCO exports do NOT guarantee category["id"] values form a
#   contiguous 0..N-1 range. Roboflow in particular reserves an id (very
#   often id=0) for a placeholder/supercategory — e.g. the "SLD-ANNOTATION"
#   class stripped just above. When that placeholder is dropped, the
#   *remaining* category ids are never renumbered, so real classes can sit
#   at ids 1..N. Downstream, NUM_CLASSES is computed as a *count*
#   (len(categories)), not max(id)+1 — so D-FINE's classification head /
#   label embeddings end up sized for exactly NUM_CLASSES valid indices
#   (0..NUM_CLASSES-1) while real category_id values used as training
#   labels can be >= NUM_CLASSES. The very first batch containing such a
#   label indexes past the end of that table -> CUDA device-side assert
#   ("index out of bounds"). Once that assert fires, the CUDA context is
#   poisoned for the rest of the process — which is almost certainly why a
#   *second*, seemingly unrelated assertion (generalized_box_iou's
#   `boxes1[:, 2:] >= boxes1[:, :2]` inside the matcher) shows up
#   immediately after in the same crash: it's reading corrupted state from
#   an already-broken context, not an independent bug in your tiling code.
#
# FIX: maintain ONE persisted, NAME-keyed registry (class_id_registry.json)
# that assigns contiguous 0-indexed ids and is ONLY EVER APPENDED TO —
# existing class names keep their id forever, matching the "category IDs
# must never change between rounds, only appended" rule. Every raw split
# file is rewritten in place to use these canonical ids before tiling,
# balancing, or class-hashing ever sees them.

CLASS_REGISTRY_PATH = f"{WORKSPACE}/class_id_registry.json"

def load_class_registry(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}

def assign_canonical_category_ids(raw_dir: str, splits: list, registry_path: str) -> dict:
    """
    Reads categories (by NAME, across all raw splits), assigns each a stable
    contiguous 0-indexed id from the persisted registry — reusing existing
    ids for known names, appending brand-new names at the end — and returns
    {name: canonical_id}. An id, once assigned, is NEVER reused or shifted.
    """
    registry = load_class_registry(registry_path)  # name -> id, {} on first run

    names_seen = set()
    for split in splits:
        anno_path = os.path.join(raw_dir, split, "_annotations.coco.json")
        if not os.path.exists(anno_path):
            continue
        with open(anno_path) as f:
            c = json.load(f)
        for cat in c.get("categories", []):
            names_seen.add(cat["name"])

    next_id = (max(registry.values()) + 1) if registry else 0
    new_names = sorted(n for n in names_seen if n not in registry)
    for name in new_names:
        registry[name] = next_id
        next_id += 1

    if new_names:
        print(f"🆕 Assigned canonical ids to {len(new_names)} new class name(s): "
              f"{', '.join(f'{n}={registry[n]}' for n in new_names)}")

    unused = sorted(n for n in registry if n not in names_seen)
    if unused:
        print(f"⚠️  {len(unused)} class(es) in the registry have ZERO instances in this "
              f"raw pull (ids kept, never reused — verify this is expected): {unused}")

    with open(registry_path, "w") as f:
        json.dump(registry, f, indent=2, sort_keys=True)

    ids = sorted(registry.values())
    assert ids == list(range(len(ids))), (
        f"🚫 Class registry corrupted — ids are not contiguous 0..N-1: {registry}. "
        f"Inspect or delete '{registry_path}' before re-running (deleting forces a "
        f"full re-derivation of ids from scratch — only safe if NOT resuming from an "
        f"existing checkpoint, since class identity/order would be reset)."
    )
    return registry


def remap_coco_category_ids_inplace(anno_path: str, name_to_id: dict, split_name: str) -> None:
    """Rewrites categories[].id and every annotation.category_id in place
    using the canonical name -> id mapping (other category fields, e.g.
    supercategory, are preserved). Raises if a category name in the file is
    missing from the mapping — should never happen, since the mapping is
    built from the union of all splits before this is called."""
    with open(anno_path) as f:
        c = json.load(f)

    old_id_to_name = {cat["id"]: cat["name"] for cat in c["categories"]}
    missing = [n for n in old_id_to_name.values() if n not in name_to_id]
    if missing:
        raise ValueError(f"[{split_name}] Class name(s) not in canonical registry: {missing}")

    by_name = {}
    for cat in c["categories"]:
        cat["id"] = name_to_id[cat["name"]]
        by_name[cat["name"]] = cat
    c["categories"] = sorted(by_name.values(), key=lambda cat: cat["id"])

    for ann in c["annotations"]:
        old_name = old_id_to_name.get(ann["category_id"])
        if old_name is None:
            raise ValueError(
                f"[{split_name}] Annotation references category_id "
                f"{ann['category_id']} not present in this file's categories list."
            )
        ann["category_id"] = name_to_id[old_name]

    with open(anno_path, "w") as f:
        json.dump(c, f)


def validate_category_id_contiguity_or_die(categories: list, annotations: list,
                                            expected_num_classes: int,
                                            split_name: str, stage_label: str) -> None:
    """Hard gate: category ids must be exactly {0, 1, ..., N-1} and every
    annotation's category_id must fall in that range. D-FINE's
    classification head / label embeddings are sized to NUM_CLASSES — any
    label outside this range crashes CUDA with an 'index out of bounds'
    device-side assert, often several batches into training rather than at
    load time. This check catches it immediately, before any GPU time is
    spent, and refuses to launch training on the affected data."""
    cat_ids = sorted({c["id"] for c in categories})
    errors = []
    if cat_ids != list(range(expected_num_classes)):
        missing = sorted(set(range(expected_num_classes)) - set(cat_ids))
        extra = sorted(set(cat_ids) - set(range(expected_num_classes)))
        errors.append(
            f"[{split_name}] category ids {cat_ids} are not exactly 0..{expected_num_classes - 1}"
            f"{f' (missing: {missing})' if missing else ''}{f' (unexpected: {extra})' if extra else ''}"
        )
    bad_labels = sorted({a["category_id"] for a in annotations
                          if not (0 <= a["category_id"] < expected_num_classes)})
    if bad_labels:
        errors.append(
            f"[{split_name}] {len(bad_labels)} annotation category_id value(s) fall "
            f"outside [0, {expected_num_classes}): {bad_labels[:20]}"
            f"{' ...' if len(bad_labels) > 20 else ''}"
        )
    if errors:
        raise ValueError(
            f"🚫 [{stage_label}] Class-ID contiguity check FAILED — refusing to launch "
            f"training on this data (this is exactly the condition that causes the CUDA "
            f"'index out of bounds' crash):\n" + "\n".join(errors)
        )
    print(f"✅ [{stage_label}] [{split_name}] {expected_num_classes} classes, ids contiguous "
          f"0..{expected_num_classes - 1}, all annotation labels in range.")


_canonical_name_to_id = assign_canonical_category_ids(RAW_DIR, ["train", "valid", "test"], CLASS_REGISTRY_PATH)
_expected_num_classes = len(_canonical_name_to_id)
_registry_categories = [{"id": cid, "name": name} for name, cid in _canonical_name_to_id.items()]

for split in ["train", "valid", "test"]:
    _anno_path = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    if not os.path.exists(_anno_path):
        continue
    remap_coco_category_ids_inplace(_anno_path, _canonical_name_to_id, split)
    with open(_anno_path) as f:
        _remapped = json.load(f)
    validate_category_id_contiguity_or_die(
        _registry_categories, _remapped["annotations"], _expected_num_classes, split, "Post-Remap Check (Raw)"
    )
print(f"✅ All raw splits remapped to canonical contiguous class ids (0..{_expected_num_classes - 1})")
print(f"   Registry: {CLASS_REGISTRY_PATH}\n")

_raw_combined = {"images": [], "annotations": [], "categories": []}
_seen_cats = {}
for split in ["train", "valid", "test"]:
    anno_path = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    if os.path.exists(anno_path):
        with open(anno_path) as f:
            _c = json.load(f)
        _raw_combined["images"].extend(_c["images"])
        _raw_combined["annotations"].extend(_c["annotations"])
        for cat in _c["categories"]:
            _seen_cats[cat["id"]] = cat
_raw_combined["categories"] = list(_seen_cats.values())
compute_dataset_statistics(_raw_combined, "Raw Combined Export")

# ══════════════════════════════════════════════════════════════
# STEP 9-11 — Full Preprocessing Pipeline
# (Scale Normalization -> Safe Auto Crop -> Adaptive Tiling & 640x640 Resizing)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 9-11 — Preprocessing (Scale Norm + Safe Auto-Crop + Adaptive Tiling)")
print("=" * 60)

import numpy as np
from PIL import Image, ImageDraw
# These SLD source sheets are legitimately large (100MP+) and trusted (our own dataset,
# not untrusted uploads) — PIL's generic decompression-bomb warning just adds log noise
# here. Our own MAX_RESIZED_PIXELS cap (Step 9-11) is the real safety net against runaway
# memory use after scale normalization, so we disable PIL's blanket pixel-count check.
Image.MAX_IMAGE_PIXELS = None

def save_image_smart(image, path: str) -> None:
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        image.convert("RGB").save(path, format="JPEG", quality=95)
    elif ext == ".png":
        image.save(path, format="PNG")
    else:
        image.save(path)


def compute_scale_factors(coco, class_name_to_id, target_height=60.0):
    ref_priority = PIPELINE_CONFIG["REFERENCE_CLASS_PRIORITY"]
    # Case-insensitive match against the dataset's real category names — the previous
    # exact-match check silently failed every run because the config list was Title Case
    # ("Circuit Breaker") while the dataset's categories are ALL CAPS.
    name_lookup = {n.strip().lower(): cid for n, cid in class_name_to_id.items()}
    matched_names = [n for n in ref_priority if n.strip().lower() in name_lookup]
    ref_ids = [name_lookup[n.strip().lower()] for n in matched_names]
    if not ref_ids:
        print(f"⚠️  Scale-Norm: reference classes {ref_priority} matched NOTHING in this dataset. "
              f"Available class names: {sorted(class_name_to_id.keys())}. "
              f"Falling back to 1.0x for every image — scale normalization is OFF this run. "
              f"Update PIPELINE_CONFIG['REFERENCE_CLASS_PRIORITY'] to match a real class name above.")
        return {img["id"]: 1.0 for img in coco["images"]}, {"with_ref": 0, "fallback": len(coco["images"]), "avg": 1.0}
    else:
        print(f"✅ Scale-Norm: using reference class(es) {matched_names} for per-image scale calibration")

    heights = defaultdict(lambda: defaultdict(list))
    for ann in coco["annotations"]:
        heights[ann["image_id"]][ann["category_id"]].append(ann["bbox"][3])

    # A handful of implausibly tiny reference-class boxes (mislabeled/degenerate
    # annotations) were dragging per-image scale factors up to 9-18x, which is almost
    # certainly a labeling error rather than a real physical drawing-scale difference —
    # a genuine Circuit Breaker/Current Transformer symbol won't legitimately be a few
    # pixels tall. Filter those out before computing scale, and use median (not mean)
    # so a single remaining outlier annotation can't skew one image's whole scale.
    min_plausible = PIPELINE_CONFIG["MIN_PLAUSIBLE_REF_HEIGHT_PX"]
    flagged_images = []

    per_image_scale, with_ref = {}, []
    for img in coco["images"]:
        iid = img["id"]
        chosen = None
        for cid in ref_ids:
            raw = heights[iid].get(cid)
            if not raw:
                continue
            plausible = [h for h in raw if h >= min_plausible]
            if len(plausible) < len(raw):
                flagged_images.append((img.get("file_name", iid), cid, len(raw) - len(plausible), min(raw)))
            if plausible:
                chosen = plausible
                break
        if chosen:
            scale = target_height / statistics.median(chosen)
            per_image_scale[iid] = scale
            with_ref.append(scale)
        else:
            per_image_scale[iid] = None

    if flagged_images:
        print(f"⚠️  Scale-Norm: {len(flagged_images)} image(s) had reference-class annotations "
              f"under {min_plausible}px (excluded as likely mislabeled) — worth a manual look:")
        for fname, cid, n_dropped, min_h in flagged_images[:10]:
            print(f"     {fname}: {n_dropped} annotation(s) as small as {min_h:.1f}px in category {cid}")

    dataset_avg = statistics.median(with_ref) if with_ref else 1.0
    fallback_n = 0
    for iid, f in per_image_scale.items():
        if f is None:
            per_image_scale[iid] = dataset_avg
            fallback_n += 1

    return per_image_scale, {"with_ref": len(with_ref), "fallback": fallback_n, "avg": dataset_avg}


def detect_white_margins(img_array, white_thresh=240, blank_frac=0.985):
    h, w = img_array.shape[:2]
    is_white = np.all(img_array > white_thresh, axis=2) if img_array.ndim == 3 else img_array > white_thresh

    top = 0
    for r in range(h):
        if np.mean(is_white[r, :]) < blank_frac:
            break
        top = r + 1

    bottom = 0
    for r in range(h - 1, -1, -1):
        if np.mean(is_white[r, :]) < blank_frac:
            break
        bottom = h - r

    left = 0
    for c in range(w):
        if np.mean(is_white[:, c]) < blank_frac:
            break
        left = c + 1

    right = 0
    for c in range(w - 1, -1, -1):
        if np.mean(is_white[:, c]) < blank_frac:
            break
        right = w - c

    return top, bottom, left, right


def compute_annotation_envelope(anns: list) -> tuple:
    valid = [a["bbox"] for a in anns if a["bbox"][2] > 0 and a["bbox"][3] > 0]
    if not valid:
        return None
    min_x = min(b[0] for b in valid)
    min_y = min(b[1] for b in valid)
    max_x = max(b[0] + b[2] for b in valid)
    max_y = max(b[1] + b[3] for b in valid)
    return min_x, min_y, max_x, max_y


def clip_box(x_min, y_min, x_max, y_max, cx1, cy1, cx2, cy2):
    orig_area = max(0.0, x_max - x_min) * max(0.0, y_max - y_min)
    if orig_area <= 0:
        return None
    ix1, iy1 = max(x_min, cx1), max(y_min, cy1)
    ix2, iy2 = min(x_max, cx2), min(y_max, cy2)
    if ix1 >= ix2 or iy1 >= iy2:
        return None
    return ix1, iy1, ix2, iy2, ((ix2 - ix1) * (iy2 - iy1)) / orig_area


def median_symbol_size(anns):
    hs = sorted(a["bbox"][3] for a in anns if a["bbox"][3] > 0)
    if not hs:
        return 48.0
    return hs[len(hs) // 2]


def preprocess_split(split, raw_dir, processed_dir):
    src_dir = os.path.join(raw_dir, split)
    dst_dir = os.path.join(processed_dir, split)
    os.makedirs(dst_dir, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # Remove stale tiles from previous runs so the JSON-vs-disk check is clean
    for _old_f in os.listdir(dst_dir):
        if _old_f.lower().endswith((".png", ".jpg", ".jpeg")):
            os.remove(os.path.join(dst_dir, _old_f))

    with open(os.path.join(src_dir, "_annotations.coco.json")) as f:
        coco = json.load(f)

    class_name_to_id = {c["name"]: c["id"] for c in coco["categories"]}
    valid_cat_ids = {c["id"] for c in coco["categories"]}
    per_image_scale, scale_stats = compute_scale_factors(
        coco, class_name_to_id, PIPELINE_CONFIG["TARGET_REFERENCE_HEIGHT"]
    )

    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    new_images, new_annotations = [], []
    next_img_id, next_ann_id = 1, 1
    n_tiles_total = 0
    n_tiles_discarded_empty = 0
    n_tiles_discarded_degenerate = 0
    n_invalid_boxes = 0
    n_images_processed = 0
    saved_filenames = set()

    for img in coco["images"]:
        iid = img["id"]
        fname = img["file_name"]
        im_path = os.path.join(src_dir, fname)
        if not os.path.exists(im_path):
            continue

        im = Image.open(im_path).convert("RGB")

        # 1. Scale Normalization
        s = per_image_scale[iid] if PIPELINE_CONFIG["ENABLE_SCALE_NORMALIZATION"] else 1.0
        max_px = PIPELINE_CONFIG["MAX_RESIZED_PIXELS"]
        projected_px = (im.width * s) * (im.height * s)
        if projected_px > max_px:
            cap_scale = (max_px / (im.width * im.height)) ** 0.5
            print(f"⚠️  [img {iid}] Scale-Norm factor {s:.2f}x on a {im.width}x{im.height} source "
                  f"would produce {projected_px/1e6:.0f}MP (over the {max_px/1e6:.0f}MP safety ceiling). "
                  f"Clamping scale to {cap_scale:.2f}x to avoid an out-of-memory kill.")
            s = min(s, cap_scale)
        new_w, new_h = max(32, round(im.width * s)), max(32, round(im.height * s))
        im = im.resize((new_w, new_h), Image.Resampling.BILINEAR)

        scaled_anns = []
        for ann in anns_by_image.get(iid, []):
            x, y, w, h = ann["bbox"]
            new_ann = copy.deepcopy(ann)
            new_ann["bbox"] = [x * s, y * s, w * s, h * s]
            scaled_anns.append(new_ann)

        # 2. Safe Auto Crop (bounded by annotation envelope)
        if PIPELINE_CONFIG["ENABLE_AUTO_CROP"]:
            arr = np.array(im)
            top, bottom, left, right = detect_white_margins(
                arr, PIPELINE_CONFIG["WHITE_THRESHOLD"], PIPELINE_CONFIG["BLANK_ROW_FRACTION"]
            )
            del arr  # free the full-size copy immediately — not needed past this point
            envelope = compute_annotation_envelope(scaled_anns)
            if envelope is not None:
                env_min_x, env_min_y, env_max_x, env_max_y = envelope
                # Per-image percentage-based safety margin (Fix 2.4)
                margin_pct = PIPELINE_CONFIG["ANNOTATION_SAFETY_MARGIN_PCT"]
                buf_x = max(PIPELINE_CONFIG["ANNOTATION_SAFETY_MARGIN_PX"], int(new_w * margin_pct))
                buf_y = max(PIPELINE_CONFIG["ANNOTATION_SAFETY_MARGIN_PX"], int(new_h * margin_pct))
                top = max(0, min(top, int(env_min_y) - buf_y))
                left = max(0, min(left, int(env_min_x) - buf_x))
                bottom = max(0, min(bottom, int(new_h - env_max_y) - buf_y))
                right = max(0, min(right, int(new_w - env_max_x) - buf_x))

            crop_top = max(0, top - PIPELINE_CONFIG["MIN_MARGIN_PX"])
            crop_bottom = max(0, bottom - PIPELINE_CONFIG["MIN_MARGIN_PX"])
            crop_left = max(0, left - PIPELINE_CONFIG["MIN_MARGIN_PX"])
            crop_right = max(0, right - PIPELINE_CONFIG["MIN_MARGIN_PX"])

            cx1, cy1 = crop_left, crop_top
            cx2, cy2 = new_w - crop_right, new_h - crop_bottom
            if cx2 - cx1 >= 50 and cy2 - cy1 >= 50:
                im = im.crop((cx1, cy1, cx2, cy2))
                cropped_anns = []
                for ann in scaled_anns:
                    x, y, w, h = ann["bbox"]
                    clipped = clip_box(x, y, x + w, y + h, cx1, cy1, cx2, cy2)
                    if clipped:
                        ix1, iy1, ix2, iy2, _ = clipped
                        new_ann = copy.deepcopy(ann)
                        new_ann["bbox"] = [ix1 - cx1, iy1 - cy1, ix2 - ix1, iy2 - iy1]
                        cropped_anns.append(new_ann)
            else:
                cropped_anns = scaled_anns
        else:
            cropped_anns = scaled_anns

        # 3. Adaptive Tiling & Resizing to 640x640
        med_px = median_symbol_size(cropped_anns)
        target_size = PIPELINE_CONFIG["MODEL_INPUT_SIZE"]
        target_symbol = PIPELINE_CONFIG["TARGET_SYMBOL_PX"]

        if PIPELINE_CONFIG["ENABLE_ADAPTIVE_TILING"] and med_px > 0:
            g_x = max(1, round(target_symbol * im.width / (target_size * med_px)))
            g_y = max(1, round(target_symbol * im.height / (target_size * med_px)))
        else:
            g_x, g_y = 1, 1

        overlap = PIPELINE_CONFIG["TILE_OVERLAP"]
        tile_w = im.width / (g_x - (g_x - 1) * overlap) if g_x > 1 else im.width
        tile_h = im.height / (g_y - (g_y - 1) * overlap) if g_y > 1 else im.height
        stride_x = tile_w * (1 - overlap) if g_x > 1 else im.width
        stride_y = tile_h * (1 - overlap) if g_y > 1 else im.height

        # Build tile positions with edge-anchored last tile (Fix 2.3)
        positions_x = [c * stride_x for c in range(g_x)]
        if g_x > 1:
            positions_x[-1] = max(0, im.width - tile_w)  # anchor last tile to right edge
        positions_y = [r * stride_y for r in range(g_y)]
        if g_y > 1:
            positions_y[-1] = max(0, im.height - tile_h)  # anchor last tile to bottom edge

        tiles = [
            (px, py, min(im.width, px + tile_w), min(im.height, py + tile_h))
            for py in positions_y for px in positions_x
        ]

        for tx1_f, ty1_f, tx2_f, ty2_f in tiles:
            # Issue 3: Round tile coordinates ONCE; use the same integers
            # for clipping, cropping, and annotation translation.
            tx1, ty1 = round(tx1_f), round(ty1_f)
            tx2, ty2 = round(tx2_f), round(ty2_f)
            tx2 = min(tx2, im.width)
            ty2 = min(ty2, im.height)
            if tx2 <= tx1 or ty2 <= ty1:
                n_tiles_discarded_degenerate += 1
                continue

            tile_anns = []
            for ann in cropped_anns:
                x, y, w, h = ann["bbox"]
                clipped = clip_box(x, y, x + w, y + h, tx1, ty1, tx2, ty2)
                if not clipped:
                    continue
                ix1, iy1, ix2, iy2, vis = clipped
                if vis >= PIPELINE_CONFIG["MIN_VISIBLE_AREA"]:
                    new_ann = copy.deepcopy(ann)
                    new_ann["bbox"] = [ix1 - tx1, iy1 - ty1, ix2 - ix1, iy2 - iy1]
                    tile_anns.append(new_ann)

            if PIPELINE_CONFIG["ENABLE_REMOVE_EMPTY_TILES"] and not tile_anns:
                n_tiles_discarded_empty += 1
                continue

            tile_img = im.crop((tx1, ty1, tx2, ty2))
            tw, th = tile_img.size
            if tw <= 0 or th <= 0:
                n_tiles_discarded_degenerate += 1
                continue

            # Resize tile to uniform model resolution (640x640)
            tile_img = tile_img.resize((target_size, target_size), Image.Resampling.BILINEAR)
            sx, sy = target_size / tw, target_size / th

            tile_fname = f"img{iid}_tile{n_tiles_total:05d}.png"
            if tile_fname in saved_filenames:
                raise RuntimeError(f"Duplicate tile filename: {tile_fname}")
            tile_out_path = os.path.join(dst_dir, tile_fname)
            save_image_smart(tile_img, tile_out_path)

            if not os.path.isfile(tile_out_path):
                raise IOError(f"Tile save failed — file not found after write: {tile_out_path}")
            try:
                with Image.open(tile_out_path) as _vimg:
                    _vimg.verify()
            except Exception as e:
                os.remove(tile_out_path)
                raise RuntimeError(f"Corrupt tile image {tile_fname}: {e}")
            with Image.open(tile_out_path) as _cimg:
                if _cimg.width != target_size or _cimg.height != target_size:
                    raise RuntimeError(
                        f"Tile {tile_fname} saved as {_cimg.width}x{_cimg.height}, "
                        f"expected {target_size}x{target_size}")
            saved_filenames.add(tile_fname)

            new_images.append({
                "id": next_img_id,
                "file_name": tile_fname,
                "width": target_size,
                "height": target_size
            })

            for ann in tile_anns:
                x, y, w, h = ann["bbox"]
                bx, by, bw, bh = x * sx, y * sy, w * sx, h * sy
                # Clamp both corners independently to [0, target_size]
                bx1 = max(0.0, min(bx, float(target_size)))
                by1 = max(0.0, min(by, float(target_size)))
                bx2 = max(0.0, min(bx + bw, float(target_size)))
                by2 = max(0.0, min(by + bh, float(target_size)))
                bw = bx2 - bx1
                bh = by2 - by1
                if bw <= 0 or bh <= 0:
                    n_invalid_boxes += 1
                    continue
                bx, by = bx1, by1
                area = bw * bh
                if not all(math.isfinite(v) for v in (bx, by, bw, bh, area)):
                    n_invalid_boxes += 1
                    continue
                if ann["category_id"] not in valid_cat_ids:
                    n_invalid_boxes += 1
                    continue
                new_annotations.append({
                    "id": next_ann_id,
                    "image_id": next_img_id,
                    "category_id": ann["category_id"],
                    "bbox": [bx, by, bw, bh],
                    "area": area,
                    "iscrowd": 0
                })
                next_ann_id += 1

            next_img_id += 1
            n_tiles_total += 1

        n_images_processed += 1

    # ── Pre-save integrity validation ───────────────────────────────────
    json_filenames = {img["file_name"] for img in new_images}
    disk_files = {f for f in os.listdir(dst_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg"))}
    missing_on_disk = json_filenames - disk_files
    extra_on_disk = disk_files - json_filenames

    print(f"[{split}] Images in JSON: {len(json_filenames)}, "
          f"image files on disk: {len(disk_files)}")
    if missing_on_disk:
        sample = sorted(missing_on_disk)[:20]
        raise ValueError(
            f"[{split}] COCO/disk filename mismatch — "
            f"{len(missing_on_disk)} image(s) referenced in JSON "
            f"but not found on disk.\n"
            f"First missing files:\n"
            + "\n".join(f"  Missing image: {f}" for f in sample)
        )
    if extra_on_disk:
        sample = sorted(extra_on_disk)[:20]
        raise ValueError(
            f"[{split}] COCO/disk filename mismatch — "
            f"{len(extra_on_disk)} stale image file(s) on disk "
            f"not referenced in JSON.\n"
            f"First extra files:\n"
            + "\n".join(f"  Extra file: {f}" for f in sample)
        )

    _img_ids = [img["id"] for img in new_images]
    if len(_img_ids) != len(set(_img_ids)):
        raise ValueError(f"[{split}] Duplicate image IDs detected in generated COCO")
    _ann_ids = [a["id"] for a in new_annotations]
    if len(_ann_ids) != len(set(_ann_ids)):
        raise ValueError(f"[{split}] Duplicate annotation IDs detected in generated COCO")
    _fnames = [img["file_name"] for img in new_images]
    if len(_fnames) != len(set(_fnames)):
        raise ValueError(f"[{split}] Duplicate filenames detected in generated COCO")

    image_id_set = set(_img_ids)
    image_dims = {img["id"]: (img["width"], img["height"])
                  for img in new_images}
    image_fname = {img["id"]: img["file_name"] for img in new_images}
    integrity_errors = []

    for img in new_images:
        if img["width"] <= 0 or img["height"] <= 0:
            integrity_errors.append(
                f"Image: {img['file_name']}  id: {img['id']}  "
                f"Reason: non-positive dimensions {img['width']}x{img['height']}")

    for ann in new_annotations:
        fn = image_fname.get(ann["image_id"], "?")
        x, y, w, h = ann["bbox"]
        area = ann.get("area", w * h)
        if ann["image_id"] not in image_id_set:
            integrity_errors.append(
                f"Image: {fn}  ann_id: {ann['id']}  "
                f"Reason: references non-existent image_id {ann['image_id']}")
        if ann["category_id"] not in valid_cat_ids:
            integrity_errors.append(
                f"Image: {fn}  ann_id: {ann['id']}  "
                f"category: {ann['category_id']}  "
                f"Reason: invalid category_id")
        if any(not math.isfinite(v) for v in (x, y, w, h, area)):
            integrity_errors.append(
                f"Image: {fn}  ann_id: {ann['id']}  "
                f"BBox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]  area: {area:.1f}  "
                f"Reason: non-finite value")
        elif w <= 0 or h <= 0 or area <= 0:
            integrity_errors.append(
                f"Image: {fn}  ann_id: {ann['id']}  "
                f"BBox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]  area: {area:.1f}  "
                f"Reason: non-positive dimensions or area")
        else:
            img_w, img_h = image_dims.get(ann["image_id"], (None, None))
            if img_w is not None and (
                    x < -0.5 or y < -0.5
                    or x + w > img_w + 0.5
                    or y + h > img_h + 0.5):
                integrity_errors.append(
                    f"Image: {fn}  ann_id: {ann['id']}  "
                    f"category: {ann['category_id']}  "
                    f"BBox: [{x:.1f}, {y:.1f}, {w:.1f}, {h:.1f}]  "
                    f"Image size: {img_w}x{img_h}  "
                    f"Reason: bbox outside image boundaries")

    if integrity_errors:
        print(f"❌ [{split}] COCO integrity check found "
              f"{len(integrity_errors)} error(s):")
        for e in integrity_errors[:20]:
            print(f"  {e}")
        if len(integrity_errors) > 20:
            print(f"  ... and {len(integrity_errors) - 20} more")
        raise ValueError(
            f"[{split}] COCO integrity validation failed with "
            f"{len(integrity_errors)} error(s). "
            f"Refusing to write invalid COCO file.")

    out_coco = {
        "images": new_images,
        "annotations": new_annotations,
        "categories": coco["categories"]
    }
    out_path = os.path.join(dst_dir, "_annotations.coco.json")
    with open(out_path, "w") as f:
        json.dump(out_coco, f)

    cat_names = [c["name"] for c in coco["categories"]]
    print(f"[{split}] Preprocessing Summary:")
    print(f"  Images processed         : {n_images_processed}")
    print(f"  Tiles generated          : {n_tiles_total}")
    print(f"  Tiles discarded (empty)  : {n_tiles_discarded_empty}")
    print(f"  Tiles discarded (degen)  : {n_tiles_discarded_degenerate}")
    print(f"  Annotations written      : {len(new_annotations)}")
    print(f"  Invalid boxes removed    : {n_invalid_boxes}")
    print(f"  Categories ({len(cat_names)})          : {cat_names}")
    print(f"  Scale-Norm               : {scale_stats['with_ref']} ref, "
          f"{scale_stats['fallback']} fallback (avg {scale_stats['avg']:.3f}x)")
    print(f"  Validation               : PASSED")
    return out_path


def save_tile_qa_samples(split, dst_dir, out_coco, n_samples=12, seed=42):
    """Validation layer for tiling: render a random sample of generated tiles with
    their bounding boxes drawn on top, so annotation alignment can be visually
    confirmed before spending GPU time training on them. Prefers tiles that actually
    have annotations, so the samples are useful rather than mostly-empty."""
    if not out_coco["images"]:
        return
    qa_dir = os.path.join(os.path.dirname(os.path.normpath(dst_dir)), "qa_samples", split)
    os.makedirs(qa_dir, exist_ok=True)

    cat_names = {c["id"]: c["name"] for c in out_coco["categories"]}
    anns_by_img = defaultdict(list)
    for a in out_coco["annotations"]:
        anns_by_img[a["image_id"]].append(a)

    candidates = [img for img in out_coco["images"] if anns_by_img.get(img["id"])]
    if not candidates:
        candidates = out_coco["images"]
    sample = random.Random(seed).sample(candidates, min(n_samples, len(candidates)))

    for img in sample:
        src_path = os.path.join(dst_dir, img["file_name"])
        if not os.path.exists(src_path):
            continue
        im = Image.open(src_path).convert("RGB")
        draw = ImageDraw.Draw(im)
        for ann in anns_by_img.get(img["id"], []):
            x, y, w, h = ann["bbox"]
            draw.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=2)
            draw.text((x, max(0, y - 12)), cat_names.get(ann["category_id"], "?"), fill=(255, 0, 0))
        im.save(os.path.join(qa_dir, f"qa_{img['file_name']}"))

    print(f"[{split}] Saved {len(sample)} annotated QA sample tiles to {qa_dir} — "
          f"spot-check a few before training to confirm boxes land on the right symbols.")


PROCESSED_ANNO = {}
for split in ["train", "valid", "test"]:
    PROCESSED_ANNO[split] = preprocess_split(split, RAW_DIR, PROCESSED_DIR)
    validate_or_die(PROCESSED_ANNO[split], os.path.join(PROCESSED_DIR, split), split, "Post-Preprocessing Check")
    if PIPELINE_CONFIG.get("ENABLE_TILE_QA_SAMPLES", True):
        with open(PROCESSED_ANNO[split]) as f:
            _qa_coco = json.load(f)
        save_tile_qa_samples(split, os.path.join(PROCESSED_DIR, split), _qa_coco,
                              n_samples=PIPELINE_CONFIG.get("QA_SAMPLES_PER_SPLIT", 12))

print("\n✅ Preprocessing & post-validation complete across all splits\n")

# ══════════════════════════════════════════════════════════════
# STEP 12 — Detect Classes & Integrity Check
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 12 — Detect Classes & Integrity Check")
print("=" * 60)

# Compute categories from union of ALL processed splits
all_processed_categories = {}
for split in ["train", "valid", "test"]:
    with open(PROCESSED_ANNO[split]) as f:
        c = json.load(f)
    for cat in c["categories"]:
        all_processed_categories[cat["id"]] = cat

# Load processed train annotations for per-class instance counting
with open(PROCESSED_ANNO["train"]) as f:
    coco_train = json.load(f)

# Count per-class instances in TRAIN specifically
train_instance_counts = defaultdict(int)
for ann in coco_train["annotations"]:
    train_instance_counts[ann["category_id"]] += 1

# Cross-validate: every raw category must have ≥1 training instance
raw_categories = {}
for split in ["train", "valid", "test"]:
    raw_anno = os.path.join(RAW_DIR, split, "_annotations.coco.json")
    if os.path.exists(raw_anno):
        with open(raw_anno) as f:
            c = json.load(f)
        for cat in c.get("categories", []):
            raw_categories[cat["id"]] = cat

missing_classes = []
for cid, cat in sorted(raw_categories.items(), key=lambda x: x[0]):
    if train_instance_counts.get(cid, 0) == 0:
        missing_classes.append(cat["name"])

if missing_classes:
    raise ValueError(
        f"🚫 Category integrity check FAILED. The following {len(missing_classes)} classes "
        f"from the raw dataset have ZERO training instances after preprocessing: {missing_classes}\n"
        f"This likely means preprocessing (crop/tile) deleted all instances of these classes."
    )

# Use the full category set (union of all splits) for training
cats = sorted(all_processed_categories.values(), key=lambda c: c["id"])
CLASS_NAMES = ["__background__"] + [c["name"] for c in cats]
NUM_CLASSES = len(cats)  # D-FINE num_classes = real classes only (no background)

# ── Gate B: authoritative post-tiling contiguity check ──────────────────
# cats/NUM_CLASSES above is the set D-FINE's classification head will
# actually be sized for THIS run. Verify it is genuinely contiguous 0..N-1
# (catches e.g. a registered class that has zero instances in this
# particular raw pull — same failure mode as the placeholder-class bug,
# just a different cause) and that every processed split's annotations
# only reference labels in that range, before spending any more time on
# balancing/checkpointing/training.
for _split_name, _split_path in PROCESSED_ANNO.items():
    with open(_split_path) as _f:
        _split_anns = json.load(_f)["annotations"]
    validate_category_id_contiguity_or_die(cats, _split_anns, NUM_CLASSES, _split_name, "Post-Tiling Union Check")

print(f"Detected {NUM_CLASSES} classes (all present in train split):")
print(f"{'ID':>5}  {'Class Name':<35}  {'Train Instances':>15}")
print(f"{'─'*5}  {'─'*35}  {'─'*15}")
for c in cats:
    count = train_instance_counts.get(c["id"], 0)
    marker = "  ⚠️ LOW" if count < 10 else ""
    print(f"  {c['id']:>3}  {c['name']:<35}  {count:>15,}{marker}")

assert len(CLASS_NAMES) == len(set(CLASS_NAMES)), "Duplicate class names detected!"
print()

# ══════════════════════════════════════════════════════════════
# STEP 13 — Class Balancing & Minority Augmentation
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 13 — Image-Level Class Balancing + Minority Augmentation")
print("=" * 60)

def compute_repeat_factors(coco, repeat_thresh=None, max_repeat=8.0, density_weight=0.5):
    total_images = len(coco["images"])
    images_per_cat, inst_per_cat_img = defaultdict(set), defaultdict(int)
    for ann in coco["annotations"]:
        images_per_cat[ann["category_id"]].add(ann["image_id"])
        inst_per_cat_img[(ann["category_id"], ann["image_id"])] += 1

    cat_freq = {cid: len(imgs) / total_images for cid, imgs in images_per_cat.items()}
    if repeat_thresh is None:
        freqs = sorted(cat_freq.values())
        repeat_thresh = freqs[len(freqs) // 2] if freqs else 0.5

    cat_repeat = {cid: max(1.0, (repeat_thresh / f) ** 0.5) if f > 0 else 1.0 for cid, f in cat_freq.items()}
    cat_avg_density = {
        cid: sum(inst_per_cat_img[(cid, i)] for i in imgs) / max(1, len(imgs))
        for cid, imgs in images_per_cat.items()
    }

    image_repeat = {}
    for img in coco["images"]:
        iid = img["id"]
        cats_here = [cid for cid in images_per_cat if (cid, iid) in inst_per_cat_img]
        if not cats_here:
            image_repeat[iid] = 1.0
            continue
        base = max(cat_repeat[cid] for cid in cats_here)
        bonus = 1.0
        for cid in cats_here:
            avg = cat_avg_density.get(cid, 1.0)
            if avg > 0:
                ratio = inst_per_cat_img[(cid, iid)] / avg
                bonus = max(bonus, 1.0 + density_weight * max(0.0, ratio - 1.0))
        image_repeat[iid] = min(max_repeat, base * bonus)
    return image_repeat, cat_freq, repeat_thresh


def build_balanced_coco(coco, image_repeat, img_dir, seed=0):
    rng = random.Random(seed)
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    new_images, new_annotations = [], []
    next_iid = max(img["id"] for img in coco["images"]) + 1
    next_aid = max((a["id"] for a in coco["annotations"]), default=-1) + 1
    stats = {"original": len(coco["images"]), "added": 0, "augmented": 0}

    for img in coco["images"]:
        factor = image_repeat[img["id"]]
        n_copies = max(1, int(factor) + (1 if rng.random() < (factor - int(factor)) else 0))

        new_images.append(copy.deepcopy(img))
        new_annotations.extend(copy.deepcopy(anns_by_image[img["id"]]))

        for copy_idx in range(n_copies - 1):
            dup_img = copy.deepcopy(img)
            dup_img["id"] = next_iid

            # Apply brightness jitter to duplicate — NO flip (Fix 2.5)
            # Flipping mirrors directional SLD symbols (diodes, arrows, polarity, ground)
            src_path = os.path.join(img_dir, dup_img["file_name"])
            base, ext = os.path.splitext(dup_img["file_name"])
            aug_fname = f"{base}_aug{copy_idx}{ext}"
            aug_path = os.path.join(img_dir, aug_fname)

            if os.path.exists(src_path) and not os.path.exists(aug_path):
                try:
                    pil_img = Image.open(src_path).convert("RGB")
                    arr = np.array(pil_img, dtype=np.float32)
                    brightness = 1.0 + rng.uniform(-0.15, 0.15)
                    arr = np.clip(arr * brightness, 0, 255).astype(np.uint8)
                    save_image_smart(arr, aug_path)
                    dup_img["file_name"] = aug_fname
                    stats["augmented"] += 1
                except Exception:
                    pass

            new_images.append(dup_img)
            for ann in anns_by_image[img["id"]]:
                dup_ann = copy.deepcopy(ann)
                dup_ann["id"] = next_aid
                dup_ann["image_id"] = next_iid
                new_annotations.append(dup_ann)
                next_aid += 1

            next_iid += 1
            stats["added"] += 1

    out = copy.deepcopy(coco)
    out["images"], out["annotations"] = new_images, new_annotations
    stats["final"] = len(new_images)
    return out, stats


image_repeat, cat_freq, repeat_thresh = compute_repeat_factors(coco_train)
train_img_dir = os.path.join(PROCESSED_DIR, "train")
balanced_coco, balance_stats = build_balanced_coco(coco_train, image_repeat, train_img_dir, seed=0)

ANNO_TRAIN_BALANCED = os.path.join(PROCESSED_DIR, "train", "_annotations_balanced.coco.json")
with open(ANNO_TRAIN_BALANCED, "w") as f:
    json.dump(balanced_coco, f)

print(f"Original Tiles     : {balance_stats['original']}")
print(f"Duplicates Added   : {balance_stats['added']} (Augmented: {balance_stats['augmented']})")
print(f"Final Balanced Set : {balance_stats['final']}")
print(f"✅ Balanced annotations written to {ANNO_TRAIN_BALANCED}")

print("Processed tiled dataset stored locally (OneDrive sync disabled for processed data).")
print(f"✅ Tiled dataset ready at local path: '{PROCESSED_DIR}'\n")

# ══════════════════════════════════════════════════════════════
# STEP 14 — Download Checkpoint
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 14 — Download Pretrained Checkpoint")
print("=" * 60)

CHECKPOINT_URL = "https://github.com/Peterande/storage/releases/download/dfinev1.0/dfine_x_obj2coco.pth"
CHECKPOINT_CONFIG = f"{DFINE_REPO}/configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml"
CHECKPOINT_PATH = f"{WORKSPACE}/dfine_x_obj2coco.pth"

if not os.path.exists(CHECKPOINT_PATH) or os.path.getsize(CHECKPOINT_PATH) < 10_000_000:
    print(f"Downloading {CHECKPOINT_URL} ...")
    urllib.request.urlretrieve(CHECKPOINT_URL, CHECKPOINT_PATH)
else:
    print("✅ Checkpoint present on disk")

size_mb = os.path.getsize(CHECKPOINT_PATH) / 1e6
print(f"Checkpoint size: {size_mb:.1f} MB")
assert size_mb > 100, "Checkpoint too small — download failed."

state = torch.load(CHECKPOINT_PATH, map_location="cpu")
key = "ema" if "ema" in state else "model"
print(f"✅ Checkpoint verified — {len(state[key])} tensors in '{key}' dict")
del state
print()

# ══════════════════════════════════════════════════════════════
# STEP 15 — Batch Size & Learning Rate Calculation
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 15 — Compute Batch Size & LR")
print("=" * 60)
import yaml

TARGET_TOTAL_BATCH = 32
BASE_LR_AT_BATCH32 = 1e-4 # matches D-FINE's own official custom/obj2coco fine-tune recipe
# (configs/dfine/objects365/dfine_hgnetv2_x_obj2coco.yml and custom/dfine_hgnetv2_x_custom.yml
# both use this exact value as the main LR for fine-tuning, not a lowered one — the "gentle
# fine-tune" effect there comes from the shorter schedule and the 1%-ratio backbone LR below,
# not from shrinking the main LR itself. Previous 5e-5 was ~5x more conservative than validated.
BASE_WD = 5e-4
BACKBONE_LR_RATIO = 0.0000025 / 0.00025

def estimate_safe_batch_per_gpu(vram_gb):
    usable = max(0.0, vram_gb - 2.0)
    per_sample_gb = (16.0 - 2.0) / 4
    return max(1, int(usable / per_sample_gb))

safe_per_gpu = estimate_safe_batch_per_gpu(MIN_VRAM_GB)
max_safe_total = safe_per_gpu * NUM_GPUS
TOTAL_BATCH_SIZE = min(TARGET_TOTAL_BATCH, max_safe_total)

SCALED_LR = BASE_LR_AT_BATCH32 * (TOTAL_BATCH_SIZE / 32)
SCALED_WD = BASE_WD

print(f"TOTAL_BATCH_SIZE : {TOTAL_BATCH_SIZE} (Target was {TARGET_TOTAL_BATCH})")
print(f"Main LR          : {SCALED_LR:.2e}")
print(f"Backbone LR      : {SCALED_LR * BACKBONE_LR_RATIO:.2e}")
print(f"Weight Decay     : {SCALED_WD:.2e}\n")

EPOCHS = 60
STOP_EPOCH = 50

# Fix 2.5 — SLD symbols are directionally meaningful (diodes, arrows, polarity,
# ground marks), so RandomHorizontalFlip is dropped from the augmentation
# pipeline. The remaining ops are copied verbatim (same types/params/order)
# from D-FINE's configs/dfine/include/dataloader.yml so nothing else changes.
TRAIN_TRANSFORM_OPS = [
    {"type": "RandomPhotometricDistort", "p": 0.5},
    {"type": "RandomZoomOut", "fill": 0},
    {"type": "RandomIoUCrop", "p": 0.8},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    # {"type": "RandomHorizontalFlip"},  # deliberately REMOVED for SLD (Fix 2.5)
    {"type": "Resize", "size": [640, 640]},
    {"type": "SanitizeBoundingBoxes", "min_size": 1},
    {"type": "ConvertPILImage", "dtype": "float32", "scale": True},
    {"type": "ConvertBoxes", "fmt": "cxcywh", "normalize": True},  # RESTORED — was silently dropped;
    # without this, boxes stay in raw 0-640 pixel space instead of normalized [0,1],
    # which is why loss_bbox/loss_giou never converged and mAP stayed at 0.0
]

custom_cfg = {
    "__include__": [CHECKPOINT_CONFIG],
    "optimizer": {
        "type": "AdamW",
        "params": [
            {"params": "^(?=.*backbone)(?!.*norm|bn).*$", "lr": SCALED_LR * BACKBONE_LR_RATIO},
            {"params": "^(?=.*(?:encoder|decoder))(?=.*(?:norm|bn)).*$", "weight_decay": 0.0},
        ],
        "lr": SCALED_LR,
        "betas": [0.9, 0.999],
        "weight_decay": SCALED_WD,
    },
    "epochs": EPOCHS,
    # Only keep the best checkpoints, not one file per epoch. D-FINE's solver saves a
    # distinct checkpoint{epoch:04}.pth every `checkpoint_freq` epochs (default: 1, i.e.
    # every epoch, accumulating one full checkpoint per epoch for the whole run). Setting
    # this above EPOCHS means that numbered dump never triggers — last.pth (overwritten
    # each epoch) and best_stg1.pth/best_stg2.pth (saved only on improvement) are all
    # that get written or synced to OneDrive.
    "checkpoint_freq": EPOCHS + 1,
    "train_dataloader": {
        "total_batch_size": TOTAL_BATCH_SIZE,
        "dataset": {
            "transforms": {
                "policy": {"epoch": STOP_EPOCH},
                "ops": TRAIN_TRANSFORM_OPS,
            }
        },
        "collate_fn": {"stop_epoch": STOP_EPOCH, "base_size_repeat": 3},
    },
}

CUSTOM_CONFIG_PATH = f"{DFINE_REPO}/configs/dfine/custom_sld_finetune.yml"
with open(CUSTOM_CONFIG_PATH, "w") as f:
    yaml.dump(custom_cfg, f, sort_keys=False, default_flow_style=False)
print(f"✅ Custom config written to {CUSTOM_CONFIG_PATH}")
print(f"✅ RandomHorizontalFlip removed from train_dataloader.dataset.transforms.ops (Fix 2.5)\n")

# ══════════════════════════════════════════════════════════════
# STEP 16 — Verify D-FINE Imports
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 16 — Verify D-FINE Imports")
print("=" * 60)

if DFINE_REPO not in sys.path:
    sys.path.insert(0, DFINE_REPO)
os.chdir(DFINE_REPO)

from src.core import YAMLConfig
import src.misc.dist_utils as dist

print(f"✅ Imports OK | PyTorch {torch.__version__} | CUDA {torch.version.cuda}\n")

# ══════════════════════════════════════════════════════════════
# STEP 17-18 — Background Graph Sync & Early Stop Monitor (log.txt)
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 17-18 — Start Background Monitoring & Graph Sync Threads")
print("=" * 60)

CHECKPOINT_SYNC_INTERVAL_SEC = 300  # 5 minutes / 300 seconds
EARLY_STOP_PATIENCE = 20
TOP_K_CHECKPOINTS = 5
MAP_50_95_INDEX = 0

def checkpoint_sync_loop():
    while True:
        time.sleep(CHECKPOINT_SYNC_INTERVAL_SEC)
        try:
            graph_uploader.upload_folder_recursive(OUTPUT_DIR, CLOUD_OUTPUT_REMOTE)
        except Exception as e:
            print(f"⚠️  Background Graph sync warning: {e}")


def read_map_history(log_path):
    history = []
    if not os.path.exists(log_path):
        return history
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "test_coco_eval_bbox" in row:
                    history.append((row["epoch"], row["test_coco_eval_bbox"][MAP_50_95_INDEX]))
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
    return history


def monitor_loop():
    log_path = os.path.join(OUTPUT_DIR, "log.txt")
    ranked = {}
    while True:
        time.sleep(300)
        history = read_map_history(log_path)
        if not history:
            continue
        for epoch, m in history:
            ranked[epoch] = m

        best_epoch, best_map = max(ranked.items(), key=lambda kv: kv[1])
        latest_epoch = history[-1][0]
        epochs_since_best = latest_epoch - best_epoch

        if epochs_since_best >= EARLY_STOP_PATIENCE:
            sentinel = os.path.join(OUTPUT_DIR, "STOP_REQUESTED")
            if not os.path.exists(sentinel):
                with open(sentinel, "w") as f:
                    f.write(f"mAP50:95 plateaued at {best_map:.4f} (epoch {best_epoch}); {epochs_since_best} epochs without improvement\n")
                print(f"\n🔔 Early stop requested: best mAP50:95={best_map:.4f} at epoch {best_epoch}. Stopping before next epoch.")

        top_k = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:TOP_K_CHECKPOINTS]
        top_dir = os.path.join(OUTPUT_DIR, f"top{TOP_K_CHECKPOINTS}")
        os.makedirs(top_dir, exist_ok=True)
        wanted = {f"epoch{ep:04d}_map{score:.4f}.pth" for ep, score in top_k}
        for existing in os.listdir(top_dir):
            if existing not in wanted:
                try:
                    os.remove(os.path.join(top_dir, existing))
                except OSError:
                    pass
        for ep, score in top_k:
            fname = f"epoch{ep:04d}_map{score:.4f}.pth"
            dst = os.path.join(top_dir, fname)
            if not os.path.exists(dst) and ep == latest_epoch:
                src = os.path.join(OUTPUT_DIR, "last.pth")
                if os.path.exists(src):
                    shutil.copy2(src, dst)


threading.Thread(target=checkpoint_sync_loop, daemon=True).start()
threading.Thread(target=monitor_loop, daemon=True).start()
print(f"🔄 Background Graph sync active (every 300 seconds)")
print(f"👀 Log monitor active (log.txt JSONL, patience={EARLY_STOP_PATIENCE}, top-{TOP_K_CHECKPOINTS} checkpoints)\n")

# ══════════════════════════════════════════════════════════════
# Fix 2.1 — Checkpoint Class-Count Compatibility Guard
# ══════════════════════════════════════════════════════════════
# State dict keys (verified against the D-FINE repo): the classification head
# lives at decoder.dec_score_head, with keys following the pattern
#   decoder.dec_score_head.{layer_idx}.layers.{sublayer}.weight / .bias
# The output dim of the LAST Linear layer in each dec_score_head sub-module
# equals num_classes.
DEC_SCORE_HEAD_PATTERN = re.compile(r"decoder\.dec_score_head\.(\d+)\.layers\.(\d+)\.weight$")


def _resolve_state_dict(ckpt):
    """Best-effort extraction of a flat {param_name: tensor} dict from a D-FINE checkpoint."""
    if not isinstance(ckpt, dict):
        return None
    for candidate_key in ("ema", "model"):
        candidate = ckpt.get(candidate_key)
        if isinstance(candidate, dict):
            # D-FINE's EMA wrapper nests the real weights under "module"
            inner = candidate.get("module")
            if isinstance(inner, dict):
                return inner
            return candidate
    # Fall back: maybe ckpt itself IS the flat state dict
    if ckpt and all(hasattr(v, "shape") for v in ckpt.values()):
        return ckpt
    return None


def check_checkpoint_class_compatibility(checkpoint_path, expected_num_classes):
    """
    Inspects a D-FINE checkpoint's decoder.dec_score_head weights to determine
    whether it was trained with `expected_num_classes` output classes.

    Safe default: if no 'decoder.dec_score_head' keys are found at all, this
    returns False (INCOMPATIBLE) rather than assuming the checkpoint is fine —
    an unreadable or unexpected checkpoint layout should never silently resume.
    """
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu")
    except Exception as e:
        print(f"⚠️  Could not load '{checkpoint_path}' for compatibility check ({e}). Defaulting to INCOMPATIBLE.")
        return False

    state_dict = _resolve_state_dict(ckpt)
    if not state_dict:
        print(f"⚠️  Could not locate a usable state dict in '{checkpoint_path}'. Defaulting to INCOMPATIBLE.")
        return False

    per_head_last_layer = {}  # head_idx -> (sublayer_idx, weight_tensor)
    for name, tensor in state_dict.items():
        m = DEC_SCORE_HEAD_PATTERN.search(name)
        if not m:
            continue
        head_idx, sub_idx = int(m.group(1)), int(m.group(2))
        if head_idx not in per_head_last_layer or sub_idx > per_head_last_layer[head_idx][0]:
            per_head_last_layer[head_idx] = (sub_idx, tensor)

    if not per_head_last_layer:
        print(f"⚠️  No 'decoder.dec_score_head' keys found in '{checkpoint_path}'. Defaulting to INCOMPATIBLE (safe default).")
        return False

    mismatches = [
        (head_idx, weight.shape[0])
        for head_idx, (_, weight) in sorted(per_head_last_layer.items())
        if weight.shape[0] != expected_num_classes
    ]
    if mismatches:
        detail = ", ".join(f"head {h}: {d} classes" for h, d in mismatches)
        print(f"⚠️  Checkpoint class-count MISMATCH vs expected {expected_num_classes}: {detail}")
        return False

    print(f"✅ Checkpoint dec_score_head matches expected {expected_num_classes} classes "
          f"across {len(per_head_last_layer)} head(s).")
    return True


# ══════════════════════════════════════════════════════════════
# STEP 19 — Launch Training
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 19 — Launch Training")
print("=" * 60)

last_checkpoint = os.path.join(OUTPUT_DIR, "last.pth")
resuming = os.path.exists(last_checkpoint)

if resuming:
    print(f"Found existing 'last.pth' in {OUTPUT_DIR} — checking class-count compatibility before resuming...")
    if not check_checkpoint_class_compatibility(last_checkpoint, NUM_CLASSES):
        print(f"🚫 'last.pth' is incompatible with the current {NUM_CLASSES}-class set (or compatibility "
              f"could not be verified). Forcing a FRESH FINE-TUNE from the Objects365+COCO checkpoint "
              f"instead of resuming (Fix 2.1).")
        resuming = False

stop_sentinel = os.path.join(OUTPUT_DIR, "STOP_REQUESTED")
if os.path.exists(stop_sentinel):
    os.remove(stop_sentinel)

launcher = [sys.executable] if NUM_GPUS == 1 else ["torchrun", "--master_port=7777", f"--nproc_per_node={NUM_GPUS}"]

train_cmd = launcher + [
    f"{DFINE_REPO}/train.py",
    "-c", CUSTOM_CONFIG_PATH,
    "--output-dir", OUTPUT_DIR,
    "--use-amp",
    "--seed=0",
]
train_cmd += ["-r", last_checkpoint] if resuming else ["-t", CHECKPOINT_PATH]
train_cmd += [
    "-u",
    f"train_dataloader.dataset.img_folder={os.path.join(PROCESSED_DIR, 'train')}",
    f"train_dataloader.dataset.ann_file={ANNO_TRAIN_BALANCED}",
    f"val_dataloader.dataset.img_folder={os.path.join(PROCESSED_DIR, 'valid')}",
    f"val_dataloader.dataset.ann_file={PROCESSED_ANNO['valid']}",
    f"num_classes={NUM_CLASSES}",
    "remap_mscoco_category=False",
]

print("Mode:", "RESUME" if resuming else "FRESH FINE-TUNE")
print("Command:", " ".join(train_cmd))
print()

# ── Gate C: final pre-launch check on the EXACT files about to be handed
# to D-FINE (train_dataloader.dataset.ann_file / val_dataloader...ann_file
# above). This is the last line of defense — independent of every check
# already run earlier in the pipeline — against launching a GPU job with a
# category_id that will crash CUDA. Cheap (two small JSON files) relative
# to the hours of compute it protects.
for _gate_name, _gate_path in [("train_balanced", ANNO_TRAIN_BALANCED), ("valid", PROCESSED_ANNO["valid"])]:
    with open(_gate_path) as _f:
        _gate_coco = json.load(_f)
    validate_category_id_contiguity_or_die(
        _gate_coco["categories"], _gate_coco["annotations"], NUM_CLASSES, _gate_name, "Pre-Launch Final Gate"
    )

process = subprocess.Popen(train_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
for line in process.stdout:
    print(line, end="")
process.wait()
print(f"\nTraining process exited with code {process.returncode}\n")

# ══════════════════════════════════════════════════════════════
# STEP 20 — View Metrics
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 20 — Training Metrics")
print("=" * 60)

log_path = os.path.join(OUTPUT_DIR, "log.txt")
history = read_map_history(log_path)
if history:
    print(f"Epochs logged: {len(history)}")
    for ep, m in history[-15:]:
        print(f"  epoch {ep:>4}  mAP50:95={m:.4f}")
    best_ep, best_m = max(history, key=lambda x: x[1])
    print(f"\nBest epoch: epoch {best_ep}, mAP50:95={best_m:.4f}")
else:
    print("No log.txt entries found yet.")
print()

# ══════════════════════════════════════════════════════════════
# STEP 21 — Save Run Config & Final Microsoft Graph Sync
# ══════════════════════════════════════════════════════════════
print("=" * 60)
print("STEP 21 — Save Config & Final Microsoft Graph Sync")
print("=" * 60)

save_cfg = {
    "model": "dfine-x",
    "run_name": RUN_NAME,
    "pretrained_from": "Objects365+COCO (dfine_x_obj2coco.pth)",
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "total_batch_size": TOTAL_BATCH_SIZE,
    "num_gpus": NUM_GPUS,
    "main_lr": SCALED_LR,
    "backbone_lr": SCALED_LR * BACKBONE_LR_RATIO,
    "weight_decay": SCALED_WD,
    "epochs": EPOCHS,
    "stop_epoch": STOP_EPOCH,
    "checkpoint_freq": EPOCHS + 1,  # only best_stg1/best_stg2/last kept, no per-epoch dumps
    "authentication_method": "MSAL Client Credentials (Microsoft Graph API)",
    "target_user_upn": USER_UPN,
    "cloud_output_remote": CLOUD_OUTPUT_REMOTE,
    "cloud_processed_dataset": None,  # Saved locally only (OneDrive sync disabled)
    "pipeline_config": PIPELINE_CONFIG,
    "saved_at": datetime.now().isoformat(),
}

config_file = os.path.join(OUTPUT_DIR, f"training_config_{RUN_NAME}.json")
with open(config_file, "w") as f:
    json.dump(save_cfg, f, indent=2)

print(f"Config saved to {config_file}")
print("Final push to OneDrive via Microsoft Graph API...")
n_uploaded = graph_uploader.upload_folder_recursive(OUTPUT_DIR, CLOUD_OUTPUT_REMOTE)
print(f"✅ Final Graph upload completed successfully ({n_uploaded} files processed).")
print(f"✅ All completed. Outputs synced to {CLOUD_OUTPUT_REMOTE}")

# ══════════════════════════════════════════════════════════════
# FIX 2.8 — Full-Sheet Inference Utility (Tile + Predict + Stitch + NMS)
# ══════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Defining infer_full_sheet() — full-sheet tiled inference utility")
print("=" * 60)


def _xyxy_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _class_aware_nms(detections, iou_threshold=0.5):
    """detections: list of dicts with bbox=[x1,y1,x2,y2] (xyxy), score, category_id."""
    by_class = defaultdict(list)
    for d in detections:
        by_class[d["category_id"]].append(d)

    kept = []
    for dets in by_class.values():
        dets = sorted(dets, key=lambda d: d["score"], reverse=True)
        chosen = []
        for d in dets:
            if all(_xyxy_iou(d["bbox"], c["bbox"]) < iou_threshold for c in chosen):
                chosen.append(d)
        kept.extend(chosen)
    return kept


def infer_full_sheet(
    image_path,
    model_predict_fn,
    reference_priority=None,
    target_reference_height=None,
    dataset_avg_scale=1.0,
    model_input_size=None,
    target_symbol_px=None,
    tile_overlap=None,
    conf_threshold=0.25,
    nms_iou_threshold=0.5,
):
    """
    Full-sheet SLD inference: scale-normalize -> tile (edge-anchored, Fix 2.3) ->
    predict per tile -> map back to full-sheet coordinates -> cross-tile NMS.

    Per the Fix 2.4 note: the annotation-envelope safety-margin crop used during
    TRAINING preprocessing is intentionally NOT replicated here — at inference
    time there are no ground-truth boxes to build that envelope from. Instead
    the full sheet is tiled including any border/legend area; those tiles
    simply produce no detections, which is harmless.

    Args:
        image_path: path to the full-resolution SLD sheet image.
        model_predict_fn: callable(PIL.Image tile) -> list of dicts, each
            {"bbox": [x, y, w, h], "score": float, "category_id": int} in
            TILE-LOCAL pixel coordinates (model_input_size x model_input_size
            space). Wrap your loaded D-FINE checkpoint / ONNX session in a
            closure and pass it here — this function only handles scale
            normalization, tiling, coordinate mapping, and NMS.
        reference_priority: class names used for scale calibration (defaults
            to PIPELINE_CONFIG["REFERENCE_CLASS_PRIORITY"]).
        target_reference_height: target px height for the reference class
            (defaults to PIPELINE_CONFIG["TARGET_REFERENCE_HEIGHT"]).
        dataset_avg_scale: fallback scale factor when no reference-class
            instance is found in the calibration pass — pass the
            scale_stats['avg'] value recorded in training_config_*.json.
        conf_threshold: minimum detection score kept before NMS.
        nms_iou_threshold: IoU threshold for cross-tile duplicate suppression.

    Returns:
        list of dicts {"bbox": [x, y, w, h], "score": float, "category_id": int}
        in ORIGINAL full-sheet pixel coordinates.
    """
    reference_priority = reference_priority or PIPELINE_CONFIG["REFERENCE_CLASS_PRIORITY"]
    target_reference_height = target_reference_height or PIPELINE_CONFIG["TARGET_REFERENCE_HEIGHT"]
    model_input_size = model_input_size or PIPELINE_CONFIG["MODEL_INPUT_SIZE"]
    target_symbol_px = target_symbol_px or PIPELINE_CONFIG["TARGET_SYMBOL_PX"]
    tile_overlap = PIPELINE_CONFIG["TILE_OVERLAP"] if tile_overlap is None else tile_overlap

    orig_im = Image.open(image_path).convert("RGB")
    orig_w, orig_h = orig_im.size
    id_to_name = {c["id"]: c["name"] for c in cats}
    ref_name_set = set(reference_priority)

    # ---- Step A: quick calibration pass for scale normalization ----
    # A single coarse pass over the whole sheet (resized to model_input_size)
    # just to locate any reference-class instances and measure their height —
    # mirrors the training-time compute_scale_factors() logic, but since there
    # are no ground-truth annotations at inference, the model's own detections
    # stand in for them.
    calib_tile = orig_im.resize((model_input_size, model_input_size), Image.Resampling.BILINEAR)
    calib_sx, calib_sy = orig_w / model_input_size, orig_h / model_input_size
    ref_heights = [
        d["bbox"][3] * calib_sy
        for d in model_predict_fn(calib_tile)
        if d["score"] >= conf_threshold and id_to_name.get(d["category_id"]) in ref_name_set
    ]
    if ref_heights:
        measured_height = sum(ref_heights) / len(ref_heights)
        scale = target_reference_height / measured_height if measured_height > 0 else 1.0
    else:
        # No reference-class instance found in the calibration pass — fall
        # back to the dataset-average scale observed during training.
        scale = dataset_avg_scale

    new_w, new_h = max(32, round(orig_w * scale)), max(32, round(orig_h * scale))
    im = orig_im.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # ---- Step B: adaptive, edge-anchored tiling (mirrors Fix 2.3) ----
    # No ground-truth boxes exist yet, so target_symbol_px is used as the
    # assumed median symbol size — reasonable since the calibration step above
    # already aims every reference-class symbol at target_reference_height.
    med_px = target_symbol_px
    if PIPELINE_CONFIG["ENABLE_ADAPTIVE_TILING"] and med_px > 0:
        g_x = max(1, round(target_symbol_px * im.width / (model_input_size * med_px)))
        g_y = max(1, round(target_symbol_px * im.height / (model_input_size * med_px)))
    else:
        g_x, g_y = 1, 1

    tile_w = im.width / (g_x - (g_x - 1) * tile_overlap) if g_x > 1 else im.width
    tile_h = im.height / (g_y - (g_y - 1) * tile_overlap) if g_y > 1 else im.height
    stride_x = tile_w * (1 - tile_overlap) if g_x > 1 else im.width
    stride_y = tile_h * (1 - tile_overlap) if g_y > 1 else im.height

    positions_x = [c * stride_x for c in range(g_x)]
    if g_x > 1:
        positions_x[-1] = max(0, im.width - tile_w)  # edge-anchor last column
    positions_y = [r * stride_y for r in range(g_y)]
    if g_y > 1:
        positions_y[-1] = max(0, im.height - tile_h)  # edge-anchor last row

    tiles = [
        (px, py, min(im.width, px + tile_w), min(im.height, py + tile_h))
        for py in positions_y for px in positions_x
    ]

    # ---- Step C: predict per tile, map tile-local -> original full-sheet coords ----
    all_dets = []
    for tx1, ty1, tx2, ty2 in tiles:
        tile_img = im.crop((round(tx1), round(ty1), round(tx2), round(ty2)))
        tw, th = tile_img.size
        tile_img_resized = tile_img.resize((model_input_size, model_input_size), Image.Resampling.BILINEAR)
        sx, sy = tw / model_input_size, th / model_input_size  # model space -> scale-normalized-sheet space

        for d in model_predict_fn(tile_img_resized):
            if d["score"] < conf_threshold:
                continue
            x, y, w, h = d["bbox"]
            # tile-local (model space) -> scale-normalized sheet -> ORIGINAL sheet
            fx = (tx1 + x * sx) / scale
            fy = (ty1 + y * sy) / scale
            fw = (w * sx) / scale
            fh = (h * sy) / scale
            all_dets.append({
                "bbox": [fx, fy, fx + fw, fy + fh],  # xyxy, needed for NMS
                "score": d["score"],
                "category_id": d["category_id"],
            })

    # ---- Step D: cross-tile class-aware NMS ----
    kept = _class_aware_nms(all_dets, iou_threshold=nms_iou_threshold)
    final_dets = []
    for k in kept:
        x1, y1, x2, y2 = k["bbox"]
        final_dets.append({
            "bbox": [x1, y1, x2 - x1, y2 - y1],
            "score": k["score"],
            "category_id": k["category_id"],
        })

    print(f"[infer_full_sheet] {os.path.basename(image_path)}: {len(tiles)} tiles, "
          f"{len(all_dets)} raw detections -> {len(final_dets)} after NMS (scale={scale:.3f}x)")
    return final_dets


print("✅ infer_full_sheet() defined.")
print("   Usage: infer_full_sheet(image_path, model_predict_fn, dataset_avg_scale=<scale_stats_avg>)")
print("   where model_predict_fn(tile_image) runs your loaded D-FINE checkpoint on one 640x640 tile")
print("   and returns [{'bbox': [x,y,w,h], 'score': float, 'category_id': int}, ...] in tile-local coords.")