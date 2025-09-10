#!/usr/bin/env python3
"""
Fine-tune script for stacked ECG reconstruction models (MLP + XGB +/- CNN meta)
Based on your original training script — kept structure and variable names as much as possible.
Designed to fine-tune pretrained models trained on PTB-XL and resume training on LSAD.

How it works (short):
 - Loads shards from Segments/{DATASET}
 - Prepares meta features (clean + raw features + age/sex)
 - Trains / fine-tunes MLP (loads pretrained if available)
 - Loads/resumes XGB trainers per timepoint (optionally resumes from pretrained)
 - Collects predictions from MLP/XGB (and optional CNN) into memory-mapped cache
 - Trains a Ridge meta-model on the collected predictions
 - Evaluates and saves metrics + figures

Notes:
 - The script tries several pretrained filename patterns so it is robust to previous naming.
 - If you want CNN to be used inside the meta (i.e., include CNN predictions to Ridge), set
   FINETUNE_INCLUDE_CNN = True below. By default the meta uses MLP + XGB.

"""

import os
import glob
import pickle
from pathlib import Path
import logging
import random

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from skimage.metrics import structural_similarity as ssim

# -------------------------
# --- Config (تعدّل هنا)
# -------------------------
DATASET = "lsad"   # "ptbxl" or "lsad"
DATA_DIR = Path(f"Segments/{DATASET}")
OUTPUT_DIR = Path(f"Stacked_Model_Results_{DATASET}_finetune")

PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
RMSE_PLOTS_DIR = PLOTS_DIR / "rmse_per_point"
for d in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, RMSE_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"

# المسار اللي فيه موديلات PTB-XL المدربة سابقاً (عدل لو مختلف)
PRETRAINED_MODELS_DIR = Path("Stacked_Model_Results_CNN_AVG_ptbxl2/models")

INPUT_LEADS = ["I", "V2", "V6"]
TARGET_LEADS = ["II", "V1", "V3", "V4", "V5"]
ALL_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

SEGMENT_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----- Fine-tune specific flags (يمكن تغييرها بسهولة) -----
FINETUNE_MLP = True
FINETUNE_MLP_EPOCHS = 150           # fewer epochs for fine-tuning
FINETUNE_LEARNING_RATE = 1e-4       # lower LR for fine-tuning

FINETUNE_XGB = True
FINETUNE_XGB_ADDITIONAL_ESTIMATORS = 50  # additional boosting rounds when resuming

FINETUNE_RIDGE = True   # re-fit ridge on new meta-training data

# Include CNN predictions into meta (set True if you want MLP+CNN+XGB features)
FINETUNE_INCLUDE_CNN = True
# If you want to fine-tune CNN weights as well, enable this (default: False)
FINETUNE_CNN = True
FINETUNE_CNN_EPOCHS = 150

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# -------------------------
# --- Loading Data ---
# -------------------------

def load_all_shards(data_dir, prefix):
    seg_files = sorted(glob.glob(str(data_dir / f"{prefix}_segments_shard_*.npy")))
    if len(seg_files) == 0:
        raise FileNotFoundError(f"No segment shards found in {data_dir} with prefix {prefix}")
    all_segments = [np.load(f, allow_pickle=True) for f in seg_files]
    segments = np.concatenate(all_segments, axis=0)

    rec_files = sorted(glob.glob(str(data_dir / f"{prefix}_records_shard_*.pkl")))
    if len(rec_files) == 0:
        raise FileNotFoundError(f"No record shards found in {data_dir} with prefix {prefix}")
    all_records = []
    for f in rec_files:
        with open(f, "rb") as fh:
            all_records.extend(pickle.load(fh))
    return segments, all_records


segments, records = load_all_shards(DATA_DIR, prefix=DATASET)
logging.info(f"Total segments: {segments.shape}")
logging.info(f"Total records: {len(records)}")

# --- Train/Val/Test split ---
n = len(segments)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

train_segments, val_segments, test_segments = segments[:train_end], segments[train_end:val_end], segments[val_end:]
train_records, val_records, test_records = records[:train_end], records[train_end:val_end], records[val_end:]

# --- Helper to flatten metadata ---
def flatten_meta(meta_dict):
    flat = []
    for k, v in meta_dict.items():
        if isinstance(v, (list, np.ndarray)):
            flat.extend(np.ravel(v))
        elif isinstance(v, (int, float)):
            flat.append(v)
        else:
            flat.append(float(hash(str(v)) % (10**6)) / 1e6)
    return flat

# --- Prepare metadata arrays ---
def prepare_meta_arrays(records_sub, segments_sub, input_leads, use_raw=True, use_clean=True):
    meta_list = []
    for rec in records_sub[:len(segments_sub)]:
        features = []
        if use_clean and "features" in rec:
            for lead in input_leads:
                lead_feats = {k: v for k, v in rec["features"].items() if k.startswith(lead + "_")}
                features.extend(list(lead_feats.values()))
        if use_raw and "meta" in rec and "raw_features" in rec["meta"]:
            for lead in input_leads:
                lead_feats = {k: v for k, v in rec["meta"]["raw_features"].items() if k.startswith(lead + "_")}
                features.extend(list(lead_feats.values()))
        if "meta" in rec:
            age = rec["meta"].get("age", 0.0)
            sex = rec["meta"].get("sex", "M")
            if isinstance(sex, str):
                sex_val = 0.0 if sex.upper() == "M" else 1.0 if sex.upper() == "F" else float(hash(str(sex)) % (10**6)) / 1e6
            else:
                sex_val = float(sex)
            features.extend([float(age), sex_val])
        meta_list.append(np.array(features, dtype=np.float32))
    max_len = max(len(m) for m in meta_list)
    meta_padded = np.array([np.pad(m, (0, max_len - len(m)), 'constant', constant_values=0.0) for m in meta_list], dtype=np.float32)
    return meta_padded

train_meta = prepare_meta_arrays(train_records, train_segments, INPUT_LEADS, use_raw=True, use_clean=True)
val_meta   = prepare_meta_arrays(val_records, val_segments, INPUT_LEADS, use_raw=True, use_clean=True)
test_meta  = prepare_meta_arrays(test_records, test_segments, INPUT_LEADS, use_raw=True, use_clean=True)

logging.info(f"Metadata shapes - Train: {train_meta.shape} Val: {val_meta.shape} Test: {test_meta.shape}")

# --- Dataset ---
class RichECGDataset(Dataset):
    def __init__(self, segments, meta_features, input_leads, target_lead, use_meta=True):
        self.samples = []
        self.use_meta = use_meta
        for idx, seg in enumerate(segments):
            x_seg = np.stack([seg[:, ALL_LEADS.index(ld)] for ld in input_leads], axis=1)  # (T, C)
            if self.use_meta:
                x_meta = meta_features[idx]
                x_seg_flat = x_seg.flatten().astype(np.float32)
                x = np.concatenate([x_seg_flat, x_meta])
            else:
                x = x_seg.astype(np.float32)
            y = seg[:, ALL_LEADS.index(target_lead)].astype(np.float32)
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                self.samples.append((x, y))
        logging.info(f"{target_lead} - num of samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# --- Pearson correlation ---
def pearson_corr(y_true, y_pred):
    vx = y_true - torch.mean(y_true, dim=1, keepdim=True)
    vy = y_pred - torch.mean(y_pred, dim=1, keepdim=True)
    corr = torch.sum(vx * vy, dim=1) / (
        torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)) + 1e-8
    )
    return corr

# --- MLP ---
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        return self.net(x)

# --- 1D CNN (if needed) ---
class CNN1D(nn.Module):
    def __init__(self, n_channels, output_dim):
        super().__init__()
        self.conv_net = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self.fc = nn.Linear(64 * SEGMENT_LENGTH, output_dim)

    def forward(self, x):
        # x: (B, T, C) -> permute to (B, C, T)
        x = x.permute(0, 2, 1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# --- SSIM ---
def compute_ssim_batch(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        # skimage.metrics.structural_similarity accepts any-dimensional arrays with same shape
        score = ssim(y_true[i], y_pred[i], data_range=max(y_true[i].max() - y_true[i].min(), 1e-8))
        scores.append(score)
    return float(np.mean(scores))

# --- XGB helper ---
def xgb_predict_full(models, X):
    preds = np.zeros((X.shape[0], len(models)), dtype=np.float32)
    for t, model in enumerate(models):
        preds[:, t] = model.predict(X)
    return preds


# --- Collect predictions (memmap) ---
def collect_predictions(memmap_dir, dataset, mlp, xgb_models, device, prefix, cnn_model=None, use_meta=True):
    memmap_dir = Path(memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    n_samples = len(dataset)
    seq_len = int(dataset[0][1].shape[0])

    feature_count = 2 + (1 if (cnn_model is not None and FINETUNE_INCLUDE_CNN) else 0)

    meta_X = np.memmap(memmap_dir / f"{prefix}_metaX.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len, feature_count))
    meta_y = np.memmap(memmap_dir / f"{prefix}_metaY.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len))

    mlp.eval()
    if cnn_model is not None:
        cnn_model.eval()

    seg_flat_len = SEGMENT_LENGTH * len(INPUT_LEADS)

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"Collecting {prefix} preds"):
            bsz = xb.shape[0]

            # If flattened (use_meta=True) xb.dim() == 2
            if xb.dim() == 2:
                xb_np = xb.cpu().numpy()

                # MLP
                mlp_in = xb.to(device)
                mlp_out = mlp(mlp_in.view(bsz, -1)).cpu().numpy()

                # XGB
                Xb_flat = xb_np.reshape(bsz, -1)
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)

                # CNN (reconstruct signal part)
                if cnn_model is not None and FINETUNE_INCLUDE_CNN:
                    if xb_np.shape[1] < seg_flat_len:
                        raise RuntimeError("Flattened xb shorter than expected segment flat length")
                    seg_part = xb_np[:, :seg_flat_len].reshape(bsz, SEGMENT_LENGTH, len(INPUT_LEADS))
                    seg_part_tensor = torch.tensor(seg_part, dtype=torch.float32).to(device)
                    cnn_out = cnn_model(seg_part_tensor).cpu().numpy()
                else:
                    cnn_out = np.zeros_like(mlp_out)

            # If structured (use_meta=False) xb.dim() == 3: (B, T, C)
            elif xb.dim() == 3:
                xb_sig = xb.to(device)
                seg_flat = xb_sig.view(bsz, -1)

                # MLP: pad zeros for meta part if needed
                try:
                    mlp_in_dim = mlp.net[0].in_features
                except Exception:
                    mlp_in_dim = seg_flat.shape[1]

                if seg_flat.shape[1] < mlp_in_dim:
                    pad_len = mlp_in_dim - seg_flat.shape[1]
                    pad_tensor = torch.zeros((bsz, pad_len), dtype=seg_flat.dtype, device=seg_flat.device)
                    mlp_in_tensor = torch.cat([seg_flat, pad_tensor], dim=1)
                else:
                    mlp_in_tensor = seg_flat
                mlp_out = mlp(mlp_in_tensor.view(bsz, -1)).cpu().numpy()

                seg_flat_np = seg_flat.cpu().numpy()
                if any([m is not None and hasattr(m, "n_features_in_") for m in xgb_models]):
                    # pad to xgb expected if necessary
                    xgb_in_dim = getattr(xgb_models[0], "n_features_in_", seg_flat_np.shape[1])
                    if seg_flat_np.shape[1] < xgb_in_dim:
                        Xb_flat = np.concatenate([seg_flat_np, np.zeros((bsz, xgb_in_dim - seg_flat_np.shape[1]), dtype=np.float32)], axis=1)
                    else:
                        Xb_flat = seg_flat_np
                else:
                    Xb_flat = seg_flat_np
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)

                if cnn_model is not None and FINETUNE_INCLUDE_CNN:
                    cnn_out = cnn_model(xb_sig).cpu().numpy()
                else:
                    cnn_out = np.zeros_like(mlp_out)

            else:
                raise RuntimeError("Unsupported xb.dim()")

            # write
            meta_X[idx:idx+bsz, :, 0] = mlp_out
            meta_X[idx:idx+bsz, :, 1] = xgb_out
            if feature_count == 3:
                meta_X[idx:idx+bsz, :, 2] = cnn_out
            meta_y[idx:idx+bsz] = yb.cpu().numpy()
            idx += bsz

    # reopen read-only
    del meta_X, meta_y
    meta_X = np.memmap(memmap_dir / f"{prefix}_metaX.dat", dtype="float32", mode="r", shape=(n_samples, seq_len, feature_count))
    meta_y = np.memmap(memmap_dir / f"{prefix}_metaY.dat", dtype="float32", mode="r", shape=(n_samples, seq_len))
    logging.info(f"Saved predictions for {prefix} at {memmap_dir} (rows={n_samples})")
    return meta_X, meta_y


# --- Main Training Loop for all leads ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        logging.info(f"\n--- Processing lead: {lead}")

        train_ds = RichECGDataset(train_segments, train_meta, INPUT_LEADS, lead, use_meta=True)
        val_ds   = RichECGDataset(val_segments, val_meta, INPUT_LEADS, lead, use_meta=True)
        test_ds  = RichECGDataset(test_segments, test_meta, INPUT_LEADS, lead, use_meta=True)

        if len(train_ds) == 0:
            logging.warning(f"Skipping {lead}: not enough samples")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, pin_memory=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

        # ----------------------------
        # Step 1: MLP (load pretrained if exists) 
        # ----------------------------
        input_dim = train_ds[0][0].numel()
        output_dim = SEGMENT_LENGTH
        mlp = MLP(input_dim, output_dim).to(DEVICE)

        # try multiple filename patterns for backwards compatibility
        candidate_mlp_files = [
            PRETRAINED_MODELS_DIR / f"mlp_{lead}.pt",
            PRETRAINED_MODELS_DIR / f"mlp_model_{lead}.pt",
            PRETRAINED_MODELS_DIR / f"mlp_{lead}.pth",
            PRETRAINED_MODELS_DIR / f"mlp_model_{lead}.pth",
        ]
        loaded = False
        for p in candidate_mlp_files:
            if p.exists():
                try:
                    mlp.load_state_dict(torch.load(p, map_location=DEVICE))
                    logging.info(f"Loaded pretrained MLP from {p}")
                    loaded = True
                    break
                except Exception as e:
                    logging.warning(f"Failed loading {p}: {e}")
        if not loaded:
            logging.info("No pretrained MLP found — training from scratch (or fine-tune from scratch)")

        mlp_lr = (FINETUNE_LEARNING_RATE if FINETUNE_MLP else LEARNING_RATE)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=mlp_lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        best_val_loss = float('inf')
        patience, counter = 15, 0
        best_state = None

        epochs_to_run = (FINETUNE_MLP_EPOCHS if FINETUNE_MLP else EPOCHS)

        for epoch in range(epochs_to_run):
            mlp.train()
            total_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                xb = xb.view(xb.size(0), -1)
                pred = mlp(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_ds)

            # validation
            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    xb_flat = xb.view(xb.size(0), -1)
                    pred = mlp(xb_flat)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)
            logging.info(f"[MLP] Epoch {epoch+1}/{epochs_to_run} Train: {avg_loss:.6f} Val: {avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = mlp.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    logging.info("Early stopping MLP")
                    break
            scheduler.step(avg_val_loss)

        if best_state is not None:
            mlp.load_state_dict(best_state)
        torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_{lead}.pt")

        # ----------------------------
        # Optional CNN loading / fine-tune
        # ----------------------------
        cnn_model = None
        if FINETUNE_INCLUDE_CNN or FINETUNE_CNN:
            cnn = CNN1D(len(INPUT_LEADS), output_dim).to(DEVICE)
            candidate_cnn_files = [
                PRETRAINED_MODELS_DIR / f"cnn_{lead}.pt",
                PRETRAINED_MODELS_DIR / f"cnn_model_{lead}.pt",
            ]
            loaded_cnn = False
            for p in candidate_cnn_files:
                if p.exists():
                    try:
                        cnn.load_state_dict(torch.load(p, map_location=DEVICE))
                        logging.info(f"Loaded pretrained CNN from {p}")
                        loaded_cnn = True
                        break
                    except Exception as e:
                        logging.warning(f"Failed loading {p}: {e}")
            if FINETUNE_CNN:
                cnn_optimizer = torch.optim.Adam(cnn.parameters(), lr=FINETUNE_LEARNING_RATE)
                best_val = float('inf')
                cnt = 0
                for ep in range(FINETUNE_CNN_EPOCHS):
                    cnn.train()
                    total = 0.0
                    for xb, yb in train_loader:
                        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                        xb_sig = xb[:, :SEGMENT_LENGTH*len(INPUT_LEADS)].view(xb.size(0), SEGMENT_LENGTH, len(INPUT_LEADS))
                        pred = cnn(xb_sig)
                        loss = loss_fn(pred, yb)
                        cnn_optimizer.zero_grad()
                        loss.backward()
                        cnn_optimizer.step()
                        total += loss.item() * xb.size(0)
                    val_l = 0.0
                    cnn.eval()
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                            xb_sig = xb[:, :SEGMENT_LENGTH*len(INPUT_LEADS)].view(xb.size(0), SEGMENT_LENGTH, len(INPUT_LEADS))
                            pred = cnn(xb_sig)
                            val_l += loss_fn(pred, yb).item() * xb.size(0)
                    avg_val = val_l / len(val_ds)
                    logging.info(f"[CNN] Epoch {ep+1}/{FINETUNE_CNN_EPOCHS} Val: {avg_val:.6f}")
                    if avg_val < best_val:
                        best_val = avg_val
                        cnt = 0
                        best_cnn_state = cnn.state_dict()
                    else:
                        cnt += 1
                        if cnt >= 8:
                            logging.info("Early stopping CNN")
                            break
                if 'best_cnn_state' in locals():
                    cnn.load_state_dict(best_cnn_state)
            if loaded_cnn or FINETUNE_CNN:
                cnn_model = cnn
                torch.save(cnn_model.state_dict(), MODELS_DIR / f"cnn_{lead}.pt")

        # ----------------------------
        # Step 3: XGB (load & optionally resume)
        # ----------------------------
        X_train = np.array([x.numpy().reshape(-1) for x, _ in train_ds])
        y_train = np.array([y.numpy() for _, y in train_ds])

        candidate_xgb_files = [
            PRETRAINED_MODELS_DIR / f"xgb_{lead}.pkl",
            PRETRAINED_MODELS_DIR / f"xgb_model_{lead}.pkl",
            PRETRAINED_MODELS_DIR / f"xgb_{lead}.pkl",
        ]
        pretrained_xgb_models = None
        for p in candidate_xgb_files:
            if p.exists():
                try:
                    with open(p, 'rb') as f:
                        pretrained_xgb_models = pickle.load(f)
                    logging.info(f"Loaded pretrained XGB list from {p}")
                    break
                except Exception as e:
                    logging.warning(f"Failed loading XGB from {p}: {e}")

        xgb_models = []
        for t in tqdm(range(SEGMENT_LENGTH), desc=f"Training XGB for {lead}"):
            if FINETUNE_XGB and pretrained_xgb_models is not None:
                prev = pretrained_xgb_models[t]
                # if prev is sklearn wrapper and has get_booster, pass booster; else pass prev
                try:
                    prev_booster = prev.get_booster() if hasattr(prev, 'get_booster') else prev
                except Exception:
                    prev_booster = prev
                model_t = XGBRegressor(tree_method='hist', n_jobs=-1, verbosity=0, n_estimators=FINETUNE_XGB_ADDITIONAL_ESTIMATORS)
                # resume training
                try:
                    model_t.fit(X_train, y_train[:, t], xgb_model=prev_booster)
                except TypeError:
                    # older/newer xgboost versions may accept different types; fallback to fit normally
                    model_t = prev
                except Exception as e:
                    logging.warning(f"XGB resume failed for timepoint {t}: {e}")
                    model_t = prev
            else:
                if pretrained_xgb_models is not None and not FINETUNE_XGB:
                    model_t = pretrained_xgb_models[t]
                else:
                    model_t = XGBRegressor(tree_method='hist', n_jobs=-1, verbosity=0)
                    model_t.fit(X_train, y_train[:, t])
            xgb_models.append(model_t)

        with open(MODELS_DIR / f"xgb_{lead}.pkl", 'wb') as f:
            pickle.dump(xgb_models, f)

        # ----------------------------
        # Collect predictions into memmap
        # ----------------------------
        train_X, train_y = collect_predictions("cache_preds", train_ds, mlp, xgb_models, DEVICE, "train", cnn_model=cnn_model if FINETUNE_INCLUDE_CNN else None)
        val_X, val_y     = collect_predictions("cache_preds", val_ds, mlp, xgb_models, DEVICE, "val", cnn_model=cnn_model if FINETUNE_INCLUDE_CNN else None)
        test_X, test_y   = collect_predictions("cache_preds", test_ds, mlp, xgb_models, DEVICE, "test", cnn_model=cnn_model if FINETUNE_INCLUDE_CNN else None)

        # -------------------------
        # Average Ensemble (بدل Ridge)
        # -------------------------
        avg_pred = np.mean(test_X, axis=2)  # average over [mlp, xgb, (cnn)]
        meta_pred = avg_pred  # rename للانسجام مع الكود السابق

        # --- Metrics ---
        rmse = np.sqrt(mean_squared_error(test_y.flatten(), meta_pred.flatten()))
        r2 = r2_score(test_y.flatten(), meta_pred.flatten())
        pearson_corr_score = np.mean([
            pearsonr(test_y[:, i], meta_pred[:, i])[0]
            for i in range(SEGMENT_LENGTH) if np.std(test_y[:, i]) > 0
        ])
        ssim_score = compute_ssim_batch(test_y, meta_pred)

        logging.info(f"Lead {lead} Evaluation Summary:")
        logging.info(f"  RMSE = {rmse:.6f}")
        logging.info(f"  R^2  = {r2:.6f}")
        logging.info(f"  Pearson = {pearson_corr_score:.6f}")
        logging.info(f"  SSIM = {ssim_score:.6f}")

        rmse_per_point = np.sqrt(np.mean((test_y - meta_pred) ** 2, axis=0)).tolist()
        report.write(f"\nEvaluation for Lead {lead}:\n")
        report.write(f"RMSE: {rmse:.6f}\nR^2: {r2:.6f}\nPearson: {pearson_corr_score:.6f}\nSSIM: {ssim_score:.6f}\n")
        report.write("RMSE per point:\n")
        report.write(", ".join(f"{v:.6f}" for v in rmse_per_point) + "\n")

        # Plot RMSE per point
        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()
