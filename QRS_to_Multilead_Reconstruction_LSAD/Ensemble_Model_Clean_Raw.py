import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from xgboost import XGBRegressor
from skimage.metrics import structural_similarity as ssim
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import glob

# --- Config ---
DATASET = "ptbxl"   # أو "lsad"

DATA_DIR = Path(f"Segments_clean_raw/{DATASET}")
OUTPUT_DIR = Path(f"Stacked_Model_Results_raw_clean_{DATASET}_full")

PLOTS_DIR = OUTPUT_DIR / "plots"
MODELS_DIR = OUTPUT_DIR / "models"
RMSE_PLOTS_DIR = PLOTS_DIR / "rmse_per_point"
for d in [OUTPUT_DIR, PLOTS_DIR, MODELS_DIR, RMSE_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"

INPUT_LEADS = ["I", "V2", "V6"]
TARGET_LEADS = ["II", "V1", "V3", "V4", "V5"]
ALL_LEADS = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]

SEGMENT_LENGTH = 128
BATCH_SIZE = 32
EPOCHS = 300
LEARNING_RATE = 1e-3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Loading Data ---
def load_all_shards(data_dir, prefix):
    seg_files = sorted(glob.glob(str(data_dir / f"{prefix}_segments_shard_*.npy")))
    all_segments = [np.load(f, allow_pickle=True) for f in seg_files]
    segments = np.concatenate(all_segments, axis=0)

    rec_files = sorted(glob.glob(str(data_dir / f"{prefix}_records_shard_*.pkl")))
    all_records = []
    for f in rec_files:
        with open(f, "rb") as fh:
            all_records.extend(pickle.load(fh))
    return segments, all_records

segments, records = load_all_shards(DATA_DIR, prefix=DATASET)
print("Total segments:", segments.shape)
print("Total records:", len(records))

# تقسيم train/val/test
n = len(segments)
train_end = int(0.7 * n)
val_end   = int(0.85 * n)

train_segments, val_segments, test_segments = (
    segments[:train_end], segments[train_end:val_end], segments[val_end:]
)

train_records, val_records, test_records = (
    records[:train_end], records[train_end:val_end], records[val_end:]
)

# --- Helper to flatten metadata dictionaries safely ---
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

# --- Prepare metadata arrays (age & sex) + raw + clean ---
def prepare_meta_arrays(records, segments, input_leads, use_raw=True, use_clean=True):
    meta_list = []
    for rec in records[:len(segments)]:
        features = []

        # clean features
        if use_clean and "features" in rec:
            for lead in input_leads:
                lead_feats = {k: v for k, v in rec["features"].items() if k.startswith(lead + "_")}
                features.extend(list(lead_feats.values()))

        # raw features
        if use_raw and "meta" in rec and "raw_features" in rec["meta"]:
            for lead in input_leads:
                lead_feats = {k: v for k, v in rec["meta"]["raw_features"].items() if k.startswith(lead + "_")}
                features.extend(list(lead_feats.values()))

        # age & sex
        if "meta" in rec:
            age = rec["meta"].get("age", 0.0)
            sex = rec["meta"].get("sex", "M")
            if isinstance(sex, str):
                if sex.upper() == "M":
                    sex_val = 0.0
                elif sex.upper() == "F":
                    sex_val = 1.0
                else:
                    sex_val = float(hash(str(sex)) % (10**6)) / 1e6
            else:
                sex_val = float(sex)
            features.extend([float(age), sex_val])

        meta_list.append(np.array(features, dtype=np.float32))

    max_len = max(len(m) for m in meta_list)
    meta_padded = np.array([np.pad(m, (0, max_len - len(m)), 'constant', constant_values=0.0) for m in meta_list], dtype=np.float32)
    return meta_padded

# --- Prepare clean and raw separately ---
train_meta = prepare_meta_arrays(train_records, train_segments, INPUT_LEADS, use_raw=True, use_clean=True)
val_meta   = prepare_meta_arrays(val_records, val_segments, INPUT_LEADS, use_raw=True, use_clean=True)
test_meta  = prepare_meta_arrays(test_records, test_segments, INPUT_LEADS, use_raw=True, use_clean=True)

print("Metadata shapes - Train:", train_meta.shape, "Val:", val_meta.shape, "Test:", test_meta.shape)

# --- Dataset Class ---
class RichECGDataset(Dataset):
    def __init__(self, segments, meta_features, input_leads, target_lead, use_meta=True):
        self.samples = []
        self.use_meta = use_meta
        for idx, seg in enumerate(segments):
            x_seg = np.stack([seg[:, ALL_LEADS.index(ld)] for ld in input_leads], axis=1)
            if self.use_meta:
                x_meta = meta_features[idx]
                x_seg_flat = x_seg.flatten().astype(np.float32)
                x = np.concatenate([x_seg_flat, x_meta])
            else:
                x = x_seg.astype(np.float32)
            y = seg[:, ALL_LEADS.index(target_lead)].astype(np.float32)
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                self.samples.append((x, y))
        print(f"{target_lead} - num of samples: {len(self.samples)}")

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

# --- MLP Model ---
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

# --- SSIM ---
def compute_ssim_batch(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        score = ssim(y_true[i], y_pred[i], data_range=max(y_true[i].max() - y_true[i].min(), 1e-8))
        scores.append(score)
    return np.mean(scores)

# --- XGB helper ---
def xgb_predict_full(models, X):
    preds = np.zeros((X.shape[0], len(models)))
    for t, model in enumerate(models):
        preds[:, t] = model.predict(X)
    return preds

# --- Collect predictions ---
def collect_predictions(memmap_dir, dataset, mlp, xgb_models, device, prefix):
    memmap_dir = Path(memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    n_samples = len(dataset)
    seq_len = int(dataset[0][1].shape[0])
    n_meta_models = 2  # MLP, XGB

    meta_X = np.memmap(memmap_dir / f"{prefix}_metaX.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len, n_meta_models))
    meta_y = np.memmap(memmap_dir / f"{prefix}_metaY.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len))

    mlp.eval()
    seg_flat_len = SEGMENT_LENGTH * len(INPUT_LEADS)

    try:
        mlp_input_dim = mlp.net[0].in_features
    except Exception:
        mlp_input_dim = None

    xgb_input_dim = None
    if len(xgb_models) > 0 and hasattr(xgb_models[0], "n_features_in_"):
        xgb_input_dim = int(xgb_models[0].n_features_in_)

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"Collecting {prefix} predictions"):
            bsz = xb.shape[0]
            if xb.dim() == 2:
                mlp_in = xb.to(device)
                mlp_out = mlp(mlp_in.view(bsz, -1)).cpu().numpy()
                Xb_flat = xb.cpu().numpy().reshape(bsz, -1)
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)
            elif xb.dim() == 3:
                seg_flat = xb.reshape(bsz, -1)
                if mlp_input_dim is None:
                    mlp_in_tensor = seg_flat.to(device)
                else:
                    pad_len = mlp_input_dim - seg_flat.shape[1]
                    if pad_len > 0:
                        pad_tensor = torch.zeros((bsz, pad_len), dtype=seg_flat.dtype)
                        mlp_in_tensor = torch.cat([seg_flat, pad_tensor], dim=1).to(device)
                    else:
                        mlp_in_tensor = seg_flat.to(device)
                mlp_out = mlp(mlp_in_tensor.view(bsz, -1)).cpu().numpy()
                seg_flat_np = seg_flat.cpu().numpy()
                if xgb_input_dim is None:
                    Xb_flat = seg_flat_np
                else:
                    pad_cols = xgb_input_dim - seg_flat_np.shape[1]
                    if pad_cols > 0:
                        Xb_flat = np.concatenate([seg_flat_np, np.zeros((bsz, pad_cols), dtype=np.float32)], axis=1)
                    else:
                        Xb_flat = seg_flat_np
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)
            else:
                raise RuntimeError(f"Unsupported xb.dim() = {xb.dim()}")

            meta_X[idx:idx+bsz, :, 0] = mlp_out
            meta_X[idx:idx+bsz, :, 1] = xgb_out
            meta_y[idx:idx+bsz] = yb.cpu().numpy()
            idx += bsz

    del meta_X, meta_y
    meta_X = np.memmap(memmap_dir / f"{prefix}_metaX.dat", dtype="float32", mode="r", shape=(n_samples, seq_len, n_meta_models))
    meta_y = np.memmap(memmap_dir / f"{prefix}_metaY.dat", dtype="float32", mode="r", shape=(n_samples, seq_len))
    print(f"[✓] Saved predictions for {prefix} at {memmap_dir} (rows={n_samples})")
    return meta_X, meta_y

# --- Training & Evaluation ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n🔧 Training stacked model for lead: {lead}...")

        train_ds = RichECGDataset(train_segments, train_meta, INPUT_LEADS, lead, use_meta=True)
        val_ds   = RichECGDataset(val_segments, val_meta, INPUT_LEADS, lead, use_meta=True)
        test_ds  = RichECGDataset(test_segments, test_meta, INPUT_LEADS, lead, use_meta=True)

        if len(train_ds) == 0:
            print(f"⚠️ Skipping {lead}: Not enough samples")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, drop_last=True)

        # --- MLP training ---
        input_dim = train_ds[0][0].numel()
        output_dim = SEGMENT_LENGTH
        mlp = MLP(input_dim, output_dim).to(DEVICE)
        loss_fn = nn.MSELoss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        patience, counter = 15, 0
        best_model_state = None

        for epoch in range(EPOCHS):
            mlp.train()
            total_loss = 0.0
            for xb, yb in tqdm(train_loader, leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                xb = xb.view(xb.size(0), -1)
                pred = mlp(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_train_loss = total_loss / len(train_ds)

            mlp.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    xb_flat = xb.view(xb.size(0), -1)
                    pred = mlp(xb_flat)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)

            print(f"Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = mlp.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"⏹ Early stopping at epoch {epoch+1}")
                    break

        if best_model_state is not None:
            mlp.load_state_dict(best_model_state)

        torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_model_{lead}.pt")

        # --- XGB training ---
        X_train = np.array([x.numpy().reshape(-1) for x, _ in train_ds])
        y_train = np.array([y.numpy() for _, y in train_ds])

        xgb_models = []
        for t in tqdm(range(SEGMENT_LENGTH), desc=f"Training XGB for lead {lead}"):
            model_t = XGBRegressor(tree_method="hist", n_jobs=-1, verbosity=0)
            model_t.fit(X_train, y_train[:, t])
            xgb_models.append(model_t)

        xgb_model = xgb_models

        # --- Collect predictions ---
        train_X, train_y = collect_predictions("cache_preds", train_ds, mlp, xgb_model, DEVICE, "train")
        val_X, val_y = collect_predictions("cache_preds", val_ds, mlp, xgb_model, DEVICE, "val")
        test_X, test_y = collect_predictions("cache_preds", test_ds, mlp, xgb_model, DEVICE, "test")

        meta_X_train = train_X.reshape(-1, 2)
        meta_y_train = train_y.reshape(-1)
        meta_X_test = test_X.reshape(-1, 2)
        meta_y_test = test_y.reshape(-1)

        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, meta_y_train)
        meta_pred = meta_model.predict(meta_X_test).reshape(test_y.shape)

        with open(MODELS_DIR / f"ridge_model_{lead}.pkl", "wb") as f:
            pickle.dump(meta_model, f)
        with open(MODELS_DIR / f"xgb_model_{lead}.pkl", "wb") as f:
            pickle.dump(xgb_model, f)

        # --- Metrics ---
        rmse = np.sqrt(mean_squared_error(test_y.flatten(), meta_pred.flatten()))
        r2 = r2_score(test_y.flatten(), meta_pred.flatten())
        pearson_corr_score = np.mean([
            pearsonr(test_y[:, i], meta_pred[:, i])[0]
            for i in range(SEGMENT_LENGTH) if np.std(test_y[:, i]) > 0
        ])
        ssim_score = compute_ssim_batch(test_y, meta_pred)

        print(f"\nLead {lead} Evaluation Summary:")
        print(f"  RMSE         = {rmse:.4f}")
        print(f"  R^2          = {r2:.4f}")
        print(f"  Pearson Corr = {pearson_corr_score:.4f}")
        print(f"  SSIM         = {ssim_score:.4f}")

        rmse_per_point = np.sqrt(np.mean((test_y - meta_pred) ** 2, axis=0)).tolist()
        report.write(f"\nEvaluation for Lead {lead}:\n")
        report.write(f"RMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {pearson_corr_score:.4f}\nSSIM: {ssim_score:.4f}\n")
        report.write(f"RMSE per point (length {SEGMENT_LENGTH}):\n")
        report.write(", ".join(f"{v:.6f}" for v in rmse_per_point) + "\n")

        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()
