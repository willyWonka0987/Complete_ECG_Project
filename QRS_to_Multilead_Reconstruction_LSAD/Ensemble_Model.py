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
DATA_DIR = Path(f"Segments/{DATASET}")
OUTPUT_DIR = Path(f"Stacked_Model_Results_CNN_AVG_{DATASET}2")

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
def prepare_meta_arrays(records, segments, input_leads, use_raw=False, use_clean=True):
    meta_list = []
    for rec in records[:len(segments)]:
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
                sex_val = 0.0 if sex.upper()=="M" else 1.0
            else:
                sex_val = float(sex)
            features.extend([float(age), sex_val])
        meta_list.append(np.array(features, dtype=np.float32))
    max_len = max(len(m) for m in meta_list)
    meta_padded = np.array([np.pad(m, (0, max_len - len(m)), 'constant', constant_values=0.0) for m in meta_list], dtype=np.float32)
    return meta_padded

train_meta = prepare_meta_arrays(train_records, train_segments, INPUT_LEADS)
val_meta   = prepare_meta_arrays(val_records, val_segments, INPUT_LEADS)
test_meta  = prepare_meta_arrays(test_records, test_segments, INPUT_LEADS)

print("Metadata shapes - Train:", train_meta.shape, "Val:", val_meta.shape, "Test:", test_meta.shape)

# --- Dataset ---
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

# --- 1D-CNN ---
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
        x = x.permute(0, 2, 1)
        x = self.conv_net(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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

# --- Main Training Loop for all leads ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n🔧 Processing lead: {lead}")

        train_ds = RichECGDataset(train_segments, train_meta, INPUT_LEADS, lead, use_meta=True)
        val_ds   = RichECGDataset(val_segments, val_meta, INPUT_LEADS, lead, use_meta=True)
        test_ds  = RichECGDataset(test_segments, test_meta, INPUT_LEADS, lead, use_meta=True)

        if len(train_ds) == 0:
            print(f"⚠️ Skipping {lead}: Not enough samples")
            continue

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

        # ----------------------------
        # Step 1: Train MLP
        # ----------------------------
        input_dim = train_ds[0][0].numel()
        output_dim = SEGMENT_LENGTH
        mlp = MLP(input_dim, output_dim).to(DEVICE)
        loss_fn = nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(mlp.parameters(), lr=LEARNING_RATE)

        best_val_loss = float("inf")
        patience, counter = 15, 0
        best_state = None

        for epoch in range(EPOCHS):
            mlp.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = mlp(xb)
                loss = loss_fn(pred, yb)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_ds)

            # Validation
            mlp.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = mlp(xb)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)
            print(f"[MLP] Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = mlp.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("⏹ Early stopping MLP")
                    break
        mlp.load_state_dict(best_state)
        torch.save(mlp.state_dict(), MODELS_DIR / f"mlp_{lead}.pt")

        # ----------------------------
        # Step 2: Train CNN
        # ----------------------------
        cnn_model = CNN1D(len(INPUT_LEADS), output_dim).to(DEVICE)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
        best_val_loss = float("inf")
        counter = 0
        best_state = None

        for epoch in range(EPOCHS):
            cnn_model.train()
            total_loss = 0
            for xb, yb in train_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                xb_sig = xb[:, :SEGMENT_LENGTH*len(INPUT_LEADS)].view(xb.size(0), SEGMENT_LENGTH, len(INPUT_LEADS))
                pred = cnn_model(xb_sig)
                loss = loss_fn(pred, yb)
                cnn_optimizer.zero_grad()
                loss.backward()
                cnn_optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_loss = total_loss / len(train_ds)

            cnn_model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    xb_sig = xb[:, :SEGMENT_LENGTH*len(INPUT_LEADS)].view(xb.size(0), SEGMENT_LENGTH, len(INPUT_LEADS))
                    pred = cnn_model(xb_sig)
                    val_loss += loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(val_ds)
            print(f"[CNN] Epoch {epoch+1} - Train Loss: {avg_loss:.4f} - Val Loss: {avg_val_loss:.4f}")
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_state = cnn_model.state_dict()
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print("⏹ Early stopping CNN")
                    break
        cnn_model.load_state_dict(best_state)
        torch.save(cnn_model.state_dict(), MODELS_DIR / f"cnn_{lead}.pt")

        # ----------------------------
        # Step 3: Train XGB
        # ----------------------------
        X_train = np.array([x.numpy().reshape(-1) for x, _ in train_ds])
        y_train = np.array([y.numpy() for _, y in train_ds])
        xgb_models = []
        for t in tqdm(range(SEGMENT_LENGTH), desc=f"Training XGB for {lead}"):
            model_t = XGBRegressor(tree_method="hist", n_jobs=-1, verbosity=0)
            model_t.fit(X_train, y_train[:, t])
            xgb_models.append(model_t)

        with open(MODELS_DIR / f"xgb_{lead}.pkl", "wb") as f:
            pickle.dump(xgb_models, f)

        # ----------------------------
        # Step 4: Average Ensembling
        # ----------------------------
        def collect_pred(ds):
            mlp.eval()
            cnn_model.eval()
            Xb = []
            Yb = []
            loader = DataLoader(ds, batch_size=64)
            with torch.no_grad():
                for xb, yb in loader:
                    xb, yb = xb.to(DEVICE), yb.numpy()
                    # MLP
                    pred_mlp = mlp(xb).cpu().numpy()
                    # CNN
                    xb_sig = xb[:, :SEGMENT_LENGTH*len(INPUT_LEADS)].view(xb.size(0), SEGMENT_LENGTH, len(INPUT_LEADS))
                    pred_cnn = cnn_model(xb_sig).cpu().numpy()
                    # XGB
                    X_flat = xb.view(xb.size(0), -1).cpu().numpy()
                    pred_xgb = xgb_predict_full(xgb_models, X_flat)
                    # Average
                    pred_avg = (pred_mlp + pred_cnn + pred_xgb) / 3.0
                    Xb.append(pred_avg)
                    Yb.append(yb)
            return np.concatenate(Xb, axis=0), np.concatenate(Yb, axis=0)

        # Get predictions
        meta_pred, y_test_meta = collect_pred(test_ds)

        # --- Evaluation ---
        rmse = np.sqrt(mean_squared_error(y_test_meta.flatten(), meta_pred.flatten()))
        r2 = r2_score(y_test_meta.flatten(), meta_pred.flatten())
        pearson_corr_score = np.mean([pearsonr(y_test_meta[:, i], meta_pred[:, i])[0] 
                                    for i in range(SEGMENT_LENGTH) if np.std(y_test_meta[:, i])>0])
        ssim_score = compute_ssim_batch(y_test_meta, meta_pred)

        print(f"\nLead {lead} Evaluation Summary (Average Ensemble):")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  R^2 = {r2:.4f}")
        print(f"  Pearson Corr = {pearson_corr_score:.4f}")
        print(f"  SSIM = {ssim_score:.4f}")

        report.write(f"\nEvaluation for Lead {lead} (Average Ensemble):\n")
        report.write(f"RMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson: {pearson_corr_score:.4f}\nSSIM: {ssim_score:.4f}\n")
