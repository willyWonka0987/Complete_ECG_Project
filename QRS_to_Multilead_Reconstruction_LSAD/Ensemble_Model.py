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

# --- Config ---
DATA_DIR = Path("Segments/ptbxl")
OUTPUT_DIR = Path("Stacked_Model_Results")
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

# --- Load segments ---
print("Loading segmented dataset...")
train_segments = np.load(DATA_DIR / "train_segments.npy")
val_segments = np.load(DATA_DIR / "val_segments.npy")
test_segments = np.load(DATA_DIR / "test_segments.npy")
print("Train:", train_segments.shape, "Val:", val_segments.shape, "Test:", test_segments.shape)

# --- Load metadata/features from records.pkl ---
with open(DATA_DIR / "records.pkl", "rb") as f:
    records = pickle.load(f)

# --- Helper to flatten metadata dictionaries safely ---
def flatten_meta(meta_dict):
    """Flatten nested metadata into a 1D array of numbers/floats."""
    flat = []
    for k, v in meta_dict.items():
        if isinstance(v, (list, np.ndarray)):
            flat.extend(np.ravel(v))
        elif isinstance(v, (int, float)):
            flat.append(v)
        else:
            # convert strings/bools to numeric hash to preserve uniqueness
            flat.append(float(hash(str(v)) % (10**6)) / 1e6)
    return flat

# --- Prepare metadata arrays ---
def prepare_meta_arrays(records, segments):
    meta_list = []
    for rec in records[:len(segments)]:
        meta_dict = rec.get("meta", {}) or {}
        meta_arr = flatten_meta(meta_dict)
        meta_list.append(meta_arr)
    max_len = max(len(m) for m in meta_list)
    # Pad meta to max length
    meta_padded = np.array([np.pad(m, (0, max_len - len(m)), 'constant', constant_values=0.0) for m in meta_list], dtype=np.float32)
    return meta_padded

train_meta = prepare_meta_arrays(records, train_segments)
val_meta   = prepare_meta_arrays(records[len(train_segments):len(train_segments)+len(val_segments)], val_segments)
test_meta  = prepare_meta_arrays(records[-len(test_segments):], test_segments)

print("Metadata shapes - Train:", train_meta.shape, "Val:", val_meta.shape, "Test:", test_meta.shape)

# --- Dataset Class ---
class RichECGDataset(Dataset):
    def __init__(self, segments, meta_features, input_leads, target_lead, use_meta=True):
        self.samples = []
        self.use_meta = use_meta
        for idx, seg in enumerate(segments):
            x_seg = np.stack([seg[:, ALL_LEADS.index(ld)] for ld in input_leads], axis=1)  # (128, C)
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

# --- CNN1D Model ---
class CNN1D(nn.Module):
    def __init__(self, in_channels=3, seq_len=128, output_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.1),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),
            nn.Flatten(),
            nn.Linear(64*128, 256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        out = self.head(x)
        return out

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
def collect_predictions(memmap_dir, dataset, mlp, xgb_models, cnn, device, prefix):
    """
    Robust prediction collector that accepts either:
      - dataset where X is flattened (use_meta=True) -> xb.dim() == 2
      - dataset where X is (T, C) (use_meta=False) -> xb.dim() == 3

    It reconstructs the CNN input from the flattened vector when needed,
    and pads zeros for meta if a model (MLP/XGB) expects meta features but input
    lacks them.
    """
    memmap_dir = Path(memmap_dir)
    memmap_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, pin_memory=True)

    n_samples = len(dataset)
    seq_len = int(dataset[0][1].shape[0])
    n_meta_models = 3  # MLP, XGB, CNN predictions per timepoint

    meta_X = np.memmap(memmap_dir / f"{prefix}_metaX.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len, n_meta_models))
    meta_y = np.memmap(memmap_dir / f"{prefix}_metaY.dat", dtype="float32",
                       mode="w+", shape=(n_samples, seq_len))

    mlp.eval()
    cnn.eval()

    # helper info
    seg_flat_len = SEGMENT_LENGTH * len(INPUT_LEADS)

    # find expected input dims for MLP and XGB (if available)
    try:
        mlp_input_dim = mlp.net[0].in_features
    except Exception:
        # fallback: infer from a sample flattened x if possible
        mlp_input_dim = None

    xgb_input_dim = None
    try:
        # xgb_models is a list of per-timepoint models; they should all have n_features_in_
        if len(xgb_models) > 0 and hasattr(xgb_models[0], "n_features_in_"):
            xgb_input_dim = int(xgb_models[0].n_features_in_)
    except Exception:
        xgb_input_dim = None

    idx = 0
    with torch.no_grad():
        for xb, yb in tqdm(loader, desc=f"Collecting {prefix} predictions"):
            bsz = xb.shape[0]

            # ---------- CASE A: flattened input (use_meta=True) -> xb.dim() == 2 ----------
            if xb.dim() == 2:
                # MLP: use full flattened vector (includes meta)
                mlp_in = xb.to(device)  # shape (B, input_dim_expected_by_mlp)
                mlp_out = mlp(mlp_in.view(bsz, -1)).cpu().numpy()  # (B, 128)

                # XGB: flatten to numpy (same format used at training time)
                Xb_flat = xb.cpu().numpy().reshape(bsz, -1)  # (B, input_dim_for_xgb)
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)  # (B, 128)

                # CNN: reconstruct segment part from the first seg_flat_len values
                if xb.size(1) < seg_flat_len:
                    raise RuntimeError(f"collect_predictions: flattened input len {xb.size(1)} < expected segment flat length {seg_flat_len}")
                seg_part = xb[:, :seg_flat_len].reshape(bsz, SEGMENT_LENGTH, len(INPUT_LEADS))
                cnn_in = seg_part.to(device)  # (B, T, C)
                cnn_out = cnn(cnn_in).cpu().numpy()  # (B, 128)

            # ---------- CASE B: structured input (use_meta=False) -> xb.dim() == 3 ----------
            elif xb.dim() == 3:
                # xb shape is (B, T, C)
                # Prepare seg_flat for mlp/xgb by flattening
                seg_flat = xb.reshape(bsz, -1)  # (B, seg_flat_len)

                # --- MLP ---
                if mlp_input_dim is None:
                    # we don't know expected mlp input dim -> try to use seg_flat only
                    mlp_in_tensor = seg_flat.to(device)
                else:
                    # pad zeros for missing meta part
                    if seg_flat.shape[1] > mlp_input_dim:
                        raise RuntimeError(f"collect_predictions: segment flat size {seg_flat.shape[1]} > mlp expected input {mlp_input_dim}")
                    pad_len = mlp_input_dim - seg_flat.shape[1]
                    if pad_len > 0:
                        pad_tensor = torch.zeros((bsz, pad_len), dtype=seg_flat.dtype)
                        mlp_in_tensor = torch.cat([seg_flat, pad_tensor], dim=1).to(device)
                    else:
                        mlp_in_tensor = seg_flat.to(device)
                mlp_out = mlp(mlp_in_tensor.view(bsz, -1)).cpu().numpy()

                # --- XGB ---
                seg_flat_np = seg_flat.cpu().numpy()
                if xgb_input_dim is None:
                    Xb_flat = seg_flat_np
                else:
                    if seg_flat_np.shape[1] > xgb_input_dim:
                        raise RuntimeError(f"collect_predictions: segment flat size {seg_flat_np.shape[1]} > xgb expected input {xgb_input_dim}")
                    pad_cols = xgb_input_dim - seg_flat_np.shape[1]
                    if pad_cols > 0:
                        Xb_flat = np.concatenate([seg_flat_np, np.zeros((bsz, pad_cols), dtype=np.float32)], axis=1)
                    else:
                        Xb_flat = seg_flat_np
                xgb_out = xgb_predict_full(xgb_models, Xb_flat)

                # --- CNN ---
                cnn_out = cnn(xb.to(device)).cpu().numpy()

            else:
                raise RuntimeError(f"collect_predictions: unsupported xb.dim() = {xb.dim()}")

            # write to memmap
            meta_X[idx:idx+bsz, :, 0] = mlp_out
            meta_X[idx:idx+bsz, :, 1] = xgb_out
            meta_X[idx:idx+bsz, :, 2] = cnn_out
            meta_y[idx:idx+bsz] = yb.cpu().numpy()
            idx += bsz

    # finalise memmaps and return read-only views
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

        cnn_train_ds = RichECGDataset(train_segments, train_meta, INPUT_LEADS, lead, use_meta=False)
        cnn_val_ds   = RichECGDataset(val_segments, val_meta, INPUT_LEADS, lead, use_meta=False)

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

        # --- CNN training ---
        cnn_model = CNN1D(in_channels=len(INPUT_LEADS), seq_len=SEGMENT_LENGTH, output_dim=SEGMENT_LENGTH).to(DEVICE)
        cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=LEARNING_RATE)
        cnn_loss_fn = nn.MSELoss()
        cnn_best_val = float("inf")
        cnn_counter = 0
        cnn_best_state = None

        cnn_train_loader = DataLoader(cnn_train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        cnn_val_loader = DataLoader(cnn_val_ds, batch_size=BATCH_SIZE, drop_last=True)

        for epoch in range(EPOCHS):
            cnn_model.train()
            total_loss = 0.0
            for xb, yb in tqdm(cnn_train_loader, leave=False):
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                pred = cnn_model(xb)
                loss = cnn_loss_fn(pred, yb)
                cnn_optimizer.zero_grad()
                loss.backward()
                cnn_optimizer.step()
                total_loss += loss.item() * xb.size(0)
            avg_train_loss = total_loss / len(cnn_train_ds)

            cnn_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in cnn_val_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    pred = cnn_model(xb)
                    val_loss += cnn_loss_fn(pred, yb).item() * xb.size(0)
            avg_val_loss = val_loss / len(cnn_val_ds)

            print(f"[CNN] Epoch {epoch+1} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")

            if avg_val_loss < cnn_best_val:
                cnn_best_val = avg_val_loss
                cnn_best_state = cnn_model.state_dict()
                cnn_counter = 0
            else:
                cnn_counter += 1
                if cnn_counter >= patience:
                    print(f"⏹ [CNN] Early stopping at epoch {epoch+1}")
                    break

        if cnn_best_state is not None:
            cnn_model.load_state_dict(cnn_best_state)

        torch.save(cnn_model.state_dict(), MODELS_DIR / f"cnn1d_model_{lead}.pt")

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
        train_X, train_y = collect_predictions("cache_preds", train_ds, mlp, xgb_model, cnn_model, DEVICE, "train")
        val_X, val_y = collect_predictions("cache_preds", val_ds, mlp, xgb_model, cnn_model, DEVICE, "val")
        test_X, test_y = collect_predictions("cache_preds", test_ds, mlp, xgb_model, cnn_model, DEVICE, "test")

        meta_X_train = train_X.reshape(-1, 3)
        meta_y_train = train_y.reshape(-1)
        meta_X_test = test_X.reshape(-1, 3)
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

        # --- Plot RMSE per point ---
        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()
