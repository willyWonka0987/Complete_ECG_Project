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
import matplotlib.pyplot as plt
import os

# --- Config ---
SAVE_DIR = Path("RichECG_Datasets1")   # where segments.npy lives
MODELS_INPUT_DIR = Path("Stacked_Model_Results/models")  # trained models directory

OUTPUT_DIR = Path("star")
PLOTS_DIR = OUTPUT_DIR / "plots"
RMSE_PLOTS_DIR = PLOTS_DIR / "rmse_per_point"
for d in [OUTPUT_DIR, PLOTS_DIR, RMSE_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

REPORT_FILE = OUTPUT_DIR / "evaluation_report.txt"

INPUT_LEADS = ["I", "II", "V2", "V6"]
TARGET_LEADS = ["III", "aVL", "aVR", "aVF", "V1", "V3", "V4", "V5"]
ALL_LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]

SEGMENT_LENGTH = 128
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Dataset (builds dataset directly from segments.npy) ---
class RichECGDataset(Dataset):
    def __init__(self, segments_path, target_lead):
        self.samples = []
        segments = np.load(segments_path)  # expected shape (N, SEGMENT_LENGTH, 12)
        if segments.ndim != 3:
            raise ValueError(f"segments.npy must be 3D. got shape {segments.shape}")
        num_segments = segments.shape[0]
        for i in range(num_segments):
            # prepare input by concatenating each INPUT_LEAD's segment (flattened)
            try:
                full_segment_inputs = []
                for lead in INPUT_LEADS:
                    lead_index = ALL_LEADS.index(lead)
                    full_segment_inputs.append(segments[i, :SEGMENT_LENGTH, lead_index])
                x = np.concatenate(full_segment_inputs).astype(np.float32)

                lead_index = ALL_LEADS.index(target_lead)
                y = segments[i, :SEGMENT_LENGTH, lead_index].astype(np.float32)

                if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                    self.samples.append((x, y))
            except ValueError:
                # if a lead is not found in ALL_LEADS, skip (but this shouldn't happen)
                continue

        print(f"{Path(segments_path).name} ({target_lead}) - samples: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# --- MLP model architecture (must match the one used when saving weights) ---
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


# --- SSIM helper (robust to constant signals) ---
def compute_ssim_batch(y_true, y_pred):
    scores = []
    for i in range(y_true.shape[0]):
        gt = y_true[i]
        pr = y_pred[i]
        data_range = float(gt.max() - gt.min())
        if data_range == 0:
            # if both constant and identical, ssim ~ 1. else 0
            scores.append(1.0 if np.allclose(gt, pr) else 0.0)
        else:
            scores.append(float(ssim(gt, pr, data_range=data_range)))
    return float(np.mean(scores))


# --- Evaluate all leads using saved models ---
with open(REPORT_FILE, "w") as report:
    for lead in TARGET_LEADS:
        print(f"\n📌 Evaluating lead: {lead}")

        # create test dataset from segments.npy (full dataset)
        segments_path = SAVE_DIR / "segments.npy"
        if not segments_path.exists():
            print(f"⚠️ segments file not found at {segments_path}. Skipping.")
            continue

        test_ds = RichECGDataset(segments_path, lead)
        if len(test_ds) == 0:
            print(f"⚠️ No samples for lead {lead}, skipping.")
            continue

        test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

        # input dim expected by the saved MLP
        input_dim = len(test_ds[0][0])  # SEGMENT_LENGTH * len(INPUT_LEADS)
        mlp = MLP(input_dim=input_dim, output_dim=SEGMENT_LENGTH).to(DEVICE)

        mlp_path = MODELS_INPUT_DIR / f"mlp_model_{lead}.pt"
        xgb_path = MODELS_INPUT_DIR / f"xgb_model_{lead}.pkl"
        ridge_path = MODELS_INPUT_DIR / f"ridge_model_{lead}.pkl"

        if not mlp_path.exists():
            print(f"⚠️ Missing MLP file {mlp_path}. Skipping lead {lead}.")
            continue
        if not xgb_path.exists() or not ridge_path.exists():
            print(f"⚠️ Missing xgb/ridge model for lead {lead} in {MODELS_INPUT_DIR}. Skipping.")
            continue

        # load models
        try:
            mlp.load_state_dict(torch.load(mlp_path, map_location=DEVICE))
            mlp.eval()
        except Exception as e:
            print(f"Error loading MLP for {lead}: {e}")
            continue

        try:
            with open(xgb_path, "rb") as f:
                xgb_model = pickle.load(f)
        except Exception as e:
            print(f"Error loading XGB for {lead}: {e}")
            continue

        try:
            with open(ridge_path, "rb") as f:
                ridge_model = pickle.load(f)
        except Exception as e:
            print(f"Error loading Ridge for {lead}: {e}")
            continue

        # collect predictions
        ys_list = []
        preds_list = []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb_device = xb.to(DEVICE)                            # shape (batch, input_dim)
                mlp_out = mlp(xb_device).cpu().numpy()              # (batch, SEGMENT_LENGTH)

                xgb_in = xb.cpu().numpy()                           # (batch, input_dim)
                # xgb_model should return shape (batch, SEGMENT_LENGTH)
                xgb_out = xgb_model.predict(xgb_in)                 # (batch, SEGMENT_LENGTH)

                # meta input: [mlp_out | xgb_out]  => shape (batch, 2*SEGMENT_LENGTH)
                meta_input = np.hstack([mlp_out, xgb_out])

                meta_pred = ridge_model.predict(meta_input)         # (batch, SEGMENT_LENGTH)

                ys_list.append(yb.numpy())
                preds_list.append(meta_pred)

        ys_all = np.vstack(ys_list)
        preds_all = np.vstack(preds_list)

        # Metrics (robust)
        rmse = float(np.sqrt(mean_squared_error(ys_all, preds_all)))
        r2 = float(r2_score(ys_all, preds_all))

        # Pearson average across timepoints (ignore constant timepoints)
        pearson_vals = []
        for t in range(SEGMENT_LENGTH):
            if np.std(ys_all[:, t]) > 0 and np.std(preds_all[:, t]) > 0:
                try:
                    pearson_vals.append(pearsonr(ys_all[:, t], preds_all[:, t])[0])
                except Exception:
                    pearson_vals.append(np.nan)
        pearson_corr = float(np.nanmean(pearson_vals)) if len(pearson_vals) > 0 else float("nan")

        ssim_score = compute_ssim_batch(ys_all, preds_all)

        # cosine similarity per sample
        cos_vals = []
        for i in range(ys_all.shape[0]):
            a = ys_all[i]
            b = preds_all[i]
            na = np.linalg.norm(a)
            nb = np.linalg.norm(b)
            if na == 0 or nb == 0:
                cos_vals.append(np.nan)
            else:
                cos_vals.append(float(np.dot(a, b) / (na * nb)))
        cosine_sim = float(np.nanmean([c for c in cos_vals if not np.isnan(c)])) if len(cos_vals) > 0 else float("nan")

        # print + save
        print(f"  RMSE   = {rmse:.6f}")
        print(f"  R^2    = {r2:.6f}")
        print(f"  Pearson= {pearson_corr:.6f}")
        print(f"  SSIM   = {ssim_score:.6f}")
        print(f"  CosSim = {cosine_sim:.6f}")

        report.write(f"\nLead {lead}:\n")
        report.write(f"RMSE: {rmse:.6f}\n")
        report.write(f"R^2: {r2:.6f}\n")
        report.write(f"Pearson Corr: {pearson_corr:.6f}\n")
        report.write(f"SSIM: {ssim_score:.6f}\n")
        report.write(f"Cosine Sim: {cosine_sim:.6f}\n")

        # RMSE per point plot
        rmse_per_point = np.sqrt(np.mean((ys_all - preds_all) ** 2, axis=0))
        plt.figure(figsize=(10, 4))
        plt.plot(rmse_per_point, marker='o')
        plt.title(f"RMSE per point - Lead {lead}")
        plt.xlabel("Point index")
        plt.ylabel("RMSE")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(RMSE_PLOTS_DIR / f"rmse_per_point_{lead}.png")
        plt.close()

        # sample prediction plots (first up to 10 samples)
        n_samples_to_plot = min(10, ys_all.shape[0])
        for i in range(n_samples_to_plot):
            plt.figure(figsize=(8, 4))
            plt.plot(ys_all[i], label="True", linewidth=2)
            plt.plot(preds_all[i], label="Predicted", linestyle="--")
            plt.title(f"Lead {lead} - Sample {i}")
            plt.legend()
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"{lead}_sample_{i}.png")
            plt.close()

print("\n✅ Evaluation finished. Results saved under", OUTPUT_DIR)
