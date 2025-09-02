"""
ecg_clean_and_plot.py

(Updated: processed/orange line is semi-transparent so the raw signal below is visible,
 now showing quantitative metrics RMSE, Pearson correlation, and SSIM for each lead.
 Fixed: raw LSAD signals are also downsampled to 100Hz before comparison.)
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import wfdb
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, medfilt, resample, iirnotch
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from skimage.metrics import structural_similarity as ssim

# ------------------ Config ------------------
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_LENGTH = 1000
OUT_ROOT = Path('./Plots_Clean_Vs_Raw')
PTBXL_OUT_DIR = OUT_ROOT / 'ptbxl_clean'
LSAD_OUT_DIR  = OUT_ROOT / 'lsad_clean'
FIG_ROOT = OUT_ROOT / 'figures'
PTBXL_PATH = Path('../../ptbxl')
DATABASE_CSV = PTBXL_PATH / 'ptbxl_database.csv'
PTBXL_USE_LR = True
PTBXL_FS = 100.0
LSAD_ROOT = Path('../../dataset_new')
RECORDS_TXT = LSAD_ROOT / 'RECORDS'
LSAD_FS_ORIG = 500.0
LSAD_FS_NEW  = 100.0

NOTCH_FREQUENCY = 50.0
NOTCH_Q = 30.0

SAMPLES_TO_SAVE = 20
DPI = 300
FIGSIZE = (14, 10)

for d in [OUT_ROOT, PTBXL_OUT_DIR, LSAD_OUT_DIR, FIG_ROOT]:
    d.mkdir(parents=True, exist_ok=True)
(FIG_ROOT / 'ptbxl').mkdir(parents=True, exist_ok=True)
(FIG_ROOT / 'lsad').mkdir(parents=True, exist_ok=True)

# ------------------ Filters ------------------

def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def baseline_wander_removal(data, fs=100.0, window_sec=0.2):
    window_size = int(window_sec * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(data, kernel_size=(window_size, 1))
    return data - baseline

def apply_notch(data, fs, freq=50.0, Q=30.0):
    b, a = iirnotch(w0=freq, Q=Q, fs=fs)
    return filtfilt(b, a, data, axis=0)

def pad_or_trim(ecg, targ_len=TARGET_LENGTH):
    if ecg.shape[0] > targ_len:
        return ecg[:targ_len, :]
    elif ecg.shape[0] < targ_len:
        return np.pad(ecg, ((0, targ_len - ecg.shape[0]), (0, 0)), 'constant')
    return ecg

def downsample_to(arr, orig_fs, new_fs):
    n_samples_new = int(round(arr.shape[0] * new_fs / orig_fs))
    if n_samples_new < 1:
        raise ValueError('Target number of samples < 1')
    return resample(arr, n_samples_new, axis=0)

def clean_ecg_pipeline_no_normalize(arr, fs):
    arr = butter_bandpass_filter(arr, fs=fs)
    arr = baseline_wander_removal(arr, fs=fs)
    arr = pad_or_trim(arr, targ_len=TARGET_LENGTH)
    return arr.astype(np.float32)

# ------------------ PTB-XL loader & processor ------------------

def _read_ptbxl_record(row):
    rel_path = Path(row['filename_lr'] if PTBXL_USE_LR else row['filename_hr'])
    full_path = PTBXL_PATH / rel_path
    record = wfdb.rdrecord(str(full_path))
    sig = record.p_signal.astype(np.float32)
    if sig.shape[1] >= 12:
        sig = sig[:, :12]
    else:
        pad = np.zeros((sig.shape[0], 12 - sig.shape[1]), dtype=np.float32)
        sig = np.concatenate([sig, pad], axis=1)
    return sig

def compute_metrics(raw, proc):
    metrics = []
    for ch in range(12):
        raw_ch = raw[:, ch]
        proc_ch = proc[:, ch]
        # Align lengths
        n = min(len(raw_ch), len(proc_ch))
        raw_ch = raw_ch[:n]
        proc_ch = proc_ch[:n]
        rmse = np.sqrt(np.mean((raw_ch - proc_ch)**2))
        try:
            corr, _ = pearsonr(raw_ch, proc_ch)
        except Exception:
            corr = np.nan
        try:
            ssim_val, _ = ssim(raw_ch.reshape(1, -1), proc_ch.reshape(1, -1), full=True)
        except Exception:
            ssim_val = np.nan
        metrics.append((rmse, corr, ssim_val))
    return metrics

def process_and_plot_signal(raw_sig, fs, out_path, title, apply_notch_flag=True):
    # --- Prepare raw signal ---
    raw_proc = raw_sig.copy()
    if fs != PTBXL_FS and fs is not None:
        raw_proc = downsample_to(raw_proc, orig_fs=fs, new_fs=PTBXL_FS)
        fs = PTBXL_FS
    raw_proc = pad_or_trim(raw_proc, targ_len=TARGET_LENGTH)
    if raw_proc.shape[1] != 12:
        if raw_proc.shape[1] < 12:
            pad = np.zeros((raw_proc.shape[0], 12 - raw_proc.shape[1]), dtype=raw_proc.dtype)
            raw_proc = np.concatenate([raw_proc, pad], axis=1)
        else:
            raw_proc = raw_proc[:, :12]

    # --- Processed version ---
    proc = raw_proc.copy()
    if apply_notch_flag:
        proc = apply_notch(proc, fs=fs, freq=NOTCH_FREQUENCY, Q=NOTCH_Q)
    proc = clean_ecg_pipeline_no_normalize(proc, fs=fs)

    # --- Metrics ---
    metrics = compute_metrics(raw_proc, proc)

    # --- Plot ---
    fig, axes = plt.subplots(3, 4, figsize=FIGSIZE, sharex=False)
    axes = axes.flatten()
    t = np.arange(TARGET_LENGTH) / PTBXL_FS

    for ch in range(12):
        ax = axes[ch]
        ax.plot(t, raw_proc[:, ch], linewidth=0.6, alpha=0.8, label='raw', color='blue')
        ax.plot(t, proc[:, ch], linewidth=0.9, linestyle='-', label='processed', color='orange', alpha=0.6)
        rmse, corr, ssim_val = metrics[ch]
        ax.set_title(f"{LEAD_NAMES[ch]}\nRMSE={rmse:.3f}, r={corr:.3f}, SSIM={ssim_val:.3f}", fontsize=9)
        ax.set_xlim(0, TARGET_LENGTH / PTBXL_FS)
        ax.grid(True, linewidth=0.3)
        if ch == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.suptitle(title)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(out_path, dpi=DPI)
    plt.close(fig)

# ------------------ Run PTB-XL processing ------------------

def run_ptbxl_plots():
    if not DATABASE_CSV.exists():
        print(f"PTB-XL metadata not found at {DATABASE_CSV}. Skipping PTB-XL.")
        return
    df = pd.read_csv(DATABASE_CSV)
    df = df[(df['validated_by_human'] == True) & (df['filename_lr'].notnull())]

    for col in ['electrodes_problems', 'pacemaker', 'burst_noise', 'static_noise']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)
    df = df[(df['electrodes_problems'] == 0) & (df['pacemaker'] == 0) & (df['burst_noise'] == 0) & (df['static_noise'] == 0)]

    saved = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        if saved >= SAMPLES_TO_SAVE:
            break
        try:
            raw = _read_ptbxl_record(row)
            out_file = FIG_ROOT / 'ptbxl' / f'ptbxl_{saved:03d}.png'
            process_and_plot_signal(raw, fs=PTBXL_FS, out_path=out_file, title=f'PTB-XL sample {saved}', apply_notch_flag=True)
            saved += 1
        except Exception as e:
            print(f"Skipping PTB-XL row {i}: {e}")

    print(f"PTB-XL: saved {saved} figure(s) -> {FIG_ROOT / 'ptbxl'}")

# ------------------ Run LSAD processing ------------------

def run_lsad_plots():
    if not RECORDS_TXT.exists():
        print(f"LSAD RECORDS file not found at {RECORDS_TXT}. Skipping LSAD.")
        return
    with open(RECORDS_TXT, 'r') as f:
        record_paths = [line.strip() for line in f if line.strip()]

    hea_files = []
    for rel in record_paths:
        record_dir = LSAD_ROOT / rel
        hea_files.extend(list(Path(record_dir).glob('*.hea')))

    saved = 0
    for i, hea in enumerate(hea_files):
        if saved >= SAMPLES_TO_SAVE:
            break
        try:
            base = hea.with_suffix('')
            rec = wfdb.rdrecord(str(base))
            sig = rec.p_signal
            if sig.shape[1] < 12:
                print(f"LSAD {hea}: less than 12 leads, skipping")
                continue
            sig = sig[:, :12].astype(np.float32)

            if sig.shape[0] >= 5000:
                sig = sig[:5000, :]

            out_file = FIG_ROOT / 'lsad' / f'lsad_{saved:03d}.png'
            process_and_plot_signal(sig, fs=LSAD_FS_ORIG, out_path=out_file, title=f'LSAD sample {saved}', apply_notch_flag=True)
            saved += 1
        except Exception as e:
            print(f"Skipping LSAD hea {hea}: {e}")

    print(f"LSAD: saved {saved} figure(s) -> {FIG_ROOT / 'lsad'}")

# ------------------ MAIN ------------------
if __name__ == '__main__':
    print('Starting processing and plotting...')
    run_ptbxl_plots()
    run_lsad_plots()
    print('\n✅ Done. Figures saved in:')
    print(f" - PTB-XL figures: {FIG_ROOT / 'ptbxl'}")
    print(f" - LSAD figures:   {FIG_ROOT / 'lsad'}")
