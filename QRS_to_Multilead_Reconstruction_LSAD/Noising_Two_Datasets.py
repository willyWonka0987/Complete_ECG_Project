import os
import ast
import numpy as np
import pandas as pd
import wfdb

from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.signal import butter, filtfilt, iirnotch, medfilt, resample, welch
from scipy.stats import skew, kurtosis

# ===============================
# CONFIG
# ===============================

# ---- Common
LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
TARGET_LENGTH = 1000            # samples after (optional) resampling
OUT_ROOT = Path('./Cleaned_Raw_Datasets')
PTBXL_OUT_DIR = OUT_ROOT / 'ptbxl_clean'
PTBXL_RAW_DIR = OUT_ROOT / 'ptbxl_raw'   # <-- new: store pre-filtered (raw) trimmed signals
LSAD_OUT_DIR  = OUT_ROOT / 'lsad_clean'
LSAD_RAW_DIR  = OUT_ROOT / 'lsad_raw'    # <-- new: store pre-filtered (raw) trimmed signals
MERGE_META_CSV = OUT_ROOT / 'merged_index.csv'
MERGE_MEMMAP_PATH = OUT_ROOT / 'merged_ecg_signals.dat'
MERGED_NPY = OUT_ROOT / 'merged_ecg_signals.npy'
MERGED_FEATURES_CSV = OUT_ROOT / 'merged_features.csv'

for d in [OUT_ROOT, PTBXL_OUT_DIR, LSAD_OUT_DIR, PTBXL_RAW_DIR, LSAD_RAW_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ---- PTB-XL ----
PTBXL_PATH = Path('../../ptbxl')
DATABASE_CSV = PTBXL_PATH / 'ptbxl_database.csv'
SCP_STATEMENTS_CSV = PTBXL_PATH / 'scp_statements.csv'
PTBXL_FS = 100.0
PTBXL_USE_LR = True

# ---- LSAD ----
LSAD_ROOT = Path('../../dataset_new')
RECORDS_TXT = LSAD_ROOT / 'RECORDS'
LSAD_FS_ORIG = 500.0
LSAD_FS_NEW  = 100.0

# ===============================
# FILTERING & NORMALIZATION UTILS
# ===============================

def butter_bandpass_filter(data: np.ndarray, lowcut: float = 0.5, highcut: float = 45.0, fs: float = 100.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass (filtfilt). Expects shape (T, C)."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def baseline_wander_removal(data: np.ndarray, fs: float = 100.0, window_sec: float = 0.2) -> np.ndarray:
    """Median filter baseline removal. Expects shape (T, C)."""
    window_size = int(window_sec * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(data, kernel_size=(window_size, 1))
    return data - baseline

def notch_filter(data: np.ndarray, fs: float = 100.0, freq: float = 50.0, Q: float = 30.0) -> np.ndarray:
    """
    Apply Notch filter (IIR) to remove powerline interference (50/60Hz).
    Expects shape (T, C).
    freq: تردد الضجيج (50Hz أو 60Hz)
    Q: جودة الفلتر (كلما ارتفعت القيمة ضاق الفلتر أكثر)
    """
    w0 = freq / (fs / 2)  # Normalize freq
    b, a = iirnotch(w0, Q)
    return filtfilt(b, a, data, axis=0)


def pad_or_trim(ecg: np.ndarray, targ_len: int = TARGET_LENGTH) -> np.ndarray:
    if ecg.shape[0] > targ_len:
        return ecg[:targ_len, :]
    elif ecg.shape[0] < targ_len:
        return np.pad(ecg, ((0, targ_len - ecg.shape[0]), (0, 0)), 'constant')
    return ecg


def normalize_ecg(data: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Apply per-lead z-score normalization:
      (x - mean) / std
    Keeps ECG morphology while unifying scale across datasets.
    """
    mean = np.mean(data, axis=0, keepdims=True)
    std = np.std(data, axis=0, keepdims=True) + eps
    return (data - mean) / std


def compute_snr(raw: np.ndarray, filtered: np.ndarray) -> float:
    """احسب SNR بين raw (قبل الفلاتر) و filtered (بعد bandpass)."""
    noise = raw - filtered
    p_signal = np.mean(filtered ** 2)
    p_noise = np.mean(noise ** 2) + 1e-8  # لتفادي القسمة على صفر
    snr_db = 10 * np.log10(p_signal / p_noise)
    return snr_db


def is_signal_valid(raw: np.ndarray, fs: float, snr_threshold: float = 5.0) -> bool:
    """رجع True إذا الإشارة صالحة حسب SNR."""
    filtered = butter_bandpass_filter(raw, fs=fs)
    snr = compute_snr(raw, filtered)
    return snr >= snr_threshold


def clean_ecg_pipeline(arr: np.ndarray, fs: float) -> np.ndarray:
    """Apply notch + bandpass + baseline removal + pad/trim, then normalization."""
    # arr = notch_filter(arr, fs=fs, freq=50.0, Q=30.0)   # <-- جديد (ممكن تفعيله إذا احتجت)
    arr = butter_bandpass_filter(arr, fs=fs)
    arr = baseline_wander_removal(arr, fs=fs)
    arr = pad_or_trim(arr, targ_len=TARGET_LENGTH)
    # arr = normalize_ecg(arr)   # <-- Normalization step (اختياري)
    return arr.astype(np.float32)


def downsample_to(arr: np.ndarray, orig_fs: float, new_fs: float) -> np.ndarray:
    """Resample along time axis to new_fs. Expects shape (T, C)."""
    n_samples_new = int(round(arr.shape[0] * new_fs / orig_fs))
    return resample(arr, n_samples_new, axis=0)

# ===============================
# FEATURE EXTRACTION
# ===============================

def _compute_bandpower_from_psd(f, pxx, low, high):
    mask = (f >= low) & (f <= high)
    if not np.any(mask):
        return 0.0
    return np.trapz(pxx[mask], f[mask])


def extract_features_from_signal(sig: np.ndarray, fs: float) -> dict:
    """Extract per-lead statistical and frequency-domain features.
    Returns a flat dict of features keyed by <LEAD>_<feat>.
    Expects sig shape (T, C) with C == len(LEAD_NAMES) ideally.
    """
    feats = {}
    T, C = sig.shape
    # ensure shape consistency
    if C < len(LEAD_NAMES):
        # pad with zeros
        pad = np.zeros((T, len(LEAD_NAMES) - C), dtype=sig.dtype)
        sig = np.concatenate([sig, pad], axis=1)
    elif C > len(LEAD_NAMES):
        sig = sig[:, :len(LEAD_NAMES)]

    # PSD params
    nperseg = min(1024, max(256, T))

    bands = {
        'bp_0_5_4': (0.5, 4.0),
        'bp_4_15': (4.0, 15.0),
        'bp_15_40': (15.0, 40.0),
    }

    for i, lead in enumerate(LEAD_NAMES):
        x = sig[:, i].astype(np.float64)
        # basic stats
        feats[f"{lead}_mean"] = float(np.nanmean(x))
        feats[f"{lead}_std"] = float(np.nanstd(x))
        # skew/kurt: fall back to 0 for constant signals
        try:
            feats[f"{lead}_skew"] = float(skew(x))
        except Exception:
            feats[f"{lead}_skew"] = 0.0
        try:
            feats[f"{lead}_kurtosis"] = float(kurtosis(x))
        except Exception:
            feats[f"{lead}_kurtosis"] = 0.0
        feats[f"{lead}_min"] = float(np.nanmin(x))
        feats[f"{lead}_max"] = float(np.nanmax(x))
        feats[f"{lead}_median"] = float(np.nanmedian(x))
        feats[f"{lead}_rms"] = float(np.sqrt(np.nanmean(x ** 2)))
        feats[f"{lead}_energy"] = float(np.nansum(x ** 2))
        # zero-crossing rate (normalized)
        if len(x) > 1:
            zc = np.sum(np.abs(np.diff(np.sign(x)))) / (2.0 * (len(x) - 1))
            feats[f"{lead}_zcr"] = float(zc)
        else:
            feats[f"{lead}_zcr"] = 0.0

        # PSD / freq-domain
        try:
            f, pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
            # dominant frequency
            idx_max = np.nanargmax(pxx)
            feats[f"{lead}_dom_freq"] = float(f[idx_max])
            feats[f"{lead}_psd_total"] = float(np.trapz(pxx, f))
            # band powers
            for bname, (low, high) in bands.items():
                bp = _compute_bandpower_from_psd(f, pxx, low, high)
                feats[f"{lead}_{bname}"] = float(bp)
        except Exception:
            feats[f"{lead}_dom_freq"] = 0.0
            feats[f"{lead}_psd_total"] = 0.0
            for bname in bands.keys():
                feats[f"{lead}_{bname}"] = 0.0

    return feats

# ===============================
# PTB-XL LOADING & CLEANING
# ===============================

def _read_ptbxl_record(row: pd.Series) -> np.ndarray:
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


def _process_ptbxl_row(idx: int, row: pd.Series, out_dir: Path) -> dict:
    try:
        sig = _read_ptbxl_record(row)

        # compute SNR on raw vs bandpass (using full-length read signal)
        filtered_for_snr = butter_bandpass_filter(sig, fs=PTBXL_FS)
        snr_db = compute_snr(sig, filtered_for_snr)

        # --- NEW: تحقق من SNR ---
        if snr_db < 5.0:
            return None

        # create record id early so we can save raw + clean with same id
        rec_id = f"ptbxl_{idx:07d}"

        # --- SAVE RAW (pre-filter) VERSION ---
        # Save a version of the signal trimmed/padded to TARGET_LENGTH but BEFORE filtering
        raw_sig = pad_or_trim(sig, targ_len=TARGET_LENGTH).astype(np.float32)
        raw_out_path = PTBXL_RAW_DIR / f"{rec_id}.npy"
        np.save(raw_out_path, raw_sig)

        # now apply the cleaning pipeline (bandpass + baseline + pad/trim)
        sig_clean = clean_ecg_pipeline(sig, fs=PTBXL_FS)

        # features (from cleaned)
        feats = extract_features_from_signal(sig_clean, fs=PTBXL_FS)

        out_path = out_dir / f"{rec_id}.npy"
        np.save(out_path, sig_clean)

        # age/sex extraction (from row)
        age = None
        sex = None
        if 'age' in row.index:
            try:
                age = int(row['age'])
            except Exception:
                age = row['age']
        if 'sex' in row.index:
            # PTB-XL encodes sex sometimes as 1/0 or 'M'/'F'
            s = row['sex']
            if pd.isna(s):
                sex = None
            else:
                try:
                    sex = 'Male' if int(s) == 1 else 'Female'
                except Exception:
                    sex = str(s)

        # diagnosis (attempt to fetch scp_codes or similar)
        diagnosis = None
        for key in ['scp_codes', 'scps', 'diagnostic', 'diagnoses']:
            if key in row.index:
                diagnosis = row[key]
                break

        meta = {
            'id': rec_id,
            'src': 'PTBXL',
            'path': str(out_path.resolve()),
            'raw_path': str(raw_out_path.resolve()),  # <-- new: path to raw (pre-filter) file
            'age': age,
            'sex': sex,
            'diagnosis': diagnosis,
            'snr_db': float(snr_db),
        }
        # merge feats into meta
        meta.update(feats)
        return meta
    except Exception as e:
        print(f"PTB-XL error @ row {idx}: {e}")
        return None


def run_ptbxl_cleaning():
    print("\n=== PTB-XL: Loading metadata and filtering noisy rows ===")
    df = pd.read_csv(DATABASE_CSV)
    print("Original shape:", df.shape)

    df = df[(df['validated_by_human'] == True) & (df['filename_lr'].notnull())]

    for col in ['electrodes_problems', 'pacemaker', 'burst_noise', 'static_noise']:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    df = df[
        (df['electrodes_problems'] == 0) &
        (df['pacemaker'] == 0) &
        (df['burst_noise'] == 0) &
        (df['static_noise'] == 0)
    ]
    print("After noise filtering:", df.shape)

    print("Cleaning PTB-XL signals …")
    results = Parallel(n_jobs=os.cpu_count() or 8, prefer="processes")(
        delayed(_process_ptbxl_row)(i, row, PTBXL_OUT_DIR)
        for i, (idx, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="PTB-XL"))
    )

    records = [r for r in results if r is not None]
    n_ok = len(records)
    n_rejected = len(results) - n_ok
    print(f"PTB-XL cleaned & saved: {n_ok} files -> {PTBXL_OUT_DIR}")
    print(f"PTB-XL raw (pre-filter) saved -> {PTBXL_RAW_DIR}")
    print(f"PTB-XL rejected (low SNR or errors): {n_rejected}")

    # save PTBXL metadata/features
    if records:
        df_meta = pd.DataFrame(records)
        df_meta.to_csv(PTBXL_OUT_DIR / 'ptbxl_metadata.csv', index=False)
        print(f"PTB-XL metadata/features saved -> {PTBXL_OUT_DIR / 'ptbxl_metadata.csv'}")
    return records

# ===============================
# LSAD LOADING & CLEANING
# ===============================

def _read_age_sex_from_heafile(hea_path: Path):
    age, sex, diagnosis = None, None, None
    try:
        with open(hea_path, 'r') as f:
            for line in f:
                if line.startswith('#Age:'):
                    age_text = line.strip().split(':', 1)[1].strip()
                    try:
                        age = int(age_text)
                    except:
                        age = age_text
                elif line.startswith('#Sex:'):
                    sex = line.strip().split(':', 1)[1].strip()
                elif line.startswith('#Dx:'):
                    diagnosis = line.strip().split(':', 1)[1].strip()
    except Exception:
        pass
    return age, sex, diagnosis


def _process_lsad_record(idx: int, hea_file: Path, out_dir: Path) -> dict:
    try:
        base = hea_file.with_suffix('')
        rec = wfdb.rdrecord(str(base))
        sig = rec.p_signal
        if sig.shape[1] < 12:
            return None
        sig = sig[:, :12].astype(np.float32)

        # Keep first 5000 samples (500 Hz)
        if sig.shape[0] >= 5000:
            sig = sig[:5000, :]

        sig_down = downsample_to(sig, orig_fs=LSAD_FS_ORIG, new_fs=LSAD_FS_NEW)

        # compute SNR on raw vs bandpass
        filtered_for_snr = butter_bandpass_filter(sig_down, fs=LSAD_FS_NEW)
        snr_db = compute_snr(sig_down, filtered_for_snr)

        # --- NEW: تحقق من SNR ---
        if snr_db < 5.0:
            return None

        # create rec id so we can save raw + clean versions with same id
        rec_id = f"lsad_{idx:07d}"

        # --- SAVE RAW (pre-filter) VERSION ---
        raw_sig = pad_or_trim(sig_down, targ_len=TARGET_LENGTH).astype(np.float32)
        raw_out_path = LSAD_RAW_DIR / f"{rec_id}.npy"
        np.save(raw_out_path, raw_sig)

        sig_clean = clean_ecg_pipeline(sig_down, fs=LSAD_FS_NEW)

        # features
        feats = extract_features_from_signal(sig_clean, fs=LSAD_FS_NEW)

        out_path = out_dir / f"{rec_id}.npy"
        np.save(out_path, sig_clean)

        # read age/sex/diagnosis from hea file
        age, sex, diagnosis = _read_age_sex_from_heafile(hea_file)

        meta = {
            'id': rec_id,
            'src': 'LSAD',
            'path': str(out_path.resolve()),
            'raw_path': str(raw_out_path.resolve()),  # <-- new: path to raw (pre-filter) file
            'age': age,
            'sex': sex,
            'diagnosis': diagnosis,
            'snr_db': float(snr_db),
        }
        meta.update(feats)
        return meta
    except Exception as e:
        print(f"LSAD error @ {hea_file}: {e}")
        return None


def run_lsad_cleaning():
    print("\n=== LSAD: Enumerating records from RECORDS file ===")
    with open(RECORDS_TXT, 'r') as f:
        record_paths = [line.strip() for line in f if line.strip()]

    hea_files = []
    for rel in record_paths:
        record_dir = LSAD_ROOT / rel
        hea_files.extend(list(Path(record_dir).glob('*.hea')))

    print(f"Found {len(hea_files)} header files (.hea)")

    print("Cleaning LSAD signals …")
    results = Parallel(n_jobs=os.cpu_count() or 8, prefer="processes")(
        delayed(_process_lsad_record)(i, hea, LSAD_OUT_DIR)
        for i, hea in enumerate(tqdm(hea_files, total=len(hea_files), desc="LSAD"))
    )

    records = [r for r in results if r is not None]
    n_ok = len(records)
    n_rejected = len(results) - n_ok
    print(f"LSAD cleaned & saved: {n_ok} files -> {LSAD_OUT_DIR}")
    print(f"LSAD raw (pre-filter) saved -> {LSAD_RAW_DIR}")
    print(f"LSAD rejected (low SNR or errors): {n_rejected}")

    # save LSAD metadata/features
    if records:
        df_meta = pd.DataFrame(records)
        df_meta.to_csv(LSAD_OUT_DIR / 'lsad_metadata.csv', index=False)
        print(f"LSAD metadata/features saved -> {LSAD_OUT_DIR / 'lsad_metadata.csv'}")
    return records

# ===============================
# MERGING
# ===============================

def _collect_npy_files() -> pd.DataFrame:
    rows = []
    for src, d in [('PTBXL', PTBXL_OUT_DIR), ('LSAD', LSAD_OUT_DIR)]:
        for p in sorted(d.glob('*.npy')):
            rows.append({'src': src, 'path': str(p.resolve())})
    df = pd.DataFrame(rows)
    return df


def build_merged_memmap():
    df = _collect_npy_files()
    if df.empty:
        print("No cleaned files found to merge. Run cleaners first.")
        return

    print(f"\nMerging {len(df)} signals into a single memmap array …")
    MERGE_META_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MERGE_META_CSV, index=False)

    N, T, C = len(df), TARGET_LENGTH, 12
    if MERGE_MEMMAP_PATH.exists():
        MERGE_MEMMAP_PATH.unlink()

    mm = np.memmap(MERGE_MEMMAP_PATH, dtype='float32', mode='w+', shape=(N, T, C))

    for i, row in enumerate(tqdm(df.itertuples(index=False), total=N, desc='Merging')):
        arr = np.load(row.path)
        if arr.shape != (T, C):
            arr = pad_or_trim(arr, targ_len=T)
            if arr.shape[1] != C:
                if arr.shape[1] < C:
                    pad = np.zeros((arr.shape[0], C - arr.shape[1]), dtype=arr.dtype)
                    arr = np.concatenate([arr, pad], axis=1)
                else:
                    arr = arr[:, :C]
        mm[i] = arr.astype(np.float32)

    del mm
    print(f"Memmap written to {MERGE_MEMMAP_PATH}")

    print("Saving a .npy copy of the merged memmap (optional)…")
    mm_read = np.memmap(MERGE_MEMMAP_PATH, dtype='float32', mode='r', shape=(N, T, C))
    np.save(MERGED_NPY, np.array(mm_read))
    del mm_read
    print(f"Merged .npy saved -> {MERGED_NPY}")

# ===============================
# MAIN
# ===============================
if __name__ == '__main__':
    ptbxl_records = run_ptbxl_cleaning()
    lsad_records = run_lsad_cleaning()

    # combine metadata/features from both sources into a single CSV
    all_records = []
    if ptbxl_records:
        all_records.extend(ptbxl_records)
    if lsad_records:
        all_records.extend(lsad_records)

    if all_records:
        df_all = pd.DataFrame(all_records)
        df_all.to_csv(MERGED_FEATURES_CSV, index=False)
        print(f"Merged metadata & features saved -> {MERGED_FEATURES_CSV}")

    build_merged_memmap()

    print("\n✅ Done. You now have:")
    print(f" - PTB-XL cleaned signals in: {PTBXL_OUT_DIR}")
    print(f" - PTB-XL raw (pre-filter) signals in: {PTBXL_RAW_DIR}")
    print(f" - LSAD cleaned signals in:  {LSAD_OUT_DIR}")
    print(f" - LSAD raw (pre-filter) signals in:  {LSAD_RAW_DIR}")
    print(f" - Merge index CSV:           {MERGE_META_CSV}")
    print(f" - Merged memmap:             {MERGE_MEMMAP_PATH}")
    print(f" - Merged NPY (optional):     {MERGED_NPY}")
    print(f" - Merged metadata/features:  {MERGED_FEATURES_CSV}")
