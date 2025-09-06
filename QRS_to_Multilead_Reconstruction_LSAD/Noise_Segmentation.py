import os
import numpy as np
import neurokit2 as nk
from pathlib import Path
from tqdm import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
import pickle
import warnings
import glob
import pandas as pd

from scipy.signal import welch
from scipy.stats import skew, kurtosis, entropy as scipy_entropy

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")

# --- Configuration ---
ROOT = Path("./Cleaned_Raw_Datasets")
PTBXL_DIR = ROOT / "ptbxl_clean"
LSAD_DIR = ROOT / "lsad_clean"
PTBXL_Raw_DIR = ROOT / "ptbxl_raw"
LSAD_Raw_DIR = ROOT / "lsad_raw"
MERGED_META_CSV = ROOT / "merged_features.csv"   # optional merged metadata
PTBXL_META_CSV = PTBXL_DIR / "ptbxl_metadata.csv"
LSAD_META_CSV = LSAD_DIR / "lsad_metadata.csv"
SAVE_ROOT = Path("Segments_clean_raw")
SAVE_ROOT.mkdir(parents=True, exist_ok=True)

SAMPLING_RATE = 100
SEGMENT_LENGTH = 128
HALF_WINDOW = SEGMENT_LENGTH // 2
R_WINDOW = 50
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

# --------------------- Helpers ---------------------

def pad_segment_centered(ecg, r_index, target_length):
    half = target_length // 2
    start = int(r_index - half)
    end = int(r_index + half)
    # If ecg is None, return zeros with 12 leads
    if ecg is None:
        return np.zeros((target_length, len(LEAD_NAMES)), dtype=np.float32)
    segment = np.zeros((target_length, ecg.shape[1]), dtype=ecg.dtype)
    src_start = max(0, start)
    src_end = min(ecg.shape[0], end)
    dst_start = src_start - start if start < 0 else 0
    dst_end = dst_start + (src_end - src_start)
    if src_end > src_start:
        segment[dst_start:dst_end] = ecg[src_start:src_end]
    return segment

def extract_rpeaks_from_info(info):
    if info is None:
        return []
    r = None
    if isinstance(info, dict):
        r = info.get("ECG_R_Peaks", None) or info.get("R_Peaks", None)
        if r is None:
            for k, v in info.items():
                if 'r_peak' in str(k).lower() or 'rpeaks' in str(k).lower():
                    r = v
                    break
    else:
        r = info
    if r is None:
        return []
    arr = np.asarray(r)
    if arr.size == 0:
        return []
    unique = np.unique(arr)
    if set(unique.tolist()).issubset({0, 1, True, False}):
        idx = np.where(arr == 1)[0]
        return idx.tolist()
    try:
        return arr.astype(int).tolist()
    except Exception:
        return []

# --------------------- Feature extraction helpers ---------------------

def _safe_float(x):
    try:
        return float(np.asarray(x).item())
    except Exception:
        try:
            return float(np.asarray(x))
        except Exception:
            return None

def compute_time_freq_features(sig, fs=SAMPLING_RATE):
    """
    Compute a compact set of time-domain and frequency-domain features for a 1D signal.
    Returns a dict of scalar features (floats/ints).
    """
    feats = {}
    sig = np.asarray(sig).astype(np.float64)
    if sig.size == 0:
        # fallback zeros
        zero_feats = {
            'mean': 0.0, 'median': 0.0, 'std': 0.0, 'var': 0.0,
            'rms': 0.0, 'ptp': 0.0, 'mad': 0.0, 'skew': 0.0,
            'kurtosis': 0.0, 'zero_crossings': 0, 'energy': 0.0,
            'total_power': 0.0, 'spectral_entropy': 0.0, 'spectral_centroid': 0.0, 'peak_freq': 0.0
        }
        # include band placeholders
        for b in [(0,5),(5,15),(15,40)]:
            zero_feats[f'band_{b[0]}_{b[1]}_power'] = 0.0
        return zero_feats

    feats['mean'] = _safe_float(np.mean(sig))
    feats['median'] = _safe_float(np.median(sig))
    feats['std'] = _safe_float(np.std(sig))
    feats['var'] = _safe_float(np.var(sig))
    feats['rms'] = _safe_float(np.sqrt(np.mean(sig**2)))
    feats['ptp'] = _safe_float(np.ptp(sig))
    feats['mad'] = _safe_float(np.mean(np.abs(sig - np.mean(sig))))
    feats['skew'] = _safe_float(skew(sig))
    feats['kurtosis'] = _safe_float(kurtosis(sig))
    feats['zero_crossings'] = int(np.sum(np.diff(np.sign(sig)) != 0))
    feats['energy'] = _safe_float(np.sum(sig**2))

    # PSD using Welch (robust). nperseg tuned not to exceed length.
    nperseg = min(256, max(8, sig.size))
    try:
        freqs, psd = welch(sig, fs=fs, nperseg=nperseg)
        total_power = _safe_float(np.trapz(psd, freqs))
        feats['total_power'] = total_power
        if psd.sum() > 0:
            pnorm = psd / (psd.sum() + 1e-12)
            feats['spectral_entropy'] = _safe_float(scipy_entropy(pnorm + 1e-12))
        else:
            feats['spectral_entropy'] = 0.0
        feats['spectral_centroid'] = _safe_float(np.sum(freqs * psd) / (np.sum(psd) + 1e-12))
        feats['peak_freq'] = _safe_float(freqs[np.argmax(psd)]) if freqs.size > 0 else 0.0
        # band powers (Hz) - choose bands useful for ECG waveform content
        bands = [(0,5),(5,15),(15,40)]
        for b in bands:
            idx = np.logical_and(freqs >= b[0], freqs <= b[1])
            bp = float(np.trapz(psd[idx], freqs[idx])) if idx.any() else 0.0
            feats[f'band_{b[0]}_{b[1]}_power'] = _safe_float(bp)
    except Exception:
        # fallback safe values
        feats['total_power'] = 0.0
        feats['spectral_entropy'] = 0.0
        feats['spectral_centroid'] = 0.0
        feats['peak_freq'] = 0.0
        for b in [(0,5),(5,15),(15,40)]:
            feats[f'band_{b[0]}_{b[1]}_power'] = 0.0

    return feats

def compute_multichannel_features(arr, fs=SAMPLING_RATE, lead_names=LEAD_NAMES):
    """
    Compute per-lead features and summary (mean/std across leads).
    Returns a flat dict with keys like 'lead_I_mean', 'lead_V1_rms', and
    'across_leads_mean_of_means', etc.
    """
    out = {}
    arr = np.asarray(arr)
    n_samples, n_leads = arr.shape
    for li in range(n_leads):
        lead = lead_names[li] if li < len(lead_names) else f'lead_{li}'
        sig = arr[:, li]
        feats = compute_time_freq_features(sig, fs=fs)
        # prefix each feature with lead name
        for k, v in feats.items():
            out[f'{lead}_{k}'] = v
    # aggregated across leads
    try:
        means = [out[f'{lead_names[i]}_mean'] for i in range(min(len(lead_names), n_leads))]
        out['across_leads_mean_of_means'] = _safe_float(np.mean(means)) if len(means) > 0 else 0.0
        out['across_leads_std_of_means'] = _safe_float(np.std(means)) if len(means) > 0 else 0.0
    except Exception:
        out['across_leads_mean_of_means'] = 0.0
        out['across_leads_std_of_means'] = 0.0
    return out

def load_raw_map(folder: Path):
    """Load all .npy raw files (12-lead) into a dict keyed by stem (filename without ext)."""
    raw_map = {}
    if folder is None or not folder.exists():
        return raw_map
    files = sorted(folder.glob("*.npy"))
    for p in files:
        try:
            arr = np.load(p)
            if arr.ndim == 2 and arr.shape[1] == 12:
                raw_map[p.stem] = arr.astype(np.float32)
        except Exception:
            continue
    return raw_map

# --------------------- Core processing ---------------------

def process_single_record(ecg, record_id, meta=None, raw_ecg=None):
    """
    ecg: cleaned/processed 2D array (T, C)
    raw_ecg: (optional) raw 2D array (T, C) corresponding to the same record (used to extract raw_segment)
    Returns: (records_out, local_segments) where records_out is list of rec_out dicts and local_segments list of clean segments arrays
    """
    if ecg is None:
        return [], []
    if np.any(np.isnan(ecg)) or np.any(np.isinf(ecg)):
        return [], []
    if ecg.ndim != 2:
        return [], []
    T, C = ecg.shape
    if C < len(LEAD_NAMES):
        pad = np.zeros((T, len(LEAD_NAMES) - C), dtype=ecg.dtype)
        ecg = np.concatenate([ecg, pad], axis=1)
    elif C > len(LEAD_NAMES):
        ecg = ecg[:, :len(LEAD_NAMES)]

    # normalize/adjust raw_ecg channels shape if provided
    if raw_ecg is not None:
        try:
            if raw_ecg.ndim != 2:
                raw_ecg = None
            else:
                # pad/truncate channels to match LEAD_NAMES
                Tr, Cr = raw_ecg.shape
                if Cr < len(LEAD_NAMES):
                    padr = np.zeros((Tr, len(LEAD_NAMES) - Cr), dtype=raw_ecg.dtype)
                    raw_ecg = np.concatenate([raw_ecg, padr], axis=1)
                elif Cr > len(LEAD_NAMES):
                    raw_ecg = raw_ecg[:, :len(LEAD_NAMES)]
        except Exception:
            raw_ecg = None

    try:
        rpeaks_all = []
        for lead_idx in range(len(LEAD_NAMES)):
            sig = nk.ecg_clean(ecg[:, lead_idx], sampling_rate=SAMPLING_RATE)
            signals, info = nk.ecg_process(sig, sampling_rate=SAMPLING_RATE)
            rpeaks = None
            if isinstance(info, dict) and "ECG_R_Peaks" in info:
                rpeaks = extract_rpeaks_from_info(info["ECG_R_Peaks"]) if isinstance(info["ECG_R_Peaks"], (list, np.ndarray)) else extract_rpeaks_from_info(info)
            else:
                try:
                    if hasattr(signals, 'columns') and 'ECG_R_Peaks' in signals.columns:
                        rvec = signals['ECG_R_Peaks'].values
                        rpeaks = extract_rpeaks_from_info(rvec)
                except Exception:
                    rpeaks = None
            if rpeaks is None or len(rpeaks) == 0:
                try:
                    pk = nk.ecg_peaks(sig, sampling_rate=SAMPLING_RATE)
                    if isinstance(pk, tuple) and len(pk) == 2:
                        pinfo = pk[1]
                    else:
                        pinfo = pk
                    rpeaks = extract_rpeaks_from_info(pinfo.get('ECG_R_Peaks') if isinstance(pinfo, dict) else pinfo)
                except Exception:
                    rpeaks = []
            rpeaks_all.append(sorted(set(int(x) for x in rpeaks)))

        ref_rpeaks = []
        for arr in rpeaks_all:
            if len(arr) > 0:
                ref_rpeaks = rpeaks_all[0]
                break
        if len(ref_rpeaks) == 0:
            return [], []

        records_out = []
        local_segments = []
        for ref_r in ref_rpeaks:
            ref_r = int(ref_r)
            r_indices = [ref_r]
            valid = True
            for lead_idx in range(1, len(LEAD_NAMES)):
                rpeaks = rpeaks_all[lead_idx]
                candidates = [r for r in rpeaks if abs(r - ref_r) <= R_WINDOW]
                if not candidates:
                    valid = False
                    break
                matched_r = min(candidates, key=lambda x: abs(x - ref_r))
                r_indices.append(int(matched_r))
            if not valid:
                continue
            segment = pad_segment_centered(ecg, ref_r, SEGMENT_LENGTH)
            # compute features for this segment (multichannel)
            seg_fs = SAMPLING_RATE
            if isinstance(meta, dict):
                # allow meta to override sampling rate if available
                seg_fs = int(meta.get('sampling_rate', meta.get('fs', seg_fs)))
            seg_features = compute_multichannel_features(segment, fs=seg_fs, lead_names=LEAD_NAMES)

            # extract corresponding raw segment if raw_ecg provided
            raw_segment = None
            if raw_ecg is not None:
                try:
                    raw_segment = pad_segment_centered(raw_ecg, ref_r, SEGMENT_LENGTH)
                except Exception:
                    raw_segment = None

            rec_out = {
                "record_id": record_id,
                "segment_index": None,
                "segment": segment,
                "raw_segment": raw_segment,            # NEW: store corresponding raw segment (or None)
                "meta": dict(meta) if meta is not None else None,
                "features": seg_features,   # features for this segment (time+freq per lead)
            }
            # if meta already contains raw_features (precomputed), keep it (no extra work here)
            records_out.append(rec_out)
            local_segments.append(segment)

        return records_out, local_segments
    except Exception:
        return [], []

def process_record_wrapper(args):
    """
    args: (path, meta, raw_path_or_none)
    """
    path, meta, raw_path = args
    try:
        ecg = np.load(path, mmap_mode="r")   # lazy loading (ما ينحمل كامل بالذاكرة)
    except Exception:
        return [], []
    record_id = meta.get('id', Path(path).stem)
    raw_ecg = None
    if raw_path:
        try:
            raw_ecg = np.load(raw_path, mmap_mode="r")
        except Exception:
            raw_ecg = None
    recs, segs = process_single_record(ecg, record_id, meta=meta, raw_ecg=raw_ecg)
    return recs, segs

# --------------------- Thresholding & Filtering ---------------------

def compute_thresholds(all_records):
    if len(all_records) == 0:
        return {}
    criteria = ["rms", "std", "zero_crossings"]
    lead_stats = {lead: {c: [] for c in criteria} for lead in LEAD_NAMES}
    for rec in all_records:
        segment = rec["segment"]
        for lead_idx, lead in enumerate(LEAD_NAMES):
            sig = segment[:, lead_idx]
            rms = np.sqrt(np.mean(sig**2))
            std = np.std(sig)
            zero_crossings = int(np.sum(np.diff(np.sign(sig)) != 0))
            if np.isfinite(rms) and np.isfinite(std) and np.isfinite(zero_crossings):  # <-- check finite
                lead_stats[lead]["rms"].append(rms)
                lead_stats[lead]["std"].append(std)
                lead_stats[lead]["zero_crossings"].append(zero_crossings)
    thresholds = {}
    for lead in LEAD_NAMES:
        thresholds[lead] = {}
        for c in criteria:
            vals = np.array(lead_stats[lead][c])
            if vals.size == 0:
                thresholds[lead][c] = {"p0.1": 0.0, "p99.9": 0.0}
            else:
                thresholds[lead][c] = {
                    "p0.1": float(np.percentile(vals, 0.1)),
                    "p99.9": float(np.percentile(vals, 99.9))
                }
    return thresholds

def filter_segments(all_records, all_segments, thresholds, save_rejected_dir):
    if len(all_records) == 0:
        return [], []
    filtered_records, filtered_segments = [], []
    rejected_count = 0
    MAX_REJECTED_IMAGES = 100
    rejected_dir = save_rejected_dir
    rejected_dir.mkdir(parents=True, exist_ok=True)
    for i, rec in enumerate(all_records):
        segment = rec["segment"]
        reject = False
        reject_leads = []
        reject_energy, reject_std, reject_zc = [], [], []
        for lead_idx, lead in enumerate(LEAD_NAMES):
            sig = segment[:, lead_idx]
            rms = np.sqrt(np.mean(sig**2))
            std = np.std(sig)
            zero_crossings = int(np.sum(np.diff(np.sign(sig)) != 0))
            if not (np.isfinite(rms) and np.isfinite(std) and np.isfinite(zero_crossings)):  # <-- check finite
                reject = True
                reject_leads.append(lead)
                reject_energy.append(True)
                reject_std.append(True)
                reject_zc.append(True)
                continue
            thr = thresholds.get(lead, None)
            if thr is None:
                rej_e = rej_s = rej_z = False
            else:
                rej_e = (rms < thr["rms"]["p0.1"]) or (rms > thr["rms"]["p99.9"])
                rej_s = (std < thr["std"]["p0.1"]) or (std > thr["std"]["p99.9"])
                rej_z = (zero_crossings < thr["zero_crossings"]["p0.1"]) or (zero_crossings > thr["zero_crossings"]["p99.9"])
            reject_energy.append(rej_e)
            reject_std.append(rej_s)
            reject_zc.append(rej_z)
            if rej_e or rej_s or rej_z:
                reject = True
                reject_leads.append(lead)
        if reject:
            rejected_count += 1
            if rejected_count <= MAX_REJECTED_IMAGES:
                n_leads = segment.shape[1]
                fig, axes = plt.subplots(n_leads, 1, figsize=(12, 2*n_leads), sharex=True)
                if n_leads == 1:
                    axes = [axes]
                for lead_idx, ax in enumerate(axes):
                    ax.plot(segment[:, lead_idx])
                    ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=30)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    if reject_energy[lead_idx] or reject_std[lead_idx] or reject_zc[lead_idx]:
                        ax.set_facecolor("#ffe6e6")
                axes[0].set_title(f"Rejected Segment {i} (leads: {', '.join(reject_leads)})")
                axes[-1].set_xlabel("Samples (128)")
                plt.tight_layout()
                save_path = rejected_dir / f"segment_{i}.png"
                plt.savefig(save_path, dpi=150)
                plt.close()
        else:
            rec["segment_index"] = len(filtered_segments)
            filtered_records.append(rec)
            filtered_segments.append(segment)
    print(f"✅ Total rejected segments: {rejected_count}")
    return filtered_records, filtered_segments

# --------------------- Loaders ---------------------

from pathlib import Path
import pandas as pd

def load_group_npy_files_with_meta(folder: Path, meta_csv: Path = None):
    """
    LSAD-friendly: لا تحمل الإشارات. رجّع فقط (path, meta).
    """
    rows = []
    if meta_csv is not None and meta_csv.exists():
        try:
            df_meta = pd.read_csv(meta_csv)
            for _, r in df_meta.iterrows():
                path = r.get('path', None)
                if pd.isna(path) or path is None:
                    continue
                p = Path(path)
                if not p.exists():
                    p = folder / Path(path).name
                    if not p.exists():
                        continue
                meta = r.to_dict()
                meta.setdefault('id', Path(p).stem)
                meta.setdefault('path', str(p.resolve()))
                rows.append((str(p.resolve()), meta))
            return rows
        except Exception:
            pass

    # fallback: scan folder
    for p in sorted(folder.glob("*.npy")):
        meta = {'id': p.stem, 'path': str(p.resolve()), 'src': folder.name}
        rows.append((str(p.resolve()), meta))
    return rows

# --------------------- Batch processing ---------------------

def process_batch(dataset_with_meta, out_prefix, n_workers=None,
                  max_buffer=50000, sample_for_thresholds=5000):
    """
    Memory-efficient batch processor (LSAD-friendly).
    - Uses lazy loading (mmap) in workers
    - Streams segments to disk in shards
    - Thresholds computed on a sample only
    """

    save_dir = SAVE_ROOT / out_prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = save_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {out_prefix}, records={len(dataset_with_meta)}")
    if len(dataset_with_meta) == 0:
        print("No records to process.")
        return

    # We'll not pre-load big raw_map into memory to avoid OOM; instead,
    # for each record we'll try to locate a matching raw .npy file path (and pass path to worker).
    if out_prefix.lower().startswith("ptb"):
        raw_root = PTBXL_Raw_DIR
    elif out_prefix.lower().startswith("lsad"):
        raw_root = LSAD_Raw_DIR
    else:
        raw_root = None

    # Shuffle order
    indices = np.arange(len(dataset_with_meta))
    np.random.shuffle(indices)
    shuffled = [dataset_with_meta[i] for i in indices]

    # set workers
    cpu_count = mp.cpu_count() or 1
    if n_workers is None:
        n_workers = max(1, min(4, cpu_count - 1))

    # Buffers and shard logic
    buffer_segments, buffer_records = [], []
    shard_idx = 0

    def flush_shard():
        nonlocal shard_idx, buffer_segments, buffer_records
        if not buffer_segments:
            return
        seg_path = save_dir / f"{out_prefix}_segments_shard_{shard_idx}.npy"
        rec_path = save_dir / f"{out_prefix}_records_shard_{shard_idx}.pkl"

        np.save(seg_path, np.array(buffer_segments, dtype=np.float32))
        with open(rec_path, "wb") as f:
            pickle.dump(buffer_records, f, protocol=pickle.HIGHEST_PROTOCOL)

        print(f"💾 Flushed shard {shard_idx} with {len(buffer_segments)} segments")
        shard_idx += 1
        buffer_segments.clear()
        buffer_records.clear()

    print("Extracting R-centered segments (streaming mode)...")
    with mp.Pool(processes=n_workers) as pool:
        args = []
        for item in shuffled:
            path = item[0]
            meta = item[1]
            record_id = meta.get('id', Path(path).stem)
            raw_path = None
            if raw_root is not None:
                candidate = raw_root / f"{record_id}.npy"
                if candidate.exists():
                    raw_path = str(candidate.resolve())
            args.append((path, meta, raw_path))

        for recs, segs in tqdm(pool.imap(process_record_wrapper, args),
                               total=len(args)):
            if recs:
                buffer_records.extend(recs)
            if segs:
                buffer_segments.extend(segs)
            if len(buffer_segments) >= max_buffer:
                flush_shard()

    # flush الأخير
    flush_shard()

    # ---- Compute thresholds on a sample only ----
    print("Computing thresholds from sample...")
    sample_records = []
    shard_files = sorted(save_dir.glob(f"{out_prefix}_records_shard_*.pkl"))
    for rec_path in shard_files:
        with open(rec_path, "rb") as f:
            recs = pickle.load(f)
            sample_records.extend(recs)
        if len(sample_records) >= sample_for_thresholds:
            break
    thresholds = compute_thresholds(sample_records)

    # ---- Filtering and final split ----
    print("Filtering and splitting...")
    all_segments_final, all_records_final = [], []
    shard_files = sorted(save_dir.glob(f"{out_prefix}_segments_shard_*.npy"))
    rec_files = sorted(save_dir.glob(f"{out_prefix}_records_shard_*.pkl"))

    for seg_path, rec_path in zip(shard_files, rec_files):
        segs = np.load(seg_path, mmap_mode="r")
        with open(rec_path, "rb") as f:
            recs = pickle.load(f)

        filtered_records, filtered_segments = filter_segments(
            recs, segs, thresholds, rejected_dir
        )
        all_segments_final.extend(filtered_segments)
        all_records_final.extend(filtered_records)

    if not all_segments_final:
        print("No segments left after filtering.")
        return

    # ---- Train/Val/Test split ----
    n = len(all_segments_final)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_segments = all_segments_final[:n_train]
    val_segments = all_segments_final[n_train:n_train+n_val]
    test_segments = all_segments_final[n_train+n_val:]

    np.save(save_dir / "train_segments.npy", np.stack(train_segments).astype(np.float32))
    np.save(save_dir / "val_segments.npy", np.stack(val_segments).astype(np.float32))
    np.save(save_dir / "test_segments.npy", np.stack(test_segments).astype(np.float32))


    with open(save_dir / "records.pkl", "wb") as f:
        pickle.dump(all_records_final, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Flatten CSV ----
    print("Saving flattened features CSV ...")
    rows = []
    for rec in all_records_final:
        base = {}
        base['record_id'] = rec.get('record_id')
        base['segment_index'] = rec.get('segment_index')
        meta = rec.get('meta', {}) or {}
        for k, v in meta.items():
            if k == 'raw_features': continue
            base[f'meta_{k}'] = v
        for k, v in (meta.get('raw_features', {}) or {}).items():
            base[f'raw_{k}'] = v
        for k, v in (rec.get('features', {}) or {}).items():
            base[f'seg_{k}'] = v
        # Note: raw_segment binary data is NOT embedded into CSV; it is available in records.pkl
        rows.append(base)

    if rows:
        df_rows = pd.DataFrame(rows)
        df_rows.to_csv(save_dir / "records_features.csv", index=False)

    print(f"✅ Saved {len(train_segments)} train, {len(val_segments)} val, {len(test_segments)} test segments to {save_dir}")
    print(f"✅ Saved records.pkl and records_features.csv to {save_dir}")

# --------------------- Main ---------------------

if __name__ == "__main__":
    # print("Loading PTB-XL cleaned files + metadata (if present) ...")
    # ptbxl_list = load_group_npy_files_with_meta(PTBXL_DIR, meta_csv=PTBXL_META_CSV)
    # print(f"PTB-XL records found: {len(ptbxl_list)}")
    print("Loading LSAD cleaned files + metadata (if present) ...")
    lsad_list = load_group_npy_files_with_meta(LSAD_DIR, meta_csv=LSAD_META_CSV)
    print(f"LSAD records found: {len(lsad_list)}")

    # Process each source independently
    # if len(ptbxl_list) > 0:
    #     process_batch(ptbxl_list, "ptbxl")
    if len(lsad_list) > 0:
        process_batch(lsad_list, "lsad")

    # Optionally process merged metadata CSV if you created merged_features.csv earlier
    if MERGED_META_CSV.exists():
        try:
            dfm = pd.read_csv(MERGED_META_CSV)
            merged_rows = []
            for _, row in dfm.iterrows():
                p = Path(row['path'])
                if p.exists():
                    try:
                        arr = np.load(p)
                        if arr.ndim == 2 and arr.shape[1] == 12:
                            merged_rows.append((arr.astype(np.float32), row.to_dict()))
                    except Exception:
                        continue
            if len(merged_rows) > 0:
                process_batch(merged_rows, "merged")
        except Exception:
            pass

    print("\nDone.")
