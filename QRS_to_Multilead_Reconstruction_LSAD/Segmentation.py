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

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="neurokit2")

# --- Configuration ---
ROOT = Path("./unified_ecg_dataset_v1")
PTBXL_DIR = ROOT / "ptbxl_clean"
LSAD_DIR = ROOT / "lsad_clean"
MERGED_META_CSV = ROOT / "merged_features.csv"   # optional merged metadata
PTBXL_META_CSV = PTBXL_DIR / "ptbxl_metadata.csv"
LSAD_META_CSV = LSAD_DIR / "lsad_metadata.csv"
SAVE_ROOT = Path("Segments")
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

# --------------------- Core processing ---------------------

def process_single_record(ecg, record_id, meta=None):
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
            rec_out = {
                "record_id": record_id,
                "segment_index": None,
                "segment": segment,
                "meta": dict(meta) if meta is not None else None,
            }
            records_out.append(rec_out)
            local_segments.append(segment)

        return records_out, local_segments
    except Exception:
        return [], []

def process_record_wrapper(args):
    ecg, meta = args
    record_id = None
    if isinstance(meta, dict) and 'id' in meta:
        record_id = meta.get('id')
    else:
        try:
            if isinstance(meta, dict) and 'path' in meta:
                record_id = Path(meta['path']).stem
            else:
                record_id = "rec_unk"
        except Exception:
            record_id = "rec_unk"
    recs, segs = process_single_record(ecg, record_id, meta=meta)
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

def load_group_npy_files_with_meta(folder: Path, meta_csv: Path = None):
    """Return a list of tuples (arr, meta_dict).
    If meta_csv exists and contains 'path' column that matches saved files, use it to preserve metadata.
    Otherwise fall back to scanning .npy files in folder and creating minimal meta.
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
                    # try relative to folder
                    p = folder / p.name
                    if not p.exists():
                        continue
                try:
                    arr = np.load(p)
                    # ensure leads==12
                    if arr.ndim == 2 and arr.shape[1] == 12:
                        meta = r.to_dict()
                        rows.append((arr.astype(np.float32), meta))
                except Exception:
                    continue
            return rows
        except Exception:
            pass
    # fallback: scan folder for .npy files
    files = sorted(glob.glob(str(folder / "*.npy")))
    for p in files:
        try:
            arr = np.load(p)
            if arr.ndim == 2 and arr.shape[1] == 12:
                meta = {'id': Path(p).stem, 'path': str(Path(p).resolve()), 'src': folder.name}
                rows.append((arr.astype(np.float32), meta))
        except Exception:
            continue
    return rows


# --------------------- Batch processing ---------------------

def process_batch(dataset_with_meta, out_prefix, n_workers=None):
    save_dir = SAVE_ROOT / out_prefix
    save_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir = save_dir / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)

    print(f"Processing: {out_prefix}, records={len(dataset_with_meta)}")
    if len(dataset_with_meta) == 0:
        print("No records to process.")
        return

    # shuffle
    indices = np.arange(len(dataset_with_meta))
    np.random.shuffle(indices)
    shuffled = [dataset_with_meta[i] for i in indices]

    # multiprocessing pool
    cpu_count = mp.cpu_count() or 1
    if n_workers is None:
        n_workers = max(1, min(4, cpu_count - 1))

    all_records = []
    all_segments = []
    print("Extracting R-centered segments...")
    with mp.Pool(processes=n_workers) as pool:
        args = [(item[0], item[1]) for item in shuffled]
        for recs, segs in tqdm(pool.imap(process_record_wrapper, args), total=len(args)):
            if recs:
                # ensure segment_index assignment will be done in filter stage
                all_records.extend(recs)
            if segs:
                all_segments.extend(segs)

    if len(all_records) == 0:
        print("No segments extracted from any record.")
        return

    # compute thresholds based on extracted segments
    thresholds = compute_thresholds(all_records)
    filtered_records, filtered_segments = filter_segments(all_records, all_segments, thresholds, rejected_dir)

    if len(filtered_segments) == 0:
        print("No segments left after filtering.")
        return

    # Train/test/val split
    n = len(filtered_segments)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_segments = filtered_segments[:n_train]
    val_segments = filtered_segments[n_train:n_train+n_val]
    test_segments = filtered_segments[n_train+n_val:]

    np.save(save_dir / "train_segments.npy", np.array(train_segments, dtype=np.float32))
    np.save(save_dir / "val_segments.npy", np.array(val_segments, dtype=np.float32))
    np.save(save_dir / "test_segments.npy", np.array(test_segments, dtype=np.float32))
    # save filtered_records (with metadata) for traceability
    with open(save_dir / "records.pkl", "wb") as f:
        pickle.dump(filtered_records, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✅ Saved {len(train_segments)} train, {len(val_segments)} val, {len(test_segments)} test segments to {save_dir}")


# --------------------- Main ---------------------

if __name__ == "__main__":
    print("Loading PTB-XL cleaned files + metadata (if present) ...")
    ptbxl_list = load_group_npy_files_with_meta(PTBXL_DIR, meta_csv=PTBXL_META_CSV)
    print(f"PTB-XL records found: {len(ptbxl_list)}")
    print("Loading LSAD cleaned files + metadata (if present) ...")
    lsad_list = load_group_npy_files_with_meta(LSAD_DIR, meta_csv=LSAD_META_CSV)
    print(f"LSAD records found: {len(lsad_list)}")

    # Process each source independently
    if len(ptbxl_list) > 0:
        process_batch(ptbxl_list, "ptbxl")
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
