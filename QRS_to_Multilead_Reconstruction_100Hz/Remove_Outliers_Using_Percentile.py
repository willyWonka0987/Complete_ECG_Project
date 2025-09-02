import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import json
import pandas as pd

# ================== الإعدادات ==================
SRC_DIR = Path("RichECG_Datasets")           # نفس SAVE_DIR من سكربت التقسيم
OUT_DIR = SRC_DIR / "filtered_segments"      # مجلد الخرج الجديد
OUT_DIR.mkdir(parents=True, exist_ok=True)

REJECTED_IMG_DIR = OUT_DIR / "rejected_preview"
REJECTED_IMG_DIR.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

LOW_PCT  = 0.1    # Percentile السفلي
HIGH_PCT = 99.5   # Percentile العلوي
MAX_REJECTED_IMAGES = 100

# =================================================

def load_segments_for_split(split):
    path = SRC_DIR / f"segments_{split}.npy"
    arr = np.load(path)  # shape = (N, L, 12), dtype float32
    return arr

def merge_all_splits():
    merged = []
    index_rows = []
    for split in ["train", "val", "test"]:
        segs = load_segments_for_split(split)
        merged.append(segs)
        # خريطة الفهارس الأصلية لكل صف بعد الدمج (مفيدة إذا بدك ترجع للأصل)
        index_rows += [(split, i) for i in range(len(segs))]
    merged = np.concatenate(merged, axis=0)
    idx_map = pd.DataFrame(index_rows, columns=["split", "orig_index"])
    return merged, idx_map

def compute_lead_metrics(segments):
    """
    يُعيد dict يحوي المقاييس لكل lead على مستوى كل مقطع:
        - rms:  (N,)
        - std:  (N,)
        - zc:   (N,)  (zero crossings)
    """
    N, L, C = segments.shape
    metrics = {lead: {"rms": None, "std": None, "zc": None} for lead in LEAD_NAMES}

    # لحساب zc بشكل متجه
    # sign shape = (N, L, C) -> diff along axis=1 -> (N, L-1, C) -> != 0 -> sum over axis=1 -> (N, C)
    signs = np.sign(segments)
    zc_counts = np.sum(np.diff(signs, axis=1) != 0, axis=1)  # (N, C)

    # rms و std
    # rms: sqrt(mean(x^2, axis=1)) -> (N, C)
    rms_all = np.sqrt(np.mean(segments**2, axis=1))
    std_all = np.std(segments, axis=1)

    for c, lead in enumerate(LEAD_NAMES):
        metrics[lead]["rms"] = rms_all[:, c]
        metrics[lead]["std"] = std_all[:, c]
        metrics[lead]["zc"]  = zc_counts[:, c]
    return metrics

def compute_thresholds(metrics):
    """
    يحسب عتبات الـ percentiles لكل (lead, criterion)
    """
    thresholds = {}
    for lead in LEAD_NAMES:
        thresholds[lead] = {}
        for crit in ["rms", "std", "zc"]:
            vals = metrics[lead][crit]
            thresholds[lead][crit] = {
                "low": float(np.percentile(vals, LOW_PCT)),
                "high": float(np.percentile(vals, HIGH_PCT)),
            }
    return thresholds

def filter_segments(segments, thresholds, save_preview=True):
    """
    يرجّع:
      kept_segments: (M, L, 12)
      kept_mask:     (N,) boolean
      rejected_mask: (N,) boolean
    """
    N, L, C = segments.shape
    kept_mask = np.ones(N, dtype=bool)

    # حساب المقاييس مرة واحدة لتسريع العمل
    metrics = compute_lead_metrics(segments)

    rejected_images = 0
    for i in tqdm(range(N), desc="Filtering segments"):
        seg = segments[i]
        reject = False
        per_lead_flags = {lead: {"rms": False, "std": False, "zc": False} for lead in LEAD_NAMES}

        for c, lead in enumerate(LEAD_NAMES):
            rms_i = metrics[lead]["rms"][i]
            std_i = metrics[lead]["std"][i]
            zc_i  = metrics[lead]["zc"][i]

            if (rms_i < thresholds[lead]["rms"]["low"]) or (rms_i > thresholds[lead]["rms"]["high"]):
                reject = True; per_lead_flags[lead]["rms"] = True
            if (std_i < thresholds[lead]["std"]["low"]) or (std_i > thresholds[lead]["std"]["high"]):
                reject = True; per_lead_flags[lead]["std"] = True
            if (zc_i  < thresholds[lead]["zc"]["low"])  or (zc_i  > thresholds[lead]["zc"]["high"]):
                reject = True; per_lead_flags[lead]["zc"]  = True

        if reject:
            kept_mask[i] = False
            if save_preview and rejected_images < MAX_REJECTED_IMAGES:
                # رسم معاينة للمرفوض مع تمييز الليدات المخالفة بتظليل الخلفية
                n_leads = seg.shape[1]
                fig, axes = plt.subplots(n_leads, 1, figsize=(10, 2*n_leads), sharex=True)
                if n_leads == 1:
                    axes = [axes]
                for lead_idx, ax in enumerate(axes):
                    ax.plot(seg[:, lead_idx])
                    ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=25)
                    ax.grid(True, linestyle="--", alpha=0.5)
                    lead = LEAD_NAMES[lead_idx]
                    if any(per_lead_flags[lead].values()):
                        ax.set_facecolor("#ffe6e6")
                axes[0].set_title(f"Rejected segment #{i}")
                axes[-1].set_xlabel("Samples")
                plt.tight_layout()
                plt.savefig(REJECTED_IMG_DIR / f"rejected_{i}.png", dpi=150)
                plt.close()
                rejected_images += 1

    kept_segments = segments[kept_mask]
    rejected_mask = ~kept_mask
    return kept_segments, kept_mask, rejected_mask

def main():
    print("Loading and merging segments (train/val/test)…")
    merged, idx_map = merge_all_splits()  # merged: (N, 128, 12)
    np.save(OUT_DIR / "merged_segments.npy", merged.astype(np.float32))
    idx_map.to_csv(OUT_DIR / "merged_index_map.csv", index=False)
    print(f"Merged shape: {merged.shape}")

    print("Computing global thresholds from merged data…")
    metrics = compute_lead_metrics(merged)
    thresholds = compute_thresholds(metrics)
    with open(OUT_DIR / "thresholds.json", "w") as f:
        json.dump(thresholds, f, indent=2)

    print("Applying percentile filtering…")
    kept_segments, kept_mask, rejected_mask = filter_segments(merged, thresholds, save_preview=True)

    # حفظ النتائج
    np.save(OUT_DIR / "merged_segments_filtered.npy", kept_segments.astype(np.float32))
    np.save(OUT_DIR / "kept_mask.npy", kept_mask)
    np.save(OUT_DIR / "rejected_mask.npy", rejected_mask)

    # تقرير مختصر
    total = len(merged)
    kept = kept_segments.shape[0]
    rej = int(rejected_mask.sum())
    report = {
        "total_segments": int(total),
        "kept_segments": int(kept),
        "rejected_segments": int(rej),
        "low_percentile": LOW_PCT,
        "high_percentile": HIGH_PCT,
        "preview_images_saved": min(rej, MAX_REJECTED_IMAGES)
    }
    with open(OUT_DIR / "filter_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("=== DONE ===")
    print(report)

if __name__ == "__main__":
    main()
