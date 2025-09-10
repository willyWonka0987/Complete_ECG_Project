import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import skew, kurtosis

# --------------------- Configuration ---------------------
DATA_ROOT = Path("./Segments")
OUTPUT_DIR = Path("./Segments_Stats")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = ["lsad", "ptbxl"]
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

MAX_POINTS_FOR_PLOT = 50000   # عدد العينات للرسم
BATCH_SIZE = 10000            # حجم الدفعة

# --------------------- Helpers ---------------------
def load_segments(dataset_name):
    splits = ["train", "val", "test"]
    all_segs = []

    for split in splits:
        seg_file = DATA_ROOT / dataset_name / f"{split}_segments.npy"
        if seg_file.exists():
            segs = np.load(seg_file, mmap_mode="r")
            all_segs.append(segs)
            print(f"Loaded {len(segs)} segments for {dataset_name} [{split}]")
        else:
            print(f"⚠️ {seg_file} not found")

    if all_segs:
        merged = np.concatenate(all_segs, axis=0)
        print(f"✅ Total {len(merged)} segments merged for {dataset_name}")
        return merged
    else:
        print(f"❌ No segments found for {dataset_name}")
        return None


def compute_signal_stats_stream(segs, batch_size=BATCH_SIZE):
    """
    Compute mean, std, min, max, percentiles, skew, kurtosis
    باستخدام batching + sampling
    """
    n_leads = segs.shape[2]
    stats = {lead: {"sum":0, "sum2":0, "count":0,
                    "min":np.inf, "max":-np.inf, "samples":[]} 
             for lead in LEAD_NAMES + ["all_leads"]}
    
    n_segments = segs.shape[0]
    for start in range(0, n_segments, batch_size):
        batch = segs[start:start+batch_size]

        for i, lead in enumerate(LEAD_NAMES):
            x = batch[:,:,i].ravel()
            stats[lead]["sum"] += x.sum()
            stats[lead]["sum2"] += (x**2).sum()
            stats[lead]["count"] += x.size
            stats[lead]["min"] = min(stats[lead]["min"], x.min())
            stats[lead]["max"] = max(stats[lead]["max"], x.max())
            if len(stats[lead]["samples"]) < MAX_POINTS_FOR_PLOT:
                take = min(MAX_POINTS_FOR_PLOT - len(stats[lead]["samples"]), x.size)
                idx = np.random.choice(x.size, take, replace=False)
                stats[lead]["samples"].extend(x[idx].tolist())

        # all leads
        x_all = batch.ravel()
        stats["all_leads"]["sum"] += x_all.sum()
        stats["all_leads"]["sum2"] += (x_all**2).sum()
        stats["all_leads"]["count"] += x_all.size
        stats["all_leads"]["min"] = min(stats["all_leads"]["min"], x_all.min())
        stats["all_leads"]["max"] = max(stats["all_leads"]["max"], x_all.max())
        if len(stats["all_leads"]["samples"]) < MAX_POINTS_FOR_PLOT:
            take = min(MAX_POINTS_FOR_PLOT - len(stats["all_leads"]["samples"]), x_all.size)
            idx = np.random.choice(x_all.size, take, replace=False)
            stats["all_leads"]["samples"].extend(x_all[idx].tolist())
    
    # finalize stats
    final_stats = {}
    for lead, s in stats.items():
        mean = s["sum"]/s["count"]
        std = np.sqrt(s["sum2"]/s["count"] - mean**2)
        samples = np.array(s["samples"])
        percentiles = {p: np.percentile(samples, p) for p in [10, 50, 90]}
        skewness = skew(samples)
        kurt = kurtosis(samples)
        final_stats[lead] = {
            "mean":mean, "std":std, "min":s["min"], "max":s["max"],
            "p10":percentiles[10], "p50":percentiles[50], "p90":percentiles[90],
            "skewness":skewness, "kurtosis":kurt
        }
    return final_stats, stats

def plot_histograms(stats_samples, dataset_name, output_dir):
    for lead in LEAD_NAMES:
        data = np.array(stats_samples[lead]["samples"])
        if len(data) == 0:
            continue

        # حساب العتبات العليا
        p99 = np.percentile(data, 99.0)
        p995 = np.percentile(data, 99.5)
        count_above_99 = np.sum(data > p99)
        count_above_995 = np.sum(data > p995)

        # حساب العتبات الدنيا
        p01 = np.percentile(data, 1.0)
        p05 = np.percentile(data, 0.5)
        p001 = np.percentile(data, 0.1)
        count_below_01 = np.sum(data < p01)
        count_below_05 = np.sum(data < p05)
        count_below_001 = np.sum(data < p001)

        plt.figure(figsize=(10,5))
        sns.histplot(data, bins=50, kde=True, color='skyblue')

        # خطوط عتبات عليا
        plt.axvline(p99, color='red', linestyle='--',
                    label=f'p99={p99:.3f} (>{count_above_99})')
        plt.axvline(p995, color='purple', linestyle='--',
                    label=f'p99.5={p995:.3f} (>{count_above_995})')

        # خطوط عتبات دنيا
        plt.axvline(p01, color='green', linestyle='--',
                    label=f'p1={p01:.3f} (<{count_below_01})')
        plt.axvline(p05, color='blue', linestyle='--',
                    label=f'p0.5={p05:.3f} (<{count_below_05})')
        plt.axvline(p001, color='orange', linestyle='--',
                    label=f'p0.1={p001:.3f} (<{count_below_001})')

        plt.title(f"{dataset_name.upper()} - Lead {lead} Histogram")
        plt.xlabel("Voltage")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_hist_lead_{lead}.png", dpi=150)
        plt.close()

    # all leads مع نفس الشي
    data = np.array(stats_samples["all_leads"]["samples"])
    if len(data) > 0:
        p99 = np.percentile(data, 99.0)
        p995 = np.percentile(data, 99.5)
        count_above_99 = np.sum(data > p99)
        count_above_995 = np.sum(data > p995)

        p01 = np.percentile(data, 1.0)
        p05 = np.percentile(data, 0.5)
        p001 = np.percentile(data, 0.1)
        count_below_01 = np.sum(data < p01)
        count_below_05 = np.sum(data < p05)
        count_below_001 = np.sum(data < p001)

        plt.figure(figsize=(10,5))
        sns.histplot(data, bins=100, kde=True, color='orange')

        plt.axvline(p99, color='red', linestyle='--',
                    label=f'p99={p99:.3f} (>{count_above_99})')
        plt.axvline(p995, color='purple', linestyle='--',
                    label=f'p99.5={p995:.3f} (>{count_above_995})')

        plt.axvline(p01, color='green', linestyle='--',
                    label=f'p1={p01:.3f} (<{count_below_01})')
        plt.axvline(p05, color='blue', linestyle='--',
                    label=f'p0.5={p05:.3f} (<{count_below_05})')
        plt.axvline(p001, color='orange', linestyle='--',
                    label=f'p0.1={p001:.3f} (<{count_below_001})')

        plt.title(f"{dataset_name.upper()} - All Leads Histogram")
        plt.xlabel("Voltage")
        plt.ylabel("Frequency")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / f"{dataset_name}_hist_all_leads.png", dpi=150)
        plt.close()


def plot_boxplots(stats_samples, dataset_name, output_dir):
    data = [np.array(stats_samples[lead]["samples"]) for lead in LEAD_NAMES]
    plt.figure(figsize=(12,6))
    sns.boxplot(data=data)
    plt.xticks(np.arange(len(LEAD_NAMES)), LEAD_NAMES)
    plt.title(f"{dataset_name.upper()} - Lead Boxplots")
    plt.ylabel("Voltage")
    plt.tight_layout()
    plt.savefig(output_dir / f"{dataset_name}_boxplots.png", dpi=150)
    plt.close()

def compare_datasets(stats_dict, dataset_names, output_dir):
    leads = LEAD_NAMES + ["all_leads"]

    # prepare DataFrames
    metrics = ["mean","std","p10","p50","p90","skewness","kurtosis"]
    data = {}
    for m in metrics:
        rows = []
        for lead in leads:
            rows.append([stats_dict[dataset_names[0]][lead][m],
                         stats_dict[dataset_names[1]][lead][m]])
        data[m] = pd.DataFrame(rows, columns=dataset_names, index=leads)

    # bar plots
    for m, df in data.items():
        df.plot(kind='bar', figsize=(12,5))
        plt.title(f"{m.capitalize()} Comparison")
        plt.ylabel(m)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_{m}.png", dpi=150)
        plt.close()

    # heatmap of differences (dataset1 - dataset2)
    diff = pd.DataFrame({m: data[m][dataset_names[0]] - data[m][dataset_names[1]] for m in metrics}, index=leads)
    plt.figure(figsize=(10,6))
    sns.heatmap(diff, annot=True, fmt=".2f", cmap="coolwarm", center=0)
    plt.title("Differences Heatmap (Dataset1 - Dataset2)")
    plt.tight_layout()
    plt.savefig(output_dir / "comparison_heatmap.png", dpi=150)
    plt.close()

# --------------------- Main ---------------------
all_stats = {}
for ds in DATASETS:
    segs = load_segments(ds)
    if segs is None:
        continue
    stats, stats_samples = compute_signal_stats_stream(segs)
    all_stats[ds] = stats

    # save text summary
    summary_file = OUTPUT_DIR / f"{ds}_summary.txt"
    with open(summary_file, "w") as f:
        for lead, s in stats.items():
            f.write(f"{lead}: {s}\n")
    
    # plots
    plot_histograms(stats_samples, ds, OUTPUT_DIR)
    plot_boxplots(stats_samples, ds, OUTPUT_DIR)

# Compare datasets
if len(all_stats) == 2:
    compare_datasets(all_stats, DATASETS, OUTPUT_DIR)

print("✅ Done. All stats, histograms, boxplots, and advanced comparisons saved in:", OUTPUT_DIR)
