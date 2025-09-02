import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# إعدادات
SAVE_DIR = Path("Segments")
LEAD_NAMES = ['I','II','III','aVR','aVL','aVF','V1','V2','V3','V4','V5','V6']

def load_data():
    train = np.load(SAVE_DIR / "train_segments.npy")
    val = np.load(SAVE_DIR / "val_segments.npy")
    test = np.load(SAVE_DIR / "test_segments.npy")
    with open(SAVE_DIR / "records.pkl", "rb") as f:
        records = pickle.load(f)
    return train, val, test, records

def plot_dataset_distribution(train, val, test):
    sizes = [len(train), len(val), len(test)]
    labels = ["Train", "Validation", "Test"]

    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("Set2"))
    plt.title("Dataset Split Distribution")
    plt.savefig(SAVE_DIR / "dataset_split.png", dpi=150)
    plt.close()

def plot_segment_length(train):
    lengths = [seg.shape[0] for seg in train]
    plt.figure(figsize=(6,4))
    sns.histplot(lengths, bins=20, kde=True, color="blue")
    plt.title("Segment Length Distribution (Train)")
    plt.xlabel("Length (samples)")
    plt.ylabel("Count")
    plt.savefig(SAVE_DIR / "segment_length_hist.png", dpi=150)
    plt.close()

def plot_signal_statistics(records):
    stats = {lead: {"rms": [], "std": [], "zc": []} for lead in LEAD_NAMES}
    for rec in records:
        seg = rec["segment"]
        for lead_idx, lead in enumerate(LEAD_NAMES):
            sig = seg[:, lead_idx]
            rms = np.sqrt(np.mean(sig**2))
            std = np.std(sig)
            zc = np.sum(np.diff(np.sign(sig)) != 0)
            stats[lead]["rms"].append(rms)
            stats[lead]["std"].append(std)
            stats[lead]["zc"].append(zc)

    # Boxplots لكل ميتركس
    for metric in ["rms", "std", "zc"]:
        plt.figure(figsize=(12,6))
        data = [stats[lead][metric] for lead in LEAD_NAMES]
        sns.boxplot(data=data)
        plt.xticks(range(len(LEAD_NAMES)), LEAD_NAMES)
        plt.title(f"{metric.upper()} Distribution per Lead")
        plt.ylabel(metric.upper())
        plt.savefig(SAVE_DIR / f"{metric}_per_lead.png", dpi=150)
        plt.close()

def plot_example_segments(train, n=5):
    for i in range(n):
        seg = train[i]
        fig, axes = plt.subplots(len(LEAD_NAMES), 1, figsize=(10, 20), sharex=True)
        for lead_idx, ax in enumerate(axes):
            ax.plot(seg[:, lead_idx], color="black")
            ax.set_ylabel(LEAD_NAMES[lead_idx], rotation=0, labelpad=30)
            ax.grid(True, linestyle="--", alpha=0.5)
        axes[0].set_title(f"Example Segment {i}")
        axes[-1].set_xlabel("Samples")
        plt.tight_layout()
        plt.savefig(SAVE_DIR / f"example_segment_{i}.png", dpi=150)
        plt.close()

if __name__ == "__main__":
    train, val, test, records = load_data()
    print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    plot_dataset_distribution(train, val, test)
    plot_segment_length(train)
    plot_signal_statistics(records)
    plot_example_segments(train, n=5)

    print("✅ Analysis plots saved in:", SAVE_DIR)
