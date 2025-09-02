import os
import ast
import numpy as np
import pandas as pd
import wfdb
from joblib import Parallel, delayed
from sklearn.preprocessing import OneHotEncoder, MultiLabelBinarizer
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, filtfilt
import pywt


# --- CONFIGURATION ---

PTBXL_PATH = Path('../../ptbxl')
DATABASE_CSV = PTBXL_PATH / 'ptbxl_database.csv'
SCP_STATEMENTS_CSV = PTBXL_PATH / 'scp_statements.csv'
TARGET_LENGTH = 1000

SAVE_DIR = Path('./ptbxl_dl_dataset_v2')
DATASETS_DIR = SAVE_DIR / 'datasets'
ECG_PLOTS_DIR = SAVE_DIR / 'ecg_plots'
STATS_PLOTS_DIR = SAVE_DIR / 'stats_plots'

for d in [DATASETS_DIR, ECG_PLOTS_DIR, STATS_PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- LOAD SCP STATEMENTS ---

scp_statements = pd.read_csv(SCP_STATEMENTS_CSV, index_col=0)
scp_code_to_superclass = {}
scp_code_to_description = {}

for idx, row in scp_statements.iterrows():
    code = idx.strip()
    superclass = row['diagnostic_class'] if pd.notna(row['diagnostic_class']) else 'Unknown'
    description = row['description'] if pd.notna(row['description']) else 'Unknown'
    scp_code_to_superclass[code] = superclass
    scp_code_to_description[code] = description

def get_superclass_and_names(labels_set):
    superclasses = set()
    names = set()
    for code in labels_set:
        if code in scp_code_to_superclass:
            superclasses.add(scp_code_to_superclass[code])
            desc = scp_code_to_description[code]
            if desc != 'Unknown':
                names.add(desc)
    if not superclasses:
        superclasses.add('Unknown')
    if not names:
        names.add('Unknown')
    return ', '.join(sorted(superclasses)), ', '.join(sorted(names))

from scipy.signal import butter, filtfilt, iirnotch, medfilt


# إزالة خط الأساس بفلتر median
def baseline_wander_removal(data, fs=100.0, window_sec=0.2):
    window_size = int(window_sec * fs)
    if window_size % 2 == 0:
        window_size += 1
    baseline = medfilt(data, kernel_size=(window_size, 1))
    return data - baseline

# --- FILTERING BASED ON PAPER ---

def butter_bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=100.0, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

# --- LOAD AND CLEAN METADATA ---

df = pd.read_csv(DATABASE_CSV)
print("Original data shape:", df.shape)

df = df[df['validated_by_human'] == True]
df = df[df['filename_lr'].notnull()]
print("After filename filtering:", df.shape)

for col in ['electrodes_problems', 'pacemaker', 'burst_noise', 'static_noise']:
    df[col] = df[col].fillna(0)

df = df[
    (df['electrodes_problems'] == 0) &
    (df['pacemaker'] == 0) &
    (df['burst_noise'] == 0) &
    (df['static_noise'] == 0)
]
print("After artifact filtering:", df.shape)

df['heart_axis'] = df['heart_axis'].fillna('NOT')
df['sex'] = df['sex'].fillna('unknown')
df['scp_codes'] = df['scp_codes'].fillna('NOT')
df = df.fillna(0)

def extract_labels_all(scp_codes_str):
    try:
        scp_codes = ast.literal_eval(scp_codes_str)
        return set([code for code, score in scp_codes.items() if score >= 0])
    except:
        return set()

df['labels'] = df['scp_codes'].apply(extract_labels_all)
df = df[df['labels'].map(len) > 0].reset_index(drop=True)
print("After label extraction:", df.shape)

encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
onehot_cols = encoder.fit_transform(df[['sex', 'heart_axis']])
onehot_df = pd.DataFrame(onehot_cols, columns=encoder.get_feature_names_out(['sex', 'heart_axis']))
df_meta = pd.concat([df[['age']].reset_index(drop=True), onehot_df.reset_index(drop=True)], axis=1)

all_labels = sorted(set.union(*df['labels']))
mlb = MultiLabelBinarizer(classes=all_labels)
Y = mlb.fit_transform(df['labels'])
label_cols = [f'label_{l}' for l in all_labels]

df['strat_fold'] = df['strat_fold'].astype(int)
df_train = df[df['strat_fold'] < 9].reset_index(drop=True)
df_val = df[df['strat_fold'] == 9].reset_index(drop=True)
df_test = df[df['strat_fold'] == 10].reset_index(drop=True)

Y_train = Y[df_train.index]
Y_val = Y[df_val.index]
Y_test = Y[df_test.index]

meta_train = df_meta.loc[df_train.index].reset_index(drop=True)
meta_val = df_meta.loc[df_val.index].reset_index(drop=True)
meta_test = df_meta.loc[df_test.index].reset_index(drop=True)


# --- ECG FUNCTIONS ---

def get_record_path(filename_lr):
    base = os.path.splitext(filename_lr)[0]
    return PTBXL_PATH / base

def clean_ecg_signal(arr, fs=100.0, use_notch=True, notch_freq=50.0, do_trim=True, trim_samples=20, use_median_baseline=True):
    # 1. فلترة تمرير النطاق
    arr = butter_bandpass_filter(arr, fs=fs)
 
    # 3. إزالة خط الأساس بواسطة median filter
    if use_median_baseline:
        arr = baseline_wander_removal(arr, fs=fs)

    return arr


def load_ecg(record_path):
    record = wfdb.rdrecord(record_path)
    arr = record.p_signal
    arr = clean_ecg_signal(arr, fs=record.fs if hasattr(record, 'fs') else 100.0)
    if arr.shape[0] > TARGET_LENGTH:
        arr = arr[:TARGET_LENGTH, :]
    elif arr.shape[0] < TARGET_LENGTH:
        arr = np.pad(arr, ((0, TARGET_LENGTH - arr.shape[0]), (0, 0)), 'constant')
    return arr

# --- PLOTTING FUNCTIONS WITH DISEASE & AXIS NAMES ---

def plot_12lead(ecg_array, title, save_path, disease_superclass=None, disease_names=None, heart_axis_name=None):
    plt.figure(figsize=(15, 8))
    for i in range(12):
        plt.subplot(6, 2, i+1)
        plt.plot(ecg_array[:, i], linewidth=1)
        plt.title(LEAD_NAMES[i], fontweight='normal', fontsize=8)
        plt.xticks(fontsize=6)
        plt.yticks(fontsize=6)
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.5)
    full_title = title
    if disease_superclass:
        full_title += f' | SCP Superclass: {disease_superclass}'
    if disease_names:
        full_title += f' | Disease: {disease_names}'
    if heart_axis_name:
        full_title += f' | Heart Axis: {heart_axis_name}'
    plt.suptitle(full_title, y=1.02, fontweight='normal', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=600)
    plt.close()

def plot_distribution(df, column, save_path):
    plt.figure(figsize=(8, 6))
    sns.histplot(df[column], kde=True, bins=30)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def plot_pie_distribution(df, column, save_path):
    plt.figure(figsize=(8, 8))
    counts = df[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=600)
    plt.close()

def save_numpy_dataset(df_split, meta_split, Y_split, name):
    ecg_data = []
    for fname in tqdm(df_split['filename_lr'], desc=f"Loading ECG for {name}"):
        path = get_record_path(fname)
        try:
            ecg = load_ecg(path)
            ecg_data.append(ecg)
        except Exception as e:
            print(f"Error reading {fname}: {e}")
            ecg_data.append(None)
    valid_indices = [i for i, ecg in enumerate(ecg_data) if ecg is not None]
    ecg_data = [ecg_data[i] for i in valid_indices]
    meta_split = meta_split.iloc[valid_indices].reset_index(drop=True)
    Y_split = Y_split[valid_indices]

    X_signals = np.array(ecg_data)
    record_ids = df_split.loc[df_split.index[valid_indices], 'ecg_id'].values
    np.savez_compressed(DATASETS_DIR / f'{name}_signals.npz', X=X_signals, record_ids=record_ids)
    meta_split.to_csv(DATASETS_DIR / f'{name}_metadata.csv', index=False)
    pd.DataFrame(Y_split, columns=label_cols).to_csv(DATASETS_DIR / f'{name}_labels.csv', index=False)

    df_amp = pd.DataFrame([np.max(ecg, axis=0) - np.min(ecg, axis=0) for ecg in ecg_data], columns=LEAD_NAMES)
    df_amp = df_amp[df_amp < df_amp.quantile(0.99)].dropna()

    plt.figure(figsize=(10, 5))
    sns.boxplot(data=df_amp, orient='h')
    plt.xlabel('Amplitude (mV)')
    plt.title(f'{name} ECG Amplitude Ranges')
    plt.tight_layout()
    plt.savefig(STATS_PLOTS_DIR / f'{name}_amplitude_plot.png', dpi=600)
    plt.close()

    for i in range(min(20, len(ecg_data))):
        labels_set = df_split.iloc[valid_indices[i]]['labels']
        heart_axis_name = df_split.iloc[valid_indices[i]]['heart_axis']
        disease_superclass, disease_names = get_superclass_and_names(labels_set)
        plot_12lead(
            ecg_data[i],
            title=f'{name} ECG #{i}',
            save_path=ECG_PLOTS_DIR / f'{name}_ecg_{i}.png',
            disease_superclass=disease_superclass,
            disease_names=disease_names,
            heart_axis_name=heart_axis_name
        )


# --- ADDITIONAL STATISTICS PLOTS ---

def run_plots():
    save_numpy_dataset(df_train, meta_train, Y_train, 'train')
    save_numpy_dataset(df_val, meta_val, Y_val, 'val')
    save_numpy_dataset(df_test, meta_test, Y_test, 'test')

    plot_distribution(df, 'age', STATS_PLOTS_DIR / 'age_distribution.png')
    plot_pie_distribution(df, 'sex', STATS_PLOTS_DIR / 'sex_distribution.png')
    plot_pie_distribution(df, 'heart_axis', STATS_PLOTS_DIR / 'heart_axis_distribution.png')

    print("PTB-XL data preparation complete with full 12-lead signal dataset")

# Uncomment to run
run_plots()
