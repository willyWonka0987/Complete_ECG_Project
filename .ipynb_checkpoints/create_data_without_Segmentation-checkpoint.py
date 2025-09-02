from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from utils import get_ecg, get_minmax, save_data

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).parent
DB_CSV_PATH = ROOT.joinpath('csv_files')

f_lst = ['ecg_id', 'superclasses', 'heart_axis']
leads = list(range(12))
cols = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']


def get_df(data_frame):
    data_frame['heart_axis'] = data_frame['heart_axis'].fillna('NOT')
    data_frame['superclasses'] = data_frame['superclasses'].fillna('NOT')
    data_frame = data_frame.fillna(0)
    data_frame = data_frame[data_frame['hr'] != 0]
    df_clean = data_frame[(data_frame['electrodes_problems'] == 0) &
                          (data_frame['pacemaker'] == 0) &
                          (data_frame['burst_noise'] == 0) &
                          (data_frame['static_noise'] == 0)]
    return df_clean[['ecg_id', 'superclasses', 'heart_axis', 'filename_lr']]


def create_labels(data_frame, column_name):
    label_encoder = LabelEncoder()
    data_frame[column_name] = label_encoder.fit_transform(data_frame[column_name])
    with open(f'{column_name}_labeling.txt', 'w') as f:
        for i, label in enumerate(label_encoder.classes_):
            f.write(f'{i}: {label}\n')
    return data_frame


def save_img_minmax(data_frame, name='train'):
    Path(ROOT.joinpath('images')).mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 5))
    sns.boxplot(data=data_frame, orient='h')
    plt.xlabel('Amplitude, mV')
    plt.ylabel('Leads')
    plt.savefig(Path('images').joinpath(f'{name}_minmax.jpg'), bbox_inches='tight', dpi=600)


if __name__ == '__main__':
    df = pd.read_csv(DB_CSV_PATH.joinpath('data_base.csv'))
    df_cut = get_df(df)
    df_cut = create_labels(df_cut, 'heart_axis')
    df_cut = create_labels(df_cut, 'superclasses')

    train, test = train_test_split(df_cut, shuffle=True, test_size=0.2, random_state=10)

    print("Generating and collecting train ECGs (no segmentation)...")
    ecg_train_list = []
    features_train_list = []

    for _, row in train.iterrows():
        ecg, features = get_ecg(pd.DataFrame([row]), leads, 1000, f_lst)  # 1000 = full 10s window
        ecg_train_list.append(ecg)
        features_train_list.append(features)

    ecg_train = np.vstack(ecg_train_list)
    features_train = np.vstack(features_train_list)

    print("Generating and collecting test ECGs (no segmentation)...")
    ecg_test_list = []
    features_test_list = []

    for _, row in test.iterrows():
        ecg, features = get_ecg(pd.DataFrame([row]), leads, 1000, f_lst)
        ecg_test_list.append(ecg)
        features_test_list.append(features)

    ecg_test = np.vstack(ecg_test_list)
    features_test = np.vstack(features_test_list)

    minmax_train = get_minmax(ecg_train)
    minmax_test = get_minmax(ecg_test)

    df_minmax_train = pd.DataFrame(minmax_train, columns=cols)
    df_minmax_test = pd.DataFrame(minmax_test, columns=cols)

    save_img_minmax(df_minmax_train, name='train')
    save_img_minmax(df_minmax_test, name='test')

    for i in cols:
        df_minmax_train = df_minmax_train[df_minmax_train[i] < df_minmax_train[i].quantile(0.99)]
        df_minmax_test = df_minmax_test[df_minmax_test[i] < df_minmax_test[i].quantile(0.99)]

    save_img_minmax(df_minmax_train, name='train_clean')
    save_img_minmax(df_minmax_test, name='test_clean')

    indx_train = df_minmax_train.index
    indx_test = df_minmax_test.index

    ecg_train_clean = ecg_train[indx_train, :, :]
    features_train_clean = features_train[indx_train]
    ecg_test_clean = ecg_test[indx_test, :, :]
    features_test_clean = features_test[indx_test]

    print("\U0001F4BE Saving final .pkl cleaned files...")
    data_dir = ROOT / 'data_no_segmentation'
    data_dir.mkdir(parents=True, exist_ok=True)

    save_data(ecg_train_clean, data_dir / 'ecg_train_clean.pkl')
    save_data(features_train_clean, data_dir / 'features_train_clean.pkl')
    save_data(ecg_test_clean, data_dir / 'ecg_test_clean.pkl')
    save_data(features_test_clean, data_dir / 'features_test_clean.pkl')

    print("✅ Done. All cleaned datasets saved (no segmentation, 10-second ECGs).")
