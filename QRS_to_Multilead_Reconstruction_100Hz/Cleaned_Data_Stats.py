import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path

# --- إعدادات ---
PTBXL_PATH = Path('../../ptbxl')
DATABASE_CSV = PTBXL_PATH / 'ptbxl_database.csv'
SCP_STATEMENTS_CSV = PTBXL_PATH / 'scp_statements.csv'
STATS_PLOTS_DIR = Path('stats_summary')
STATS_PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# --- تحميل البيانات ---
df = pd.read_csv(DATABASE_CSV)
df['scp_codes'] = df['scp_codes'].fillna('NOT')
df['heart_axis'] = df['heart_axis'].fillna('NOT')
df['sex'] = df['sex'].fillna('unknown')
df = df[df['validated_by_human'] == True]

# --- تحويل scp_codes إلى تصنيفات ---
import ast
scp_statements = pd.read_csv(SCP_STATEMENTS_CSV, index_col=0)
scp_code_to_superclass = {k: v['diagnostic_class'] if pd.notna(v['diagnostic_class']) else 'Unknown' for k, v in scp_statements.iterrows()}

def extract_superclasses(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        return [scp_code_to_superclass.get(c, 'Unknown') for c in codes.keys()]
    except:
        return ['Unknown']

df['diagnostic_superclass'] = df['scp_codes'].apply(extract_superclasses)

# --- تحضير بيانات لكل فئة ---
def flatten_and_count(col_list):
    flat = [item for sublist in col_list for item in sublist]
    return pd.Series(flat).value_counts()

# --- رسم Pie chart ---
def plot_pie(series, title, path):
    plt.figure(figsize=(8, 8))
    series = series[series > 10]  # إظهار فقط الفئات التي لها تمثيل معقول
    plt.pie(series, labels=series.index, autopct='%1.1f%%', startangle=140)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()

# --- رسم Histogram ---
def plot_hist(series, title, xlabel, path):
    plt.figure(figsize=(8, 6))
    sns.histplot(series, bins=30, kde=True)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.savefig(path, dpi=600)
    plt.close()

# --- تنفيذ الرسومات ---
plot_hist(df['age'], 'Age Distribution', 'Age', STATS_PLOTS_DIR / 'age_distribution.png')
plot_pie(df['sex'].value_counts(), 'Sex Distribution', STATS_PLOTS_DIR / 'sex_distribution.png')
plot_pie(df['heart_axis'].value_counts(), 'Heart Axis Distribution', STATS_PLOTS_DIR / 'heart_axis_distribution.png')
plot_pie(flatten_and_count(df['diagnostic_superclass']), 'Diagnostic Superclasses Distribution', STATS_PLOTS_DIR / 'scp_superclass_distribution.png')

print(f"Saved summary plots to: {STATS_PLOTS_DIR.resolve()}")
