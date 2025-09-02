import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import re

# مسارات الملفات
dataset_dir = Path('../../dataset_new')
condition_file = dataset_dir / 'ConditionNames_SNOMED-CT.csv'
records_txt = dataset_dir / 'RECORDS'

# مجلد الخرج بجانب السكريبت
code_dir = Path(os.path.dirname(os.path.abspath(__file__)))
disease_img_dir = code_dir / 'Diseases_Dist'
os.makedirs(disease_img_dir, exist_ok=True)

# قراءة ملف الأمراض مع معالجة BOM والفراغات
cond_df = pd.read_csv(condition_file, encoding='utf-8-sig')
cond_df.columns = cond_df.columns.str.strip()
cond_df['Snomed_CT'] = cond_df['Snomed_CT'].astype(str).str.strip()

# بناء قواميس مطابقة بدون فراغات
snomed_to_full = dict(zip(cond_df['Snomed_CT'], cond_df['Full Name']))
snomed_to_acronym = dict(zip(cond_df['Snomed_CT'], cond_df['Acronym Name']))

# قراءة ملفات hea
with open(records_txt, 'r') as f:
    record_paths = [line.strip() for line in f if line.strip()]
hea_files = []
for rel_path in record_paths:
    record_dir = dataset_dir / rel_path
    hea_files.extend(list(record_dir.glob("*.hea")))

# استخراج أكواد التشخيصات من كل ملف hea مع إزالة الفراغات
all_dx_codes = []
for hea_path in hea_files:
    with open(hea_path, 'r') as hfile:
        for line in hfile:
            if line.startswith('#Dx:'):
                codes = [c.strip() for c in line.replace('#Dx:', '').split(',') if c.strip()]
                all_dx_codes.extend(codes)
                break

# طباعة أول 10 أكواد للتحقق
print("أول 10 أكواد مستخرجة من hea:", all_dx_codes[:10])
print("أول 10 أكواد من جدول الأمراض:", list(snomed_to_full.keys())[:10])

# حساب عدد كل مرض - فقط الأكواد المعرفة في جدول الأمراض
disease_counts = Counter([c for c in all_dx_codes if c in snomed_to_full])
disease_summary = []
for code, count in disease_counts.items():
    acronym = snomed_to_acronym[code]
    full_name = snomed_to_full[code]
    disease_summary.append({'Snomed_CT': code, 'Acronym Name': acronym, 'Full Name': full_name, 'Count': count})
df_summary = pd.DataFrame(disease_summary).sort_values('Count', ascending=False)

# Pie Chart
top_n = 10
top_diseases = df_summary.head(top_n)
plt.figure(figsize=(8,8))
plt.pie(top_diseases['Count'], labels=top_diseases['Full Name'], autopct='%1.1f%%', startangle=140)
plt.title(f'Top {top_n} Diseases Distribution')
plt.tight_layout()
plt.savefig(disease_img_dir / 'top_diseases_pie.png')
plt.close()

# Histogram لكل الأمراض
plt.figure(figsize=(16,7))
plt.bar(df_summary['Full Name'], df_summary['Count'], color='dodgerblue')
plt.xticks(rotation=90, fontsize=8)
plt.ylabel('Count')
plt.title('Disease Distribution Histogram')
plt.tight_layout()
plt.savefig(disease_img_dir / 'disease_histogram.png')
plt.close()

print(f'تم حفظ الرسوم في {disease_img_dir.resolve()}')
