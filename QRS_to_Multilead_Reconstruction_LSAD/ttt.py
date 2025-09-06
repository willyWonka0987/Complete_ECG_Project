# check_input.py
import pickle
from Ensemble_Model import ECGDataset, prepare_meta_arrays, INPUT_LEADS, SEGMENT_LENGTH

# ✅ ملاحظة مهمة:
# لا يوجد أي استدعاء لـ train_model أو main هنا
# الكود فقط يحمّل الداتا ويطبع معلومات عنها

# حمل البيانات المخزنة (المفترض موجودة في ./records.pkl)
with open("records.pkl", "rb") as f:
    records = pickle.load(f)

# نجهز الـ Dataset مع use_meta=True (عشان يضم الميزات + السيغمنت)
ds = ECGDataset(records, input_leads=INPUT_LEADS, target_lead="II", use_meta=True)

# خذ أول عينة
x, y = ds[0]

print("📏 حجم الدخل (input_dim):", x.numel())
print("📏 طول السيغمنت المسطّح:", SEGMENT_LENGTH * len(INPUT_LEADS))
print("📏 طول الميزات الميتا:", x.numel() - SEGMENT_LENGTH * len(INPUT_LEADS))

# جلب أسماء الميزات الميتا
_, meta_keys = prepare_meta_arrays(records, INPUT_LEADS, use_raw=False)
print("\n📝 مفاتيح الميزات الميتا بالترتيب:")
for i, k in enumerate(meta_keys):
    print(f"{i:03d} -> {k}")
