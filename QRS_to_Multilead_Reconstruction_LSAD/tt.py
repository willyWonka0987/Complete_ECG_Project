import pickle
with open("Segments/ptbxl/records.pkl", "rb") as f:
    records = pickle.load(f)

ages = [r["meta"].get("age", None) for r in records if r and "meta" in r]
print("Sample ages:", ages[:20])
