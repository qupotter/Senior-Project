import pandas as pd

v1 = pd.read_csv("ethics_deontology_minimal_counterfactuals.csv")
v2 = pd.read_csv("ethics_deontology_minimal_counterfactuals_v2.csv")

combined = pd.concat([v1, v2], ignore_index=True)
aligned = combined[combined["token_aligned"]].copy()

print(f"Combined total: {len(combined)}")
print(f"Aligned: {len(aligned)}")
print(aligned["swapped_factor"].value_counts())

aligned.to_csv("ethics_deontology_aligned_counterfactuals.csv", index=False)