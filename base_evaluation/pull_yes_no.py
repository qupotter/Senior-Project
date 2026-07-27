import pandas as pd
df = pd.read_csv("ethics_deontology_ablation_fewshot.csv")
print(df.groupby("target_label")["accuracy_rate"].mean())
print(df.groupby("target_label")["majority_correct"].mean())