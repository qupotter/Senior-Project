import pandas as pd
import re

df = pd.read_csv("ethics_deontology_ablation_fewshot.csv")

def get_subtask(scenario):
    if re.match(r"^I am\b", str(scenario).strip(), re.IGNORECASE):
        return "role"
    return "request"

df["subtask"] = df["scenario"].apply(get_subtask)
print(df["subtask"].value_counts())
print(df.groupby(["subtask", "target_label"])["accuracy_rate"].mean())
print(df.groupby("subtask")["majority_correct"].mean())