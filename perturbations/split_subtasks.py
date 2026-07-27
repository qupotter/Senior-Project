import pandas as pd
import re

df = pd.read_csv("ethics_deontology_perturbations.csv")

def get_subtask(scenario):
    return "role" if re.match(r"^I am\b", str(scenario).strip(), re.IGNORECASE) else "request"

df["subtask"] = df["original_scenario"].apply(get_subtask)
print(df["subtask"].value_counts())

request_df = df[df["subtask"] == "request"].copy()
role_df    = df[df["subtask"] == "role"].copy()

request_df.to_csv("ethics_deontology_perturbations_request.csv", index=False)
role_df.to_csv("ethics_deontology_perturbations_role.csv", index=False)
print(f"Request: {len(request_df)}, Role: {len(role_df)}")