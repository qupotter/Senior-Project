import pandas as pd

INPUT_CSV = "ablations_data/ethics_deontology_ablation_fewshot.csv"
OUTPUT_CSV = "ethics_deontology_ablation_fewshot_cleaned.csv"

df = pd.read_csv(INPUT_CSV)

df["excuse"] = (
    df["excuse"]
    .fillna("")
    .str.replace(r"\s*Answer format:.*$", "", regex=True)
    .str.strip()
)

df.to_csv(OUTPUT_CSV, index=False)
print(f"Wrote cleaned file: {OUTPUT_CSV}")