import pandas as pd

df = pd.read_csv("Crop_recommendation.csv")

def assign_fertilizer(row):
    if row["N"] < 50:
        return "Urea"
    elif row["P"] < 50:
        return "DAP"
    elif row["K"] < 50:
        return "MOP"
    else:
        return "NPK"

df["fertilizer"] = df.apply(assign_fertilizer, axis=1)

df.to_csv("updated_fertilizer_data.csv", index=False)

print("✅ Fertilizer column added!")