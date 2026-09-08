import pandas as pd

df = pd.read_csv("crop_yield_fertilizer_300.csv")

# Rename columns (VERY IMPORTANT)
df = df.rename(columns={
    "Temperature (°C)": "temperature",
    "Rainfall (mm)": "rainfall",
    "Soil pH": "ph",
    "Nitrogen (N)": "N",
    "Phosphorus (P)": "P",
    "Potassium (K)": "K",
    "Yield (tons/ha)": "yield"
})

# Optional: drop ID & Crop (not needed for yield prediction)
df = df.drop(columns=["ID", "Crop"])

# Save clean file
df.to_csv("clean_yield_data.csv", index=False)

print("✅ Clean dataset ready!")