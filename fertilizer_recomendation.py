import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv("updated_fertilizer_data.csv")

# -------------------------------
# Split features & target
# -------------------------------
X = df[["N","P","K","temperature","humidity","ph","rainfall"]]
y = df["fertilizer"]

# -------------------------------
# Encode labels
# -------------------------------
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# -------------------------------
# Train-test split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# -------------------------------
# Train model
# -------------------------------
model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# -------------------------------
# Accuracy
# -------------------------------
accuracy = model.score(X_test, y_test)
print("✅ Accuracy:", accuracy)

# -------------------------------
# Save model
# -------------------------------
joblib.dump(model, "fertilizer_model.pkl")
joblib.dump(encoder, "fertilizer_encoder.pkl")

print("✅ Model saved successfully!")