
from src.data_preprocessing import load_and_preprocess_data
from src.model import train_and_save_model, evaluate_model
import joblib

# Load and preprocess the data
X_train, X_test, y_train, y_test, scaler = load_and_preprocess_data("data/heart.csv")

# Train and save the model
model = train_and_save_model(X_train, y_train, "models/heart_model.pkl")

# Save the scaler
joblib.dump(scaler, "models/scaler.pkl")

# Evaluate the model
evaluate_model(model, X_test, y_test)
