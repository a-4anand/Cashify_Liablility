from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# ----------------------------
# Load models
# ----------------------------
model = joblib.load("risk_category_model.pkl")
model_reg = joblib.load("liability_score_model.pkl")
le = joblib.load("label_encoder.pkl")

# ----------------------------
# Load dataset exactly as in training
# ----------------------------
df = pd.read_csv("liability_dataset.csv")
data = pd.read_csv("liability_dataset.csv")

df["phone_number"] = df["phone_number"].astype(str).str.strip()

# Add encoded column if not present (for safety)
if "risk_category_encoded" not in df.columns:
    df["risk_category_encoded"] = le.transform(df["risk_category"])

# ----------------------------
# Home Page
# ----------------------------
@app.route("/")
def index():
    return render_template("index.html")


# ----------------------------
# Prediction Logic
# ----------------------------
def predict_customer(phone):
    phone = str(phone).strip()
    row = df[df["phone_number"] == phone]

    if row.empty:
        return {"error": "Phone number not found."}


    # ----------------------------
    # Build classification input (X1)
    # EXACTLY like training
    # ----------------------------
    X_class = row.drop(columns=[
        "phone_number",
        "risk_category",
        "risk_category_encoded"
    ])

    # ----------------------------
    # Build regression input (X2)
    # EXACTLY like training
    # ----------------------------
    X_reg = row.drop(columns=[
        "phone_number",
        "risk_category",
        "liability_score"
    ])

    if "risk_category_encoded" in X_reg.columns:
        X_reg = X_reg.drop(columns=["risk_category_encoded"])


    # Predict classification
    class_pred = model.predict(X_class)[0]
    class_label = le.inverse_transform([class_pred])[0]

    # Predict regression
    score_pred = model_reg.predict(X_reg)[0]

    return {
        "phone_number": phone,
        "predicted_risk_category": class_label,
        "predicted_liability_score": round(float(score_pred), 2),

        # actual dataset values
        "actual_risk_category": row["risk_category"].values[0],
        "actual_liability_score": float(row["liability_score"].values[0]),

        # NEW: fields for UI stats
        "avg_order_value": float(row["avg_order_value"].values[0]),
        "total_orders": int(row["total_orders"].values[0]),
        "return_rate": float(row["return_rate"].values[0]),
        "fraud_attempts": int(row["fraud_attempts"].values[0]),

        # NEW: explanation placeholder
        "explanation": "Top drivers: return rate, device mismatch, invoice flag, account age."
    }


# ----------------------------
# API Routes
# ----------------------------
@app.route("/predict", methods=["GET"])
def predict_get():
    phone = request.args.get("phone")
    if not phone:
        return jsonify({"error": "Phone is required"}), 400

    return jsonify(predict_customer(phone))


@app.route("/predict", methods=["POST"])
def predict_post():
    data = request.json
    phone = data.get("phone")
    if not phone:
        return jsonify({"error": "Phone is required"}), 400

    return jsonify(predict_customer(phone))


# ----------------------------
# Run
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
