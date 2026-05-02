from __future__ import annotations

from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, request, send_from_directory
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


ROOT = Path(__file__).resolve().parent
DATASET_PATH = ROOT / "heart_2022_no_nans.csv"

app = Flask(__name__)

MODEL: Pipeline | None = None


def age_to_category(age: int) -> str:
    if age < 25:
        return "Age 18 to 24"
    if age < 30:
        return "Age 25 to 29"
    if age < 35:
        return "Age 30 to 34"
    if age < 40:
        return "Age 35 to 39"
    if age < 45:
        return "Age 40 to 44"
    if age < 50:
        return "Age 45 to 49"
    if age < 55:
        return "Age 50 to 54"
    if age < 60:
        return "Age 55 to 59"
    if age < 65:
        return "Age 60 to 64"
    if age < 70:
        return "Age 65 to 69"
    if age < 75:
        return "Age 70 to 74"
    if age < 80:
        return "Age 75 to 79"
    return "Age 80 or older"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def train_model() -> Pipeline:
    df = pd.read_csv(DATASET_PATH)
    features = ["Sex", "AgeCategory", "BMI", "HeightInMeters", "WeightInKilograms"]

    X = df[features].copy()
    y = (df["HadHeartAttack"] == "Yes").astype(int)

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["Sex", "AgeCategory"]),
            ("num", "passthrough", ["BMI", "HeightInMeters", "WeightInKilograms"]),
        ]
    )

    model = Pipeline(
        steps=[
            ("prep", preprocessor),
            ("clf", RandomForestClassifier(n_estimators=220, random_state=42, n_jobs=-1)),
        ]
    )
    model.fit(X, y)
    return model


def get_model() -> Pipeline:
    global MODEL
    if MODEL is None:
        MODEL = train_model()
    return MODEL


def build_actions(risk: float) -> list[str]:
    if risk >= 60:
        return [
            "Kardiyoloji uzmanı ile en kısa sürede değerlendirme planlayın.",
            "Tansiyon ve kolesterol takibini günlük kayıtla sürdürün.",
            "Tuz ve doymuş yağ tüketimini azaltıp kalp dostu beslenmeye geçin.",
        ]
    if risk >= 30:
        return [
            "Aile hekimi kontrolünü ihmal etmeyin ve haftalık tansiyon takibi yapın.",
            "Haftada en az 150 dakika orta tempo fiziksel aktivite hedefleyin.",
            "Beslenmede lif oranını artırıp işlenmiş gıdayı azaltın.",
        ]
    return [
        "Mevcut sağlıklı yaşam alışkanlıklarını sürdürün.",
        "Yılda en az bir kez rutin kardiyovasküler kontrol yaptırın.",
        "Düzenli uyku ve günlük hareket seviyesini koruyun.",
    ]


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.post("/api/predict")
def predict():
    payload = request.get_json(silent=True) or {}

    try:
        age = int(payload.get("age"))
        sex = str(payload.get("sex", "erkek")).lower()
        sys = float(payload.get("sys"))
        dia = float(payload.get("dia"))
        chol = float(payload.get("chol"))
        height_cm = float(payload.get("heightCm"))
        weight_kg = float(payload.get("weightKg"))
        family_history = bool(payload.get("familyHistory", False))
    except (TypeError, ValueError):
        return jsonify({"error": "Lutfen tum alanlara gecerli deger girin."}), 400

    if height_cm <= 0 or weight_kg <= 0:
        return jsonify({"error": "Boy ve kilo sifirdan buyuk olmalidir."}), 400

    bmi = weight_kg / ((height_cm / 100) ** 2)
    map_value = (sys + 2 * dia) / 3

    sex_value = "Male" if sex == "erkek" else "Female"
    features_df = pd.DataFrame(
        [
            {
                "Sex": sex_value,
                "AgeCategory": age_to_category(age),
                "BMI": bmi,
                "HeightInMeters": height_cm / 100,
                "WeightInKilograms": weight_kg,
            }
        ]
    )

    model = get_model()
    base_risk = float(model.predict_proba(features_df)[0][1])

    risk_adjust = 0.0
    if sys >= 140:
        risk_adjust += 0.08
    elif sys >= 130:
        risk_adjust += 0.04

    if dia >= 90:
        risk_adjust += 0.06
    elif dia >= 85:
        risk_adjust += 0.03

    if chol >= 240:
        risk_adjust += 0.08
    elif chol >= 200:
        risk_adjust += 0.04

    if family_history:
        risk_adjust += 0.07

    if map_value >= 105:
        risk_adjust += 0.04

    total_risk = clamp(base_risk + risk_adjust, 0.01, 0.99)
    risk_pct = round(total_risk * 100, 1)

    if risk_pct >= 60:
        level = "Yuksek Risk"
        status = "Kardiyoloji degerlendirmesi onerilir"
    elif risk_pct >= 30:
        level = "Orta Risk"
        status = "Yakin takip onerilir"
    else:
        level = "Dusuk Risk"
        status = "Rutin takip yeterli"

    return jsonify(
        {
            "riskScore": risk_pct,
            "riskLevel": level,
            "status": status,
            "bmi": round(bmi, 1),
            "map": round(map_value),
            "actions": build_actions(risk_pct),
        }
    )


if __name__ == "__main__":
    get_model()
    app.run(host="127.0.0.1", port=5000, debug=False)
