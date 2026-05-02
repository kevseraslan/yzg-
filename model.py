import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. VERİYİ YÜKLE
print("⏳ Heart 2022 veri seti yükleniyor...")
df = pd.read_csv("heart_2022_no_nans.csv")
target_col = 'HadHeartAttack'

# 2. ÖN İŞLEME
# Kategorik verileri sayısal değerlere dönüştürür
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = pd.factorize(df[col])[0]

df = df.dropna()
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. MODELLERİ EĞİT
print("🚀 Modeller eğitiliyor...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost
xgb_acc = 0
if XGB_AVAILABLE:
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)

# --- 4. YENİ: RİSK SKORU HESAPLAMA (predict_proba) ---
print("\n" + "="*45)
print(f"🔎 ÖRNEK HASTA ANALİZİ (Risk Skoru Tahmini)")
print("="*45)

# Test setinden bir örnek (10. hasta) seçelim
index = 10
ornek_hasta = X_test.iloc[[index]]
gercek_durum = "Riskli" if y_test.iloc[index] == 1 else "Sağlıklı"

# predict_proba: [Sağlıklı Olasılığı, Riskli Olasılığı] döner. 
# Biz [0][1] diyerek 'Riskli' (1) olma ihtimalini alıyoruz.
rf_risk_skoru = rf_model.predict_proba(ornek_hasta)[0][1] * 100

print(f"Seçilen Hastanın Gerçek Durumu: {gercek_durum}")
print(f"Random Forest Risk Tahmini: %{rf_risk_skoru:.2f}")

if XGB_AVAILABLE:
    xgb_risk_skoru = xgb_model.predict_proba(ornek_hasta)[0][1] * 100
    print(f"XGBoost Risk Tahmini:       %{xgb_risk_skoru:.2f}")

# 5. GENEL SONUÇLAR[cite: 1, 2]
print("\n" + "="*45)
print(f"📊 MODEL PERFORMANS SONUÇLARI")
print("="*45)
print(f"Random Forest Doğruluk: %{rf_acc*100:.2f}")
if XGB_AVAILABLE:
    print(f"XGBoost Doğruluk:       %{xgb_acc*100:.2f}")
print("-" * 45)

# 6. GÖRSELLEŞTİRME[cite: 1, 2]
fig_count = 2 if XGB_AVAILABLE else 1
fig, axes = plt.subplots(1, fig_count, figsize=(8 * fig_count, 6))

rf_cm = confusion_matrix(y_test, rf_pred)
ax_rf = axes[0] if XGB_AVAILABLE else axes
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax_rf)
ax_rf.set_title('Random Forest Karmaşıklık Matrisi')
ax_rf.set_xticklabels(['Sağlıklı', 'Riskli']); ax_rf.set_yticklabels(['Sağlıklı', 'Riskli'])

if XGB_AVAILABLE:
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title('XGBoost Karmaşıklık Matrisi')
    axes[1].set_xticklabels(['Sağlıklı', 'Riskli']); axes[1].set_yticklabels(['Sağlıklı', 'Riskli'])

plt.tight_layout()
plt.show()