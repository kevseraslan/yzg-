import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1. VERİYİ YÜKLE
print("⏳ Heart 2022 veri seti yükleniyor...")
df = pd.read_csv("heart_2022_no_nans.csv")
target_col = 'HadHeartAttack'

# 2. ÖN İŞLEME
# Kategorik sütunları sayısal değerlere (0, 1, 2...) çevir
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = pd.factorize(df[col])[0]

df = df.dropna()
X = df.drop(target_col, axis=1)
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# 3. MODELLERİ EĞİT
print("🚀 Modeller eğitiliyor (Random Forest & XGBoost)...")

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

# XGBoost
xgb_acc = 0
xgb_pred = None
if XGB_AVAILABLE:
    xgb_model = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, eval_metric='logloss', random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    xgb_acc = accuracy_score(y_test, xgb_pred)

# 4. SONUÇLARI KIYASLA (Terminal Çıktısı)
print("\n" + "="*45)
print(f"📊 ALGORİTMA PERFORMANS KIYASLAMASI")
print("="*45)
print(f"Random Forest Doğruluk: %{rf_acc*100:.2f}")
if XGB_AVAILABLE:
    print(f"XGBoost Doğruluk:       %{xgb_acc*100:.2f}")
else:
    print("XGBoost kütüphanesi yüklü değil!")
print("-" * 45)

# 5. GÖRSELLEŞTİRME: KARMAŞIKLIK MATRİSLERİ
# Eğer iki model de varsa yan yana, yoksa tek grafik çiziyoruz
fig_count = 2 if XGB_AVAILABLE else 1
fig, axes = plt.subplots(1, fig_count, figsize=(8 * fig_count, 6))

# Random Forest Matrisi
rf_cm = confusion_matrix(y_test, rf_pred)
ax_rf = axes[0] if XGB_AVAILABLE else axes
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', ax=ax_rf)
ax_rf.set_title(f'Random Forest Karmaşıklık Matrisi\n(Acc: %{rf_acc*100:.2f})')
ax_rf.set_xlabel('Tahmin Edilen')
ax_rf.set_ylabel('Gerçek Durum')
ax_rf.set_xticklabels(['Sağlıklı', 'Riskli'])
ax_rf.set_yticklabels(['Sağlıklı', 'Riskli'])

# XGBoost Matrisi (Eğer yüklüyse)
if XGB_AVAILABLE:
    xgb_cm = confusion_matrix(y_test, xgb_pred)
    sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', ax=axes[1])
    axes[1].set_title(f'XGBoost Karmaşıklık Matrisi\n(Acc: %{xgb_acc*100:.2f})')
    axes[1].set_xlabel('Tahmin Edilen')
    axes[1].set_ylabel('Gerçek Durum')
    axes[1].set_xticklabels(['Sağlıklı', 'Riskli'])
    axes[1].set_yticklabels(['Sağlıklı', 'Riskli'])

plt.tight_layout()
plt.show()

# 6. BAŞARI ORANLARI ÇUBUK GRAFİĞİ
if XGB_AVAILABLE:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=['Random Forest', 'XGBoost'], y=[rf_acc, xgb_acc], hue=['Random Forest', 'XGBoost'], palette='viridis', legend=False)
    plt.ylim(0.90, 1.0)
    plt.title('Modellerin Genel Başarı Kıyaslaması')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.show()