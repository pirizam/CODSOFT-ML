import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import warnings

warnings.filterwarnings('ignore')

print("Loading datasets...")
train = pd.read_csv("fraudTrain.csv")
test = pd.read_csv("fraudTest.csv")
df = pd.concat([train, test]).reset_index(drop=True)

print("\n--- EXPLORATORY DATA ANALYSIS ---")
fraud_count = df['is_fraud'].value_counts()
print(f"Total Transactions: {len(df)}")
print(f"Legitimate: {fraud_count[0]} | Fraudulent: {fraud_count[1]}")
print(f"Fraud Percentage: {(fraud_count[1]/len(df))*100:.3f}%\n")

print("Preprocessing & Feature Engineering...")
# Drop non-numeric/high-cardinality columns
columns_to_drop = ['trans_date_trans_time', 'dob', 'first', 'last', 'street', 'city', 'trans_num']
df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

# Encode categoricals
encoder = LabelEncoder()
categorical_cols = ['merchant', 'category', 'gender', 'state', 'job']
for col in categorical_cols:
    if col in df.columns:
        df[col] = encoder.fit_transform(df[col])

print("Applying Fast Undersampling...")
# Separate the classes
legit = df[df['is_fraud'] == 0]
fraud = df[df['is_fraud'] == 1]

# Undersample the legitimate transactions to match the fraud count
legit_sampled = legit.sample(n=len(fraud), random_state=42)
balanced_df = pd.concat([legit_sampled, fraud]).sample(frac=1, random_state=42) # Shuffle

X = balanced_df.drop('is_fraud', axis=1)
y = balanced_df['is_fraud']

print("Splitting & Scaling Data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
}

results = {}
confusion_matrices = {}

print("\n--- TRAINING MULTIPLE MODELS ---")
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, preds)
    roc = roc_auc_score(y_test, preds)
    
    results[name] = {'Accuracy': acc, 'ROC AUC': roc}
    confusion_matrices[name] = confusion_matrix(y_test, preds)
    print(f"✅ {name} - Accuracy: {acc:.4f} | ROC AUC: {roc:.4f}")

# ==========================================
# GENERATE PROFESSIONAL DASHBOARD
# ==========================================
print("\nGenerating Visual Dashboard...")

fig = plt.figure(figsize=(18, 10))
fig.suptitle('Credit Card Fraud Detection Pipeline Results', fontsize=20, fontweight='bold', y=0.98)

# 1. Bar Chart for Accuracies
ax1 = plt.subplot(2, 3, (1, 3))
names = list(results.keys())
accuracies = [results[m]['Accuracy'] * 100 for m in names]
colors = ['#ff9999', '#66b3ff', '#00cc44']

bars = ax1.bar(names, accuracies, color=colors, width=0.4)
ax1.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_ylim(0, 100)

for bar, val in zip(bars, accuracies):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             f'{val:.2f}%', ha='center', fontweight='bold', fontsize=12)

# 2. Confusion Matrices
for i, name in enumerate(names):
    ax = plt.subplot(2, 3, i + 4)
    sns.heatmap(confusion_matrices[name], annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Legit', 'Fraud'], yticklabels=['Legit', 'Fraud'], ax=ax, annot_kws={"size": 14})
    ax.set_title(f'{name}\nConfusion Matrix', fontsize=12, fontweight='bold')
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('fraud_detection_dashboard.png', dpi=300)
print("🎯 SUCCESS! Dashboard saved as 'fraud_detection_dashboard.png'")
print("="*50)