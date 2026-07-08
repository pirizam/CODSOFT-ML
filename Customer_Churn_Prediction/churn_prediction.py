import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import xgboost as xgb
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
# Make sure Churn_Modelling.csv is in the same folder as this script
df = pd.read_csv('Churn_Modelling.csv')

print("Preprocessing data...")
# Drop useless columns for prediction
df.drop(['RowNumber', 'CustomerId', 'Surname'], axis=1, inplace=True, errors='ignore')

# Encode categorical variables
encoder = LabelEncoder()
df['Gender'] = encoder.fit_transform(df['Gender'])
df['Geography'] = encoder.fit_transform(df['Geography'])

# Separate features and target
X = df.drop('Exited', axis=1)
y = df['Exited']

print("Splitting and Scaling data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Training XGBoost Classifier...")
xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=4, # handles imbalance 
    random_state=42,
    eval_metric='logloss',
    verbosity=0
)

xgb_model.fit(X_train_scaled, y_train)
xgb_preds = xgb_model.predict(X_test_scaled)

acc = accuracy_score(y_test, xgb_preds)
roc = roc_auc_score(y_test, xgb_preds)

print(f"\n✅ XGBoost Accuracy : {acc:.4f}")
print(f"✅ XGBoost ROC AUC  : {roc:.4f}")

print("\n--- Classification Report ---")
print(classification_report(y_test, xgb_preds, target_names=['Stayed', 'Churned']))

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, xgb_preds))

# ==========================================
# CUSTOMER CHURN PREDICTION SYSTEM TESTS
# ==========================================
print("\n" + "="*50)
print("CUSTOMER CHURN PREDICTION SYSTEM TESTS")
print("="*50)

def predict_churn(credit_score, geography, gender, age, tenure, balance, num_products, has_crcard, is_active, salary):
    # Encode inputs
    geo_encoded = 0 if geography == 'France' else (1 if geography == 'Germany' else 2)
    gender_encoded = 1 if gender == 'Male' else 0
    
    # Create input array
    input_data = np.array([[credit_score, geo_encoded, gender_encoded, age, tenure, balance, num_products, has_crcard, is_active, salary]])
    input_scaled = scaler.transform(input_data)
    
    # Predict probabilities
    prob = xgb_model.predict_proba(input_scaled)[0]
    stay_prob = prob[0] * 100
    churn_prob = prob[1] * 100
    
    print(f"\nProfile: {age}yr old {gender} from {geography}, Balance: ${balance}, Active: {'Yes' if is_active else 'No'}")
    print(f"Stay % : {stay_prob:.2f}%")
    print(f"Churn % : {churn_prob:.2f}%")
    
    if churn_prob > 50:
        print("🚨 RESULT: CUSTOMER WILL CHURN!")
        print("→ Action: Send retention offer immediately!")
    else:
        print("✅ RESULT: CUSTOMER WILL STAY!")
    print("-" * 30)

print("\nTest 1 — Young active customer:")
predict_churn(700, 'France', 'Male', 30, 5, 50000, 2, 1, 1, 60000)

print("\nTest 2 — Old inactive German customer:")
predict_churn(400, 'Germany', 'Female', 55, 1, 150000, 1, 0, 0, 80000)