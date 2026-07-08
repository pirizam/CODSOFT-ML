import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings

warnings.filterwarnings('ignore')

print("Loading dataset...")
# NLP datasets often use latin-1 encoding instead of utf-8
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
except FileNotFoundError:
    print("Error: Could not find 'spam.csv'. Please make sure it is in the same folder!")
    exit()

print("Preprocessing text data...")
# The standard kaggle spam dataset uses 'v1' for labels and 'v2' for the message text
# We drop the empty 'Unnamed' columns if they exist
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to binary: 'spam' = 1, 'ham' (legitimate) = 0
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

# Separate features and target
X = df['message']
y = df['label']

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Applying TF-IDF Vectorization (Turning text into numbers)...")
vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("Training Multinomial Naive Bayes Model...")
# Naive Bayes is universally considered the best lightweight model for text classification
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

print("Generating predictions...")
y_pred = model.predict(X_test_tfidf)

print("\n" + "="*40)
print("FINAL MODEL OUTPUT")
print("="*40)
print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.4f}")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred, target_names=['Legitimate (Ham)', 'Spam']))

# ==========================================
# LIVE SPAM DETECTOR TEST
# ==========================================
print("\n" + "="*50)
print("LIVE SPAM DETECTOR TESTS")
print("="*50)

def check_message(text_message):
    print(f"Message: \"{text_message}\"")
    # We must vectorize the new text just like we did the training data
    text_vectorized = vectorizer.transform([text_message])
    prediction = model.predict(text_vectorized)[0]
    
    if prediction == 1:
        print("🚨 ALERT: This message was classified as SPAM!")
    else:
        print("✅ SAFE: This is a legitimate message.")
    print("-" * 40)

# Test 1: A normal message (maybe from Cheachii or Madan!)
check_message("what's up bro, how are you doing? I was thinking we could catch up over coffee sometime this week.")

# Test 2: A classic spam structure
check_message("i have a great offer for you! click here to claim your free gift card now!")

# Test 3: A tricky banking scam
check_message("Dear customer, your account has been compromised. Please verify your details immediately to avoid suspension.")