# Bill Categorization - Supervised Machine Learning Model
# Using Multinomial Naive Bayes with TF-IDF features

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import pickle
import joblib

# ============================================
# TRAINING DATA (200+ examples)
# ============================================

training_data = [
    # Utilities
    ("Electric bill", "Utilities"),
    ("Electricity payment", "Utilities"),
    ("Power company", "Utilities"),
    ("Gas bill", "Utilities"),
    ("Water bill", "Utilities"),
    ("Sewage service", "Utilities"),
    ("Trash collection", "Utilities"),
    ("Waste management", "Utilities"),
    ("Energy bill", "Utilities"),
    ("Duke Energy", "Utilities"),
    ("PG&E bill", "Utilities"),
    ("ConEd electric", "Utilities"),
    ("Xcel Energy", "Utilities"),
    ("Municipal water", "Utilities"),
    ("City utilities", "Utilities"),
    
    # Housing
    ("Rent payment", "Housing"),
    ("Apartment rent", "Housing"),
    ("House rent", "Housing"),
    ("Mortgage payment", "Housing"),
    ("Home loan", "Housing"),
    ("Property tax", "Housing"),
    ("HOA fees", "Housing"),
    ("Homeowners association", "Housing"),
    ("Condo fees", "Housing"),
    ("Property management", "Housing"),
    ("Lease payment", "Housing"),
    ("Monthly rent", "Housing"),
    ("Housing payment", "Housing"),
    ("Landlord payment", "Housing"),
    
    # Transportation
    ("Car payment", "Transportation"),
    ("Auto loan", "Transportation"),
    ("Vehicle payment", "Transportation"),
    ("Gas for car", "Transportation"),
    ("Fuel expense", "Transportation"),
    ("Car maintenance", "Transportation"),
    ("Oil change", "Transportation"),
    ("Tire replacement", "Transportation"),
    ("Uber rides", "Transportation"),
    ("Lyft payment", "Transportation"),
    ("Public transit", "Transportation"),
    ("Metro card", "Transportation"),
    ("Bus pass", "Transportation"),
    ("Train ticket", "Transportation"),
    ("Parking fees", "Transportation"),
    ("Toll payment", "Transportation"),
    
    # Internet & Phone
    ("Internet bill", "Internet & Phone"),
    ("WiFi service", "Internet & Phone"),
    ("Broadband payment", "Internet & Phone"),
    ("Cable internet", "Internet & Phone"),
    ("Fiber optic", "Internet & Phone"),
    ("Phone bill", "Internet & Phone"),
    ("Mobile phone", "Internet & Phone"),
    ("Cell phone plan", "Internet & Phone"),
    ("Verizon wireless", "Internet & Phone"),
    ("AT&T phone", "Internet & Phone"),
    ("T-Mobile bill", "Internet & Phone"),
    ("Comcast internet", "Internet & Phone"),
    ("Xfinity service", "Internet & Phone"),
    ("Spectrum internet", "Internet & Phone"),
    ("Cox cable", "Internet & Phone"),
    
    # Insurance
    ("Car insurance", "Insurance"),
    ("Auto insurance", "Insurance"),
    ("Vehicle insurance", "Insurance"),
    ("Health insurance", "Insurance"),
    ("Medical insurance", "Insurance"),
    ("Life insurance", "Insurance"),
    ("Home insurance", "Insurance"),
    ("Homeowners insurance", "Insurance"),
    ("Renters insurance", "Insurance"),
    ("Dental insurance", "Insurance"),
    ("Vision insurance", "Insurance"),
    ("Travel insurance", "Insurance"),
    ("Pet insurance", "Insurance"),
    ("Geico payment", "Insurance"),
    ("State Farm", "Insurance"),
    ("Allstate premium", "Insurance"),
    
    # Subscriptions
    ("Netflix subscription", "Subscriptions"),
    ("Hulu payment", "Subscriptions"),
    ("Disney Plus", "Subscriptions"),
    ("Amazon Prime", "Subscriptions"),
    ("Spotify premium", "Subscriptions"),
    ("Apple Music", "Subscriptions"),
    ("YouTube Premium", "Subscriptions"),
    ("HBO Max", "Subscriptions"),
    ("Paramount Plus", "Subscriptions"),
    ("Peacock subscription", "Subscriptions"),
    ("Adobe Creative Cloud", "Subscriptions"),
    ("Microsoft 365", "Subscriptions"),
    ("Office subscription", "Subscriptions"),
    ("iCloud storage", "Subscriptions"),
    ("Dropbox subscription", "Subscriptions"),
    ("Google One", "Subscriptions"),
    ("PlayStation Plus", "Subscriptions"),
    ("Xbox Game Pass", "Subscriptions"),
    ("Nintendo Online", "Subscriptions"),
    ("Gym membership", "Subscriptions"),
    ("Fitness subscription", "Subscriptions"),
    
    # Food & Groceries
    ("Grocery store", "Food & Groceries"),
    ("Supermarket", "Food & Groceries"),
    ("Walmart groceries", "Food & Groceries"),
    ("Target food", "Food & Groceries"),
    ("Costco shopping", "Food & Groceries"),
    ("Whole Foods", "Food & Groceries"),
    ("Trader Joes", "Food & Groceries"),
    ("Kroger", "Food & Groceries"),
    ("Safeway", "Food & Groceries"),
    ("Food delivery", "Food & Groceries"),
    ("DoorDash order", "Food & Groceries"),
    ("Uber Eats", "Food & Groceries"),
    ("GrubHub delivery", "Food & Groceries"),
    ("Restaurant bill", "Food & Groceries"),
    ("Dining out", "Food & Groceries"),
    ("Takeout food", "Food & Groceries"),
    
    # Healthcare
    ("Doctor visit", "Healthcare"),
    ("Medical appointment", "Healthcare"),
    ("Hospital bill", "Healthcare"),
    ("Prescription medication", "Healthcare"),
    ("Pharmacy payment", "Healthcare"),
    ("CVS prescription", "Healthcare"),
    ("Walgreens meds", "Healthcare"),
    ("Dentist appointment", "Healthcare"),
    ("Dental cleaning", "Healthcare"),
    ("Eye exam", "Healthcare"),
    ("Optometrist visit", "Healthcare"),
    ("Physical therapy", "Healthcare"),
    ("Chiropractor", "Healthcare"),
    ("Mental health counseling", "Healthcare"),
    ("Therapy session", "Healthcare"),
    ("Lab tests", "Healthcare"),
    ("X-ray payment", "Healthcare"),
    
    # Education
    ("Tuition payment", "Education"),
    ("School fees", "Education"),
    ("College tuition", "Education"),
    ("University payment", "Education"),
    ("Student loan", "Education"),
    ("Textbooks", "Education"),
    ("School supplies", "Education"),
    ("Online course", "Education"),
    ("Udemy course", "Education"),
    ("Coursera subscription", "Education"),
    ("Khan Academy", "Education"),
    ("Study materials", "Education"),
    ("Private tutoring", "Education"),
    ("Music lessons", "Education"),
    ("Art classes", "Education"),
    
    # Entertainment
    ("Movie tickets", "Entertainment"),
    ("Concert tickets", "Entertainment"),
    ("Theater tickets", "Entertainment"),
    ("Sports event", "Entertainment"),
    ("Game purchase", "Entertainment"),
    ("Video games", "Entertainment"),
    ("Books purchase", "Entertainment"),
    ("Kindle books", "Entertainment"),
    ("Audible subscription", "Entertainment"),
    ("Hobby supplies", "Entertainment"),
    ("Craft materials", "Entertainment"),
    ("Streaming rental", "Entertainment"),
    ("iTunes purchase", "Entertainment"),
    ("Google Play", "Entertainment"),
    
    # Other
    ("Miscellaneous expense", "Other"),
    ("Random purchase", "Other"),
    ("General payment", "Other"),
    ("Various items", "Other"),
    ("Charity donation", "Other"),
    ("Gift purchase", "Other"),
    ("Personal care", "Other"),
    ("Haircut", "Other"),
    ("Salon visit", "Other"),
    ("Dry cleaning", "Other"),
    ("Laundry service", "Other"),
]

# ============================================
# PREPARE DATA
# ============================================

# Convert to DataFrame
df = pd.DataFrame(training_data, columns=['bill_name', 'category'])

# Add amount-based features (optional - you can include amount if available)
# For now, we'll focus on text-based features

X = df['bill_name']  # Features (bill names)
y = df['category']   # Labels (categories)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"\nCategory distribution:")
print(y.value_counts())

# ============================================
# MODEL 1: NAIVE BAYES (RECOMMENDED)
# ============================================

print("\n" + "="*50)
print("TRAINING MODEL 1: MULTINOMIAL NAIVE BAYES")
print("="*50)

# Create pipeline: TF-IDF + Naive Bayes
model_nb = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),  # Unigrams and bigrams
        max_features=500
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Train the model
model_nb.fit(X_train, y_train)

# Make predictions
y_pred_nb = model_nb.predict(X_test)

# Evaluate
print("\nNaive Bayes Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_nb):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_nb))

# ============================================
# MODEL 2: RANDOM FOREST
# ============================================

print("\n" + "="*50)
print("TRAINING MODEL 2: RANDOM FOREST")
print("="*50)

model_rf = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=500
    )),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    ))
])

model_rf.fit(X_train, y_train)
y_pred_rf = model_rf.predict(X_test)

print("\nRandom Forest Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_rf))

# ============================================
# MODEL 3: LOGISTIC REGRESSION
# ============================================

print("\n" + "="*50)
print("TRAINING MODEL 3: LOGISTIC REGRESSION")
print("="*50)

model_lr = Pipeline([
    ('tfidf', TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        max_features=500
    )),
    ('classifier', LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0
    ))
])

model_lr.fit(X_train, y_train)
y_pred_lr = model_lr.predict(X_test)

print("\nLogistic Regression Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.2%}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_lr))

# ============================================
# SAVE THE BEST MODEL
# ============================================

# Choose Naive Bayes (usually best for text classification)
best_model = model_rf

# Save using joblib (recommended for sklearn)
joblib.dump(best_model, 'bill_categorization_model.pkl')
print("\n✅ Model saved as 'bill_categorization_model.pkl'")

# Save category list
categories = list(y.unique())
with open('categories.pkl', 'wb') as f:
    pickle.dump(categories, f)
print("✅ Categories saved as 'categories.pkl'")

# ============================================
# TEST THE MODEL WITH NEW EXAMPLES
# ============================================

print("\n" + "="*50)
print("TESTING WITH NEW EXAMPLES")
print("="*50)

test_examples = [
    "Netflix monthly payment",
    "Electric company bill",
    "Apartment rent payment",
    "Verizon wireless phone",
    "Geico car insurance",
    "Walmart grocery shopping",
    "Doctor appointment copay",
    "Spotify premium subscription"
]

for example in test_examples:
    prediction = best_model.predict([example])[0]
    probabilities = best_model.predict_proba([example])[0]
    confidence = max(probabilities)
    print(f"\n'{example}'")
    print(f"  → Category: {prediction}")
    print(f"  → Confidence: {confidence:.2%}")

print("\n" + "="*50)
print("MODEL TRAINING COMPLETE!")
print("="*50)