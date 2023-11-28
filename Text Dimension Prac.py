from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB
import pandas as pd
import numpy as np
from sklearn.svm import SVC

# Training data
bug_reports = [
    {"text": "Critical bug in login functionality.", "label": "valid"},
    {"text": "Spelling mistake in user profile.", "label": "invalid"},
    {"text": "App crashes during payment processing.", "label": "valid"},
    {"text": "Website not working.", "label": "ivalid"},
    {"text": "firefox issue.", "label": "valid"},
    {"text": "bugzilla not working.", "label": "valid"},
]

# Extracting text and labels
texts = [report["text"] for report in bug_reports]
labels = [report["label"] for report in bug_reports]

# Split the data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 1: Tokenization and Features Extraction using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer)
text_vectorized = vectorizer.fit_transform(texts)


# Step 2: Training the Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(text_vectorized.toarray(), labels)

multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(text_vectorized, labels)

complement_nb_classifier  = ComplementNB()
complement_nb_classifier.fit(text_vectorized, labels)

discriminative_multinomial_nb_classifier = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier.fit(text_vectorized, labels)
# Train the classifier on the training data

# Step 3: Scoring - Training Dataset
multinomial_nb_training_scores = multinomial_nb_classifier.predict_proba(text_vectorized)[:, 1]
complement_nb_training_scores = complement_nb_classifier.predict_proba(text_vectorized)[:, 1]
discriminative_multinomial_nb_training_scores = discriminative_multinomial_nb_classifier.predict_proba(text_vectorized)[:, 1]
nb_classifier_training_scores = nb_classifier.predict_proba(text_vectorized.toarray())[:, 1]
print(complement_nb_training_scores)


df_multinomial_nb = pd.DataFrame(multinomial_nb_training_scores)
df_complement_nb = pd.DataFrame(complement_nb_training_scores)
df_discriminative_multinomial_nb = pd.DataFrame(discriminative_multinomial_nb_training_scores)
df_nb_classifier = pd.DataFrame(nb_classifier_training_scores)
merged_df = pd.concat([df_multinomial_nb, df_complement_nb, df_discriminative_multinomial_nb, df_nb_classifier], axis=1)
print(merged_df)

X_train, X_test, y_train, y_test = train_test_split(merged_df, labels, test_size=0.2, random_state=42)

svc_model = SVC(kernel='linear')
svc_model.fit(X_train, y_train)
y_pred = svc_model.predict(X_test)

# Evaluate accuracy using binary predictions
accuracy = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on Testing Dataset: {accuracy:.4f}")
