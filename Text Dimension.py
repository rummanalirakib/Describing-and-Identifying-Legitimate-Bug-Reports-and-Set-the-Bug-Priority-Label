from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import ComplementNB

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
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Step 1: Tokenization and Features Extraction using CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
print(X_train_vectorized)
print(y_train)

# Step 2: Training the Naive Bayes Classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)
complement_nb_classifier  = ComplementNB()
complement_nb_classifier.fit(X_train_vectorized, y_train)
# Train the classifier on the training data

# Step 3: Scoring - Training Dataset
training_scores = classifier.predict_proba(X_train_vectorized)[:, 1]
training_scores_complement = complement_nb_classifier.predict_proba(X_train_vectorized)[:, 1]
print(training_scores_complement)

# Step 4: Scoring - Testing Dataset
testing_scores = classifier.predict_proba(X_test_vectorized)[:, 1]
testing_scores_complement = complement_nb_classifier.predict_proba(X_test_vectorized)[:, 1]

# Displaying scores
print("\nTraining Scores:")
for i, score in enumerate(training_scores):
    # Use reshape(-1, 1) to convert the 1D array to a 2D array
    predicted_label = classifier.predict(X_train_vectorized[i, :].reshape(1, -1))[0]
    print(f"Bug Report {i+1}: {score:.4f} (Predicted Label: {predicted_label})")

print("\nTesting Scores:")
for i, score in enumerate(testing_scores):
    # Use reshape(-1, 1) to convert the 1D array to a 2D array
    predicted_label = classifier.predict(X_test_vectorized[i, :].reshape(1, -1))[0]
    print(f"Bug Report {i+1}: {score:.4f} (Predicted Label: {predicted_label})")


# Evaluating the classifier
#accuracy = accuracy_score(y_test, classifier.predict(X_test_vectorized))
accuracy = accuracy_score(y_test, testing_scores)
print(f"\nAccuracy on Testing Dataset: {accuracy:.4f}")
