import csv
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import ComplementNB
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

def preprocess_text(raw_text):
    # Convert to lowercase
    lowercase_text = raw_text.lower()

    # Remove punctuation
    cleaned_text = re.sub(r'[^\w\s]', '', lowercase_text)

    # Tokenize the text
    tokens = word_tokenize(cleaned_text)

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    # Join the tokens back into a preprocessed text
    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text


def LableMapping(resolution):
    if resolution == 'FIXED' or resolution=='WONTFIX':
        return "Valid Bug"
    else:
        return "Invalid Bug"



summary = []
finalLabels = []
description = []
# Open the CSV file for reading
with open('New DataSet.csv', 'r', encoding='ISO-8859-1') as csv_file:
    csv_reader = csv.reader(csv_file)
    csv_data = list(csv_reader)

    for i in range(len(csv_data)):
        finalLabels.insert(i, LableMapping(csv_data[i][4]))
        summary.insert(i, preprocess_text(csv_data[i][1]))
        description.insert(i, preprocess_text(csv_data[i][8]))


# Split the data into training and testing sets
vectorizer = CountVectorizer(stop_words='english')
text_vectorized = vectorizer.fit_transform(summary)

description_vectorized = vectorizer.fit_transform(description)
#print(description_vectorized)


# Step 2: Training the Naive Bayes Classifier
nb_classifier = GaussianNB()
nb_classifier.fit(text_vectorized.toarray(), finalLabels)

multinomial_nb_classifier = MultinomialNB()
multinomial_nb_classifier.fit(text_vectorized, finalLabels)

complement_nb_classifier  = ComplementNB()
complement_nb_classifier.fit(text_vectorized, finalLabels)

discriminative_multinomial_nb_classifier = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier.fit(text_vectorized, finalLabels)



#nb_classifier_desc = GaussianNB()
#nb_classifier_desc.fit(description_vectorized.toarray(), finalLabels)

multinomial_nb_classifier_desc = MultinomialNB()
multinomial_nb_classifier_desc.fit(description_vectorized, finalLabels)

complement_nb_classifier_desc  = ComplementNB()
complement_nb_classifier_desc.fit(description_vectorized, finalLabels)

discriminative_multinomial_nb_classifier_desc = MultinomialNB(alpha=1e-3)
discriminative_multinomial_nb_classifier_desc.fit(description_vectorized, finalLabels)
# Train the classifier on the training data

# Step 3: Scoring - Training Dataset
multinomial_nb_training_scores = multinomial_nb_classifier.predict_proba(text_vectorized)[:, 1]
complement_nb_training_scores = complement_nb_classifier.predict_proba(text_vectorized)[:, 1]
discriminative_multinomial_nb_training_scores = discriminative_multinomial_nb_classifier.predict_proba(text_vectorized)[:, 1]
nb_classifier_training_scores = nb_classifier.predict_proba(text_vectorized.toarray())[:, 1]

multinomial_nb_training_scores_desc = multinomial_nb_classifier_desc.predict_proba(description_vectorized)[:, 1]
complement_nb_training_scores_desc = complement_nb_classifier_desc.predict_proba(description_vectorized)[:, 1]
discriminative_multinomial_nb_training_scores_desc = discriminative_multinomial_nb_classifier_desc.predict_proba(description_vectorized)[:, 1]
#nb_classifier_training_scores_desc = nb_classifier_desc.predict_proba(description_vectorized.toarray())[:, 1]


df_multinomial_nb = pd.DataFrame(multinomial_nb_training_scores)
df_complement_nb = pd.DataFrame(complement_nb_training_scores)
df_discriminative_multinomial_nb = pd.DataFrame(discriminative_multinomial_nb_training_scores)
df_nb_classifier = pd.DataFrame(nb_classifier_training_scores)

df_multinomial_nb_desc = pd.DataFrame(multinomial_nb_training_scores_desc)
df_complement_nb_desc = pd.DataFrame(complement_nb_training_scores_desc)
df_discriminative_multinomial_nb_desc = pd.DataFrame(discriminative_multinomial_nb_training_scores_desc)
#df_nb_classifier_desc = pd.DataFrame(nb_classifier_training_scores_desc)
merged_df = pd.concat([df_multinomial_nb, df_complement_nb, df_discriminative_multinomial_nb, df_nb_classifier, df_multinomial_nb_desc, df_complement_nb_desc, df_discriminative_multinomial_nb_desc], axis=1)

X_train, X_test, y_train, y_test = train_test_split(merged_df, finalLabels, test_size=0.3, random_state=42)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
svc_model = SVC(kernel='linear')
rf_classifier.fit(X_train, y_train)
y_pred = rf_classifier.predict(X_test)

# Evaluate accuracy using binary predictions
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='micro')
recall = recall_score(y_test, y_pred, average='micro')
f1 = f1_score(y_test, y_pred, average='micro')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:\n", report)

