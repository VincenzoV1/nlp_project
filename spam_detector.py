import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report

# Load the dataset
df = pd.read_csv('emails.csv')

# Choose Vectorizer: Set to 'tfidf' or 'count'
vectorizer_choice = 'tfidf'  # change to 'count' if needed

if vectorizer_choice == 'tfidf':
    vectorizer = TfidfVectorizer()
else:
    vectorizer = CountVectorizer()

# Vectorize the text column
text_vec = vectorizer.fit_transform(df['text'])

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    text_vec, df['spam'], test_size=0.45, random_state=42, shuffle=True
)

# Gradient Boosting Classifier
classifier = ensemble.GradientBoostingClassifier(
    n_estimators=100, learning_rate=0.5, max_depth=6
)

# Train and Predict
classifier.fit(X_train, y_train)
predictions = classifier.predict(X_test)

# Print results
print("Using Vectorizer:", vectorizer_choice)
print(classification_report(y_test, predictions))
