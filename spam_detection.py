import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn.metrics import classification_report, accuracy_score

# Create df from spam emails csv
df = pd.read_csv('emails.csv')

# Vectorize text
text_vec = CountVectorizer().fit_transform(df['text'])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(text_vec,
                                                    df['spam'],
                                                    test_size=0.45,
                                                    random_state=42,
                                                    shuffle=True)

# Gradient Boosting Classifier
classifier = ensemble.GradientBoostingClassifier(n_estimators=100,
                                                 learning_rate=0.5,
                                                 max_depth=6)

# Train classifier
classifier.fit(X_train, y_train)

# Generate predictions
predictions = classifier.predict(X_test)

# Print classification report
print(classification_report(y_test, predictions))
