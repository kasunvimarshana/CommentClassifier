import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

class RatingClassifier:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression()

    def train(self, X_train, y_train):
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)

    def predict(self, X_test):
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        return self.model.predict(X_test_tfidf)

    def save(self, file_path):
        joblib.dump((self.tfidf_vectorizer, self.model), file_path)

    @classmethod
    def load(cls, file_path):
        tfidf_vectorizer, model = joblib.load(file_path)
        classifier = cls()
        classifier.tfidf_vectorizer = tfidf_vectorizer
        classifier.model = model
        return classifier

def main():
    # Load data and split into train/test sets
    file_path = 'user_comments_ratings.csv'
    data = pd.read_csv(file_path)
    X = data['comment']
    y = data['rating']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train classifier
    classifier = RatingClassifier()
    classifier.train(X_train, y_train)

    # Save trained classifier
    classifier.save('trained_classifier.joblib')
    print("Trained classifier saved.")

    # Load saved classifier
    loaded_classifier = RatingClassifier.load('trained_classifier.joblib')
    print("Trained classifier loaded.")

    # Predict ratings using loaded classifier
    y_pred = loaded_classifier.predict(X_test)

    # Evaluate accuracy
    accuracy = (y_pred == y_test).mean()
    print("Accuracy:", accuracy)

if __name__ == "__main__":
    main()
