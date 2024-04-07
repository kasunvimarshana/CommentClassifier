import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

class CommentClassifier:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_features=1000)
        self.rating_model = LogisticRegression()
        self.toxicity_model = LogisticRegression()

    def train(self, X_train, y_rating_train, y_toxicity_train):
        X_train_tfidf = self.tfidf_vectorizer.fit_transform(X_train)
        self.rating_model.fit(X_train_tfidf, y_rating_train)
        self.toxicity_model.fit(X_train_tfidf, y_toxicity_train)

    def predict(self, X_test):
        X_test_tfidf = self.tfidf_vectorizer.transform(X_test)
        rating_prediction = self.rating_model.predict(X_test_tfidf)
        toxicity_prediction = self.toxicity_model.predict(X_test_tfidf)
        return rating_prediction, toxicity_prediction

    def save(self, file_path):
        joblib.dump((self.tfidf_vectorizer, self.rating_model, self.toxicity_model), file_path)

    @classmethod
    def load(cls, file_path):
        tfidf_vectorizer, rating_model, toxicity_model = joblib.load(file_path)
        classifier = cls()
        classifier.tfidf_vectorizer = tfidf_vectorizer
        classifier.rating_model = rating_model
        classifier.toxicity_model = toxicity_model
        return classifier

def main():
    # Load data and split into train/test sets
    file_path = 'combined_comments.csv'  # Update with your combined dataset
    data = pd.read_csv(file_path)
    X = data['comment']
    y_rating = data['rating']
    y_toxicity = data['toxicity']
    X_train, X_test, y_rating_train, y_rating_test, y_toxicity_train, y_toxicity_test = train_test_split(X, y_rating, y_toxicity, test_size=0.2, random_state=42)

    # Instantiate and train classifier
    classifier = CommentClassifier()
    classifier.train(X_train, y_rating_train, y_toxicity_train)

    # Save trained classifier
    classifier.save('trained_comment_classifier.joblib')
    print("Trained classifier saved.")

    # Load saved classifier
    loaded_classifier = CommentClassifier.load('trained_comment_classifier.joblib')
    print("Trained classifier loaded.")

    # Predict ratings using loaded classifier
    rating_prediction, toxicity_prediction = loaded_classifier.predict(X_test)

    # Evaluate accuracy
    rating_accuracy = (rating_prediction == y_rating_test).mean()
    print("Rating Accuracy:", rating_accuracy)

    # Evaluate toxicity accuracy
    toxicity_accuracy = (toxicity_prediction == y_toxicity_test).mean()
    print("Toxicity Accuracy:", toxicity_accuracy)

if __name__ == "__main__":
    main()
