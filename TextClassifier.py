import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

class TextClassifier:
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
    file_path = 'toxic_comments.csv'  # Update with your toxic comment dataset
    data = pd.read_csv(file_path)
    X = data['comment']
    y = data['toxicity']  # Assuming 'toxicity' is the label indicating whether a comment is toxic or not
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Instantiate and train classifier
    classifier = TextClassifier()
    classifier.train(X_train, y_train)

    # Save trained classifier
    classifier.save('trained_text_classifier.joblib')
    print("Trained classifier saved.")

    # Load saved classifier
    loaded_classifier = TextClassifier.load('trained_text_classifier.joblib')
    print("Trained classifier loaded.")

    # Predict toxicity using loaded classifier
    user_comments = [
        "The product is excellent, I love it!",
        "Not satisfied with the service, very poor experience.",
        "Great app, very user-friendly interface."
    ]
    predicted_toxicity = loaded_classifier.predict(user_comments)

    # Display predicted toxicity
    for comment, toxicity in zip(user_comments, predicted_toxicity):
        if toxicity == 1:
            print(f"Comment: '{comment}' --> Toxic")
        else:
            print(f"Comment: '{comment}' --> Non-Toxic")

if __name__ == "__main__":
    main()
