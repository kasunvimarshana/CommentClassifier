from RatingClassifier import RatingClassifier

def main():
    # Load saved classifier
    loaded_classifier = RatingClassifier.load('trained_classifier.joblib')
    print("Trained classifier loaded.")

    # User input example
    user_comments = [
        "The product is excellent, I love it!",
        "Not satisfied with the service, very poor experience.",
        "Great app, very user-friendly interface."
    ]

    # Predict ratings for user comments
    predicted_ratings = loaded_classifier.predict(user_comments)

    # Display predicted ratings
    for comment, rating in zip(user_comments, predicted_ratings):
        print(f"Comment: {comment} --> Predicted Rating: {rating}")

if __name__ == "__main__":
    main()
