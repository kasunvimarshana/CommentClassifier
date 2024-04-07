from CommentClassifier import CommentClassifier
import matplotlib.pyplot as plt

def main():
    # Load saved classifier
    loaded_classifier = CommentClassifier.load('trained_comment_classifier.joblib')
    print("Trained classifier loaded.")

    # User input example
    user_comments = [
        "The product is excellent, I love it!",
        "Not satisfied with the service, very poor experience.",
        "Great app, very user-friendly interface.",
        "Great product",
        "not usable",
    ]

    # Predict ratings for user comments
    rating_prediction, toxicity_prediction = loaded_classifier.predict(user_comments)

    # Display predicted rating and toxicity
    for comment, rating, toxicity in zip(user_comments, rating_prediction, toxicity_prediction):
        if toxicity == 1:
            print(f"Comment: '{comment}' --> Toxic, Rating: {rating}")
        else:
            print(f"Comment: '{comment}' --> Non-Toxic, Rating: {rating}")

    # Plot predictions
    plt.figure(figsize=(10, 5))

    # Draw a graph
    '''
    # plt.barh(range(len(user_comments)), rating_prediction, color='skyblue', label='Rating')
    # plt.barh(range(len(user_comments)), toxicity_prediction, color='lightcoral', label='Toxicity')
    # plt.yticks(range(len(user_comments)), [f'Comment {i+1}' for i in range(len(user_comments))])
    # plt.xlabel('Rating/Toxicity Prediction')
    # plt.ylabel('Comments')
    # plt.title('Predictions for User Comments')
    # plt.legend()
    # plt.show()
    '''

    # Plot predicted ratings
    plt.subplot(1, 2, 1)
    plt.bar(range(len(user_comments)), rating_prediction, color='skyblue')
    plt.title('Predicted Ratings')
    plt.xlabel('Comment Index')
    plt.ylabel('Rating')
    plt.xticks(range(len(user_comments)), [f'Comment {i+1}' for i in range(len(user_comments))])

    # Plot predicted toxicities
    plt.subplot(1, 2, 2)
    plt.bar(range(len(user_comments)), toxicity_prediction, color='lightcoral')
    plt.title('Predicted Toxicities')
    plt.xlabel('Comment Index')
    plt.ylabel('Toxicity (0: Non-Toxic, 1: Toxic)')
    plt.xticks(range(len(user_comments)), [f'Comment {i+1}' for i in range(len(user_comments))])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
