from transformers import pipeline
classifier = pipeline("sentiment-analysis")
feedback = input("Enter your feedback: ")
result = classifier(feedback)[0]
print("\nFeedback:", feedback)
print("Sentiment:", result["label"])
print("Score:", round(result["score"], 2))