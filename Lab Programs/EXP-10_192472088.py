from transformers import pipeline

classifier = pipeline("sentiment-analysis")

text = "The movie was fantastic."

print(classifier(text))