# sentiment_analysis_hf.py

# This script performs sentiment analysis using a Hugging Face Transformers pipeline.
# Sentiment analysis determines if text expresses a positive, negative, or neutral emotion.

from transformers import pipeline  # Hugging Face pipeline abstracts complex NLP tasks

# Load a sentiment-analysis pipeline using a default pretrained model (usually DistilBERT or BERT)
classifier = pipeline("sentiment-analysis")  # Loads a model fine-tuned for sentiment tasks

# Define a sample sentence to analyze
sentence = "I love working with machine learning and AI!"

# Run the classifier on the sentence
result = classifier(sentence)

# Print the result, which includes the predicted label and confidence score
print("Sentiment Analysis Result:")
print(result)

# Example output:
# [{'label': 'POSITIVE', 'score': 0.9998}]

# Notes:
# - The model is pretrained and fine-tuned on sentiment classification datasets (like SST-2)
# - Transformers are models based on attention mechanisms and are state-of-the-art for many NLP tasks
# - The pipeline makes using such models easy without needing manual tokenization or decoding
