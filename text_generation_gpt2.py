# text_generation_gpt2.py

# This script uses the GPT-2 language model to generate text given a prompt.
# GPT-2 is a Transformer-based model trained to predict the next word in a sequence.

from transformers import pipeline  # Hugging Face abstraction for running models

# Load the text-generation pipeline using GPT-2 as the default model
# This wraps tokenization, model loading, and generation into a simple interface
generator = pipeline("text-generation", model="gpt2")

# Provide a text prompt that GPT-2 will continue from
prompt = "Artificial intelligence will revolutionize"

# Generate up to 50 new tokens (words, subwords, punctuation) following the prompt
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print("Generated Text:")
print(result[0]['generated_text'])

# Notes:
# - GPT-2 is a generative pretrained transformer
# - It was trained on a massive dataset to model natural language
# - "max_length" sets the total number of tokens (prompt + generated)
# - "num_return_sequences" lets you generate multiple variations
