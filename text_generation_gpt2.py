# text_generation_gpt2.py

# This script uses the GPT-2 language model to generate text from a given prompt

from transformers import pipeline  # Import Hugging Face's pipeline tool for easy model access

# Load the text-generation pipeline using GPT-2 (a Transformer-based language model)
generator = pipeline("text-generation", model="gpt2")

# Define the input prompt (starting point for text generation)
prompt = "Artificial intelligence will revolutionize"

# Generate a continuation of the prompt with a maximum length of 50 tokens
# Only one sequence (result) is generated
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text to the console
print("Generated Text:")
print(result[0]['generated_text'])  # Access and display the generated sequence