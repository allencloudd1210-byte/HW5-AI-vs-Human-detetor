from transformers import pipeline

print("Loading model...")
try:
    pipe = pipeline("text-classification", model="openai-community/roberta-base-openai-detector")
    print("Model loaded successfully.")
    
    text = "This is a test sentence written by a human."
    result = pipe(text)
    print(f"Prediction result: {result}")
    
    text_ai = "As an AI language model, I can generate text."
    result_ai = pipe(text_ai)
    print(f"AI Prediction result: {result_ai}")
    
except Exception as e:
    print(f"Error: {e}")
