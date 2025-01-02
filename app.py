from flask import Flask, request, jsonify
import random
import json
import os
import logging
import math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from flask_cors import CORS  # Import the CORS module

# Download necessary NLTK data files (only once)
import nltk
nltk.download('punkt')

nltk.download('stopwords')
nltk.download('wordnet')

# Setup logging for better monitoring and debugging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Preprocessing utilities
def preprocess(text):
    """Tokenize, remove stopwords, and lemmatize the input text."""
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    tokens = word_tokenize(text.lower())
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word.isalnum() and word not in stop_words]
    return tokens

# Load knowledge base from the provided file
def load_knowledge_base(filepath):
    """Load the custom knowledge base from a JSON file."""
    if os.path.exists(filepath):
        try:
            with open(filepath, 'r') as file:
                knowledge_base = json.load(file)
                # Validate structure of knowledge base
                if not all('question' in entry and 'answer' in entry for entry in knowledge_base):
                    raise ValueError("Knowledge base entries must contain 'question' and 'answer' fields.")
                return knowledge_base
        except json.JSONDecodeError:
            logging.error("Error: The file is not a valid JSON.")
        except ValueError as ve:
            logging.error(f"Error: {ve}")
        return []
    else:
        logging.error(f"Error: The file {filepath} does not exist.")
        return []

# Vectorize text (convert tokens to a frequency vector)
def vectorize(tokens):
    """Convert tokens to a frequency vector."""
    return Counter(tokens)

# Compute cosine similarity between two vectors
def cosine_similarity(vec1, vec2):
    """Calculate cosine similarity between two frequency vectors."""
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([val**2 for val in vec1.values()])
    sum2 = sum([val**2 for val in vec2.values()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    return numerator / denominator if denominator else 0.0

# Find the best match from the knowledge base
def find_best_match(user_input, knowledge_base):
    """Find the most relevant response from the knowledge base."""
    user_tokens = preprocess(user_input)
    user_vector = vectorize(user_tokens)

    best_match = None
    highest_similarity = 0

    for entry in knowledge_base:
        question_tokens = preprocess(entry['question'])
        question_vector = vectorize(question_tokens)

        similarity = cosine_similarity(user_vector, question_vector)
        if similarity > highest_similarity:
            highest_similarity = similarity
            best_match = entry

    return best_match if highest_similarity > 0.1 else None

# Greeting function to return a random greeting response
def greeting_response():
    """Generate a random greeting response."""
    responses = [
        "Hello! How can I assist you today?",
        "Hi there! How's it going?",
        "Hey! How can I help you?",
        "Hi, how are you?",
        "Hello! How can I help you today?"
    ]
    return random.choice(responses)

# Exit function to return a goodbye response
def exit_response():
    """Generate a random exit response."""
    responses = [
        "Goodbye! Have a great day!",
        "Bye! Take care!",
        "See you later! Goodbye!",
        "It was nice talking to you. Bye!"
    ]
    return random.choice(responses)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.get_json()
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "Please provide a valid input."})

    # Check if the user input is a greeting or exit command
    greetings = ["hi", "hello", "hey", "how are you", "how's it going"]
    exits = ["exit", "bye", "goodbye"]

    if any(greeting in user_input.lower() for greeting in greetings):
        return jsonify({"response": greeting_response()})
    
    if any(exit_word in user_input.lower() for exit_word in exits):
        return jsonify({"response": exit_response()})
    
    # Load the knowledge base
    knowledge_base = load_knowledge_base('knowledge_base.json')
    
    if not knowledge_base:
        return jsonify({"response": "Sorry, the knowledge base is not available."})
    
    # Find the best match for the user input from the knowledge base
    best_match = find_best_match(user_input, knowledge_base)

    if best_match:
        return jsonify({"response": best_match['answer']})
    else:
        return jsonify({"response": "I'm sorry, I didn't understand that."})

if __name__ == '__main__':
    app.run()
