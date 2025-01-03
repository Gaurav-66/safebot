from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import random
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Load the knowledge base
def load_knowledge_base(filepath):
    """Load the knowledge base from a JSON file."""
    if os.path.exists(filepath):
        with open(filepath, 'r') as file:
            return json.load(file)
    else:
        logging.error(f"Error: The file {filepath} does not exist.")
        return []

# Find the best match using TF-IDF
def find_best_match(user_input, knowledge_base):
    """Find the most relevant response using TF-IDF."""
    corpus = [entry["question"] for entry in knowledge_base]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(corpus)

    # Transform the user input
    user_vector = vectorizer.transform([user_input])

    # Calculate cosine similarity
    similarities = cosine_similarity(user_vector, tfidf_matrix)
    best_match_index = similarities.argmax()
    highest_similarity = similarities[0, best_match_index]

    return knowledge_base[best_match_index] if highest_similarity > 0.1 else None

# Greeting response
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

# Exit response
def exit_response():
    """Generate a random exit response."""
    responses = [
        "Goodbye! Have a great day!",
        "Bye! Take care!",
        "See you later! Goodbye!",
        "It was nice talking to you. Bye!"
    ]
    return random.choice(responses)

# Fallback response when no match is found
def fallback_response():
    """Generate a fallback response when no match is found."""
    responses = [
        "Sorry, I didn't quite understand that.",
        "I'm not sure how to respond to that.",
        "Can you please rephrase your question?",
        "I couldn't find an answer for that, please ask something else."
    ]
    return random.choice(responses)

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.get_json()
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "Please provide a valid input."})

    # Check for greetings or exit commands
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
    
    # Find the best match
    best_match = find_best_match(user_input, knowledge_base)
    if best_match:
        return jsonify({"response": best_match["answer"]})
    else:
        return jsonify({"response": fallback_response()})

if __name__ == '__main__':
    app.run()
