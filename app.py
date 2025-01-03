from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import json
import os
import random
import logging
import re

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Flask app setup
app = Flask(__name__)
CORS(app)

# Custom preprocessing function to clean and tokenize the text
def custom_preprocess(text):
    """Custom preprocessing: lowercase, remove punctuation, and strip extra spaces."""
    # Convert to lowercase
    text = text.lower()
    # Remove all non-alphanumeric characters (e.g., punctuation)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Load the knowledge base from a JSON file
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
    try:
        # Create a corpus of questions from the knowledge base
        corpus = [entry["question"] for entry in knowledge_base]
        vectorizer = TfidfVectorizer(preprocessor=custom_preprocess)  # Use custom preprocessing
        tfidf_matrix = vectorizer.fit_transform(corpus)

        # Transform the user input
        user_vector = vectorizer.transform([user_input])

        # Calculate cosine similarity
        similarities = cosine_similarity(user_vector, tfidf_matrix)
        best_match_index = similarities.argmax()
        highest_similarity = similarities[0, best_match_index]

        # Return the best match if similarity is above threshold, else fallback
        if highest_similarity > 0.05:
            return knowledge_base[best_match_index]
        else:
            # Default fallback message
            return {"question": "fallback", "answer": fallback_response()}
    except Exception as e:
        logging.error(f"Error in find_best_match: {str(e)}")
        return {"question": "error", "answer": "Something went wrong. Please try again."}

# Fallback response when no match is found
def fallback_response():
    """Generate a fallback response when no match is found."""
    responses = [
        "I'm sorry, I couldn't find an answer to your question.",
        "Can you please rephrase your question?",
        "I'm still learning. Could you ask that in a different way?",
        "I don't know the answer to that, but I'm here to help with other questions!"
    ]
    return random.choice(responses)

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

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests."""
    data = request.get_json()
    user_input = data.get("user_input", "").strip()

    if not user_input:
        return jsonify({"response": "I didn't catch that. Could you please rephrase?"})

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
    return jsonify({"response": best_match["answer"]})

if __name__ == '__main__':
    app.run()
