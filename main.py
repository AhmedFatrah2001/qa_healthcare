import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from functools import lru_cache
from flask import Flask, request, jsonify

# Load dataset
df = pd.read_csv("medical_qa_dataset.csv")
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Load precomputed embeddings
print("Loading precomputed question embeddings...")
question_embeddings = np.load("question_embeddings.npy")

# Load model for input encoding
model = SentenceTransformer('all-MiniLM-L6-v2')

# Cache for input embeddings
@lru_cache(maxsize=100)
def encode_input_question(input_question):
    return model.encode(input_question, normalize_embeddings=True)

# Function to find the most relevant answer
def get_answer(input_question):
    input_embedding = encode_input_question(input_question)
    similarities = np.dot(question_embeddings, input_embedding)
    most_similar_index = np.argmax(similarities)
    similarity_score = similarities[most_similar_index]
    return answers[most_similar_index], similarity_score

# Initialize Flask app
app = Flask(__name__)

@app.route('/get_answer', methods=['POST'])
def answer_question():
    data = request.json
    user_question = data.get('question')
    
    if not user_question:
        return jsonify({"error": "Question is required"}), 400

    try:
        answer, similarity = get_answer(user_question)
        return jsonify({
            "question": user_question,
            "answer": answer,
            "similarity": float(similarity)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the server
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
