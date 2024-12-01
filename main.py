import numpy as np
import pandas as pd
from sentence_transformers import models, SentenceTransformer
from functools import lru_cache
from flask import Flask, request, jsonify
from sklearn.preprocessing import normalize

# Load dataset
df = pd.read_csv("medical_qa_dataset.csv")
questions = df['question'].tolist()
answers = df['answer'].tolist()

# Load precomputed embeddings
print("Loading precomputed question embeddings...")
question_embeddings = np.load("combined_question_embeddings.npy")

# Load models for input encoding
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load the BioBERT transformer model
biobert_transformer = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=256)

# Add pooling to create sentence embeddings
biobert_pooling = models.Pooling(biobert_transformer.get_word_embedding_dimension())

# Create a SentenceTransformer model with the components
biobert_model = SentenceTransformer(modules=[biobert_transformer, biobert_pooling])

# Cache for input embeddings
@lru_cache(maxsize=100)
def encode_input_question(input_question):
    # Generate embeddings for the input question
    minilm_embedding = minilm_model.encode(input_question, normalize_embeddings=True)
    biobert_embedding = biobert_model.encode(input_question, normalize_embeddings=True)
    # Combine embeddings (concatenate)
    combined_embedding = np.concatenate([minilm_embedding, biobert_embedding])
    return combined_embedding

# Function to find the most relevant answer

def get_answer(input_question):
    # Compute the input embedding
    input_embedding = encode_input_question(input_question)

    # Normalize embeddings
    normalized_question_embeddings = normalize(question_embeddings, axis=1)
    normalized_input_embedding = normalize(input_embedding.reshape(1, -1), axis=1)

    # Compute cosine similarities
    similarities = np.dot(normalized_question_embeddings, normalized_input_embedding.T).flatten()

    # Find the most similar question
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
