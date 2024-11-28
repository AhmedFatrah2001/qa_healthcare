import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# Load dataset
df = pd.read_csv("medical_qa_dataset.csv")
questions = df['question'].tolist()

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Precompute embeddings
print("Precomputing question embeddings...")
question_embeddings = model.encode(questions, normalize_embeddings=True)

# Save embeddings to a file
np.save("question_embeddings.npy", question_embeddings)
print("Embeddings saved to 'question_embeddings.npy'")
