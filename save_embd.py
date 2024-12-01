import numpy as np
from sentence_transformers import SentenceTransformer,models
import pandas as pd

# Load dataset
df = pd.read_csv("medical_qa_dataset.csv")
questions = df['question'].tolist()

# Load models
minilm_model = SentenceTransformer('all-MiniLM-L6-v2')
# Load the BioBERT transformer model
biobert_transformer = models.Transformer('dmis-lab/biobert-v1.1', max_seq_length=256)

# Add pooling to create sentence embeddings
biobert_pooling = models.Pooling(biobert_transformer.get_word_embedding_dimension())

# Create a SentenceTransformer model with the components
biobert_model = SentenceTransformer(modules=[biobert_transformer, biobert_pooling])

# Precompute embeddings
print("Precomputing question embeddings...")
minilm_embeddings = minilm_model.encode(questions, normalize_embeddings=True)
biobert_embeddings = biobert_model.encode(questions, normalize_embeddings=True)

# Combine embeddings (concatenate)
combined_embeddings = np.concatenate((minilm_embeddings, biobert_embeddings), axis=1)

# Save embeddings to a file
np.save("combined_question_embeddings.npy", combined_embeddings)
print("Combined embeddings saved to 'combined_question_embeddings.npy'")
