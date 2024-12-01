
# Medical QA Server

A Flask-based server that provides answers to medical questions using combined embeddings from two Sentence-BERT models: `all-MiniLM-L6-v2` (general-purpose) and `BioBERT` (domain-specific for biomedical texts). This approach enhances the accuracy of semantic search by leveraging complementary strengths of the models.

## Features

- Combines embeddings from `all-MiniLM-L6-v2` and `BioBERT` for richer semantic representations.
- Precomputes and saves combined embeddings for predefined questions to ensure fast response times.
- Flask API endpoint to answer questions dynamically.
- Ideal for medical-related query answering systems.

---

## Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Steps

1. **Clone the Repository**
   ```bash
   git clone https://github.com/AhmedFatrah2001/qa_healthcare
   cd qa_healthcare
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare the Dataset**
   Place your `medical_qa_dataset.csv` file in the root directory. Ensure it contains the following columns:
   - `question`: Predefined medical questions.
   - `answer`: Corresponding answers to the questions.

4. **Precompute and Save Combined Embeddings**
   Run the `save_embd.py` script to precompute and save the combined embeddings:
   ```bash
   python save_embd.py
   ```
   This will create a `combined_question_embeddings.npy` file in the root directory.

5. **Run the Server**
   Start the Flask server by running `main.py`:
   ```bash
   python main.py
   ```
   The server will be available at `http://127.0.0.1:5000`.

---

## API Usage

### Endpoint: `/get_answer`
- **Method**: `POST`
- **Payload**:
  ```json
  {
      "question": "Your medical question here"
  }
  ```
- **Response**:
  ```json
  {
      "question": "Your question",
      "answer": "Relevant answer from the dataset",
      "similarity": 0.9235
  }
  ```

### Example cURL Command
```bash
curl -X POST http://127.0.0.1:5000/get_answer -H "Content-Type: application/json" -d '{"question": "What are the symptoms of diabetes?"}'
```

---

## How It Works

### Combined Embeddings
1. **Precomputation**:
   - During precomputation, embeddings are generated using two models:
     - `all-MiniLM-L6-v2`: A general-purpose Sentence-BERT model.
     - `BioBERT`: A biomedical domain-specific model.
   - The embeddings from both models are **concatenated** to form a richer representation for each question.

2. **Similarity Search**:
   - During inference, the input question is encoded using both models, and the embeddings are concatenated.
   - Similarity is calculated using the dot product between the input embedding and the precomputed combined embeddings.

### Why Combine Models?
- `all-MiniLM-L6-v2` captures general linguistic nuances.
- `BioBERT` specializes in biomedical vocabulary, making it more effective for domain-specific queries.
- Combining them ensures that both general and domain-specific information are considered.

---

## File Structure

```
.
├── main.py                      # Flask server with question-answering logic
├── save_embd.py                 # Script to precompute and save combined question embeddings
├── medical_qa_dataset.csv       # Dataset of predefined questions and answers
├── combined_question_embeddings.npy # Precomputed combined embeddings (created after running save_embd.py)
├── requirements.txt             # List of Python dependencies
├── README.md                    # Project documentation
└── .gitignore                   # Files to ignore in version control
```

---

## Requirements

The required Python packages are listed in `requirements.txt`. Install them using:
```bash
pip install -r requirements.txt
```

### `requirements.txt`
```
flask
numpy
pandas
sentence-transformers
```

---

## License

This project is open-source and free to use.
