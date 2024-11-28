
# Medical QA Server

A simple Flask-based server that provides answers to medical questions using Sentence-BERT for semantic search. This project includes functionality to precompute and save question embeddings for faster response times.

## Features

- Precomputes and saves embeddings for predefined questions.
- Flask API endpoint to answer questions.
- Uses Sentence-BERT (`all-MiniLM-L6-v2`) for efficient semantic similarity.

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

4. **Precompute and Save Embeddings**
   Run the `save_embd.py` script to precompute and save the question embeddings:
   ```bash
   python save_embd.py
   ```
   This will create a `question_embeddings.npy` file in the root directory.

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

## File Structure

```
.
├── main.py              # Flask server with question-answering logic
├── save_embd.py         # Script to precompute and save question embeddings
├── medical_qa_dataset.csv # Dataset of predefined questions and answers
├── question_embeddings.npy # Precomputed question embeddings (created after running save_embd.py)
├── requirements.txt     # List of Python dependencies
├── README.md            # Project documentation
└── .gitignore           # Files to ignore in version control
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

This project is open-source
