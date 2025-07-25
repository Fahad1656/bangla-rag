# 🌟 Bangla RAG System

A **Bangla Retrieval-Augmented Generation (RAG)** system designed for **deep semantic question answering** in Bangla and English. This project uses OCR, chunking, embeddings, and a FastAPI backend to provide context-aware answers to Bangla questions.

---

## 🧱 Project Structure

```
bangla-rag/
│
├── api_routes/
│   ├── interference.py
│   └── inference.py
│
├── data_preprocessing/
│   ├── extract.py
│   └── preprocess_data.py
│
├── rag_pipeline/
│   └── rag.py
│
├── models/
│   └── model.py
│
├── storage/
│   └── store.py
│
├── main.py
├── requirements.txt
├── .env.example
├── README.md

```

---

## 🧵 Step 1: Environment Setup

### ✅ Install Python Dependencies

Make sure you have **Python 3.8+** installed.

```bash
pip install -r requirements.txt
```

---

## 🤖 Step 2: Install Ollama & Load Model

1. Download and install Ollama: [https://ollama.ai](https://ollama.ai)
2. Pull the required model:

```bash
ollama pull aya-expanse:8b
```

---

## 🚀 Step 3: Run the API Server

Start the FastAPI development server:

```bash
uvicorn main:app --reload --port 8000
```

Visit the server at: [http://127.0.0.1:8000](http://127.0.0.1:8000)

### ✅ API Endpoint:

**`POST /api/ask`**

---

## 📄 API Usage Example

### Headers:

```http
Content-Type: application/json
```

### Request Body (Bangla Question):

```json
{
  "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
}
```

### Response:

```json
{
  "answer": "মামাকে",
  "language prompted": "বাংলা",
  "confidence": 1.0
}
```

### Request Body (Another Bangla Question):

```json
{
  "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
}
```

### Response:

```json
{
  "answer": "১৫ বছর",
  "language prompted": "বাংলা",
  "confidence": 1.0
}
```

### Request Body (English Question):

```json
{
  "question": "What was the age of Kallyani at the time of her marriage?"
}
```

### Response:

```json
{
  "answer": "১৫ বছর",
  "language prompted": "English",
  "confidence": 1.0
}
```

---

## 📦 Dependencies

```txt
langchain==0.1.0
langchain-community==0.0.10
sentence-transformers==2.2.2
langchain-core==0.1.23
langchain-text-splitters==0.3.8
transformers==4.33.3
torch==2.7.1
torchvision==0.22.1
torchaudio==2.2.0
faiss-cpu==1.7.4
llama-index-core==0.12.52
llama-index-embeddings-huggingface==0.5.5
llama-index-instrumentation==0.3.0
llama-index-llms-ollama==0.6.2
llama-index-workflows==1.1.0
bangla-pdf-ocr==0.1.1
dataclasses-json==0.6.7
langdetect==1.0.9
regex==2024.11.6
nltk==3.9.1
ftfy==6.1.3
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0
requests==2.32.4
ollama==0.1.7
accelerate==0.21.1
scikit-learn==1.6.1
tqdm==4.67.1
loguru==0.7.0
python-dotenv==1.0.0
jupyter==1.0.0
ipykernel==6.26.0
onnx==1.17.0
huggingface_hub==0.21.4
numpy==1.23.5
```

---

## 🧠 Technical Design Decisions

### ❓ What method or library did you use to extract the text, and why? Did you face any formatting challenges with the PDF content?

I used the **`bangla_pdf_ocr`** library for extracting text from scanned PDF documents. I chose it because it provides strong OCR capabilities specifically tuned for Bangla script. I faced several challenges such as inconsistent line breaks, mixed punctuation, and noisy formatting, which I addressed through normalization and preprocessing steps.

### ❓ What chunking strategy did you choose (e.g. paragraph-based, sentence-based, character limit)? Why do you think it works well for semantic retrieval?

I use **sentence-based dynamic chunking** with a max token limit (default 512). I merge sentences without exceeding the limit, ensuring semantic coherence. This works well because it preserves full ideas and avoids cutting context mid-thought, which improves retrieval accuracy.

### ❓ What embedding model did you use? Why did you choose it? How does it capture the meaning of the text?

I use a **multilingual SentenceTransformer** model from HuggingFace/Ollama. I chose it for its proven performance on semantic tasks across languages. It encodes text into dense vector representations that preserve meaning, not just syntax, which is crucial for Bangla-English mixed input.

### ❓ How are you comparing the query with your stored chunks? Why did you choose this similarity method and storage setup?

I compare query and chunk embeddings using **cosine similarity**, stored and indexed via **FAISS**. Cosine similarity measures angle rather than magnitude, making it ideal for semantic search. FAISS gives fast, scalable indexing with IVF or flat modes.

### ❓ How do you ensure that the question and the document chunks are compared meaningfully? What would happen if the query is vague or missing context?

I prefix input with semantic tags (e.g., `query:`, `passage:`), perform language detection, and ensure Bangla-aware sentence segmentation. If a query lacks context, the system still returns best matches but with lower confidence. To handle vagueness, I plan to add query clarification logic.

### ❓ Do the results seem relevant? If not, what might improve them (e.g. better chunking, better embedding model, larger document)?

Yes, for most clear Bangla questions, results are accurate. But I can improve further with:

* Overlapping chunks (sliding window)
* Stronger, larger multilingual embedding models
* Re-ranking techniques
* Enriching the dataset
* Self-refinement feedback loops

---

## ✨ Author

**FAHAD MOHAMMAD REJWANUL ISLAM**.
