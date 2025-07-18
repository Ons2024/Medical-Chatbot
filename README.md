# Medical-Chatbot

Medical-Chatbot is an AI-powered web application designed to assist users with medical queries by leveraging advanced language models and a searchable medical knowledge base.

## Features

- **Interactive Chat Interface:** Users can ask medical questions and receive instant responses.
- **PDF Knowledge Extraction:** Medical information is extracted from [data/Medical_book.pdf](data/Medical_book.pdf) and indexed for retrieval.
- **Semantic Search:** Uses sentence-transformer embeddings and Pinecone vector database for accurate information retrieval.
- **Modern UI:** Built with Bootstrap and custom CSS for a responsive, user-friendly experience.

## How It Works

1. **Data Extraction:** The system loads and processes medical PDFs using functions from [`src/helper.py`](src/helper.py).
2. **Embedding & Indexing:** Text chunks are embedded using HuggingFace models and stored in Pinecone via [`store_index.py`](store_index.py).
3. **Chatbot Engine:** User queries are matched against the indexed data, and relevant answers are generated using OpenAI or similar models in [`app.py`](app.py).
4. **Frontend:** The chat interface is rendered with Flask templates ([`templates/index.html`](templates/index.html)) and styled with [`static/style.css`](static/style.css).

## Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt