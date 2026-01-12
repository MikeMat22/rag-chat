# Document Q&A - RAG Application

A retrieval-augmented generation (RAG) application that allows you to upload documents and ask questions about their content. Built with LangChain, Pinecone, and Groq.

## How it works

1. Upload PDF, TXT, or Markdown files
2. Documents are split into chunks (1000 characters with 200 overlap)
3. Each chunk is converted to a vector embedding using HuggingFace's MiniLM model
4. Embeddings are stored in Pinecone vector database
5. When you ask a question, the system finds the most relevant chunks using semantic search
6. Retrieved chunks are sent as context to Llama 3.3 (via Groq) which generates an answer

## Tech stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangChain |
| Vector Database | Pinecone (serverless) |
| LLM | Llama 3.3 70B via Groq API |
| Embeddings | HuggingFace all-MiniLM-L6-v2 |
| Frontend | Streamlit |

## Local setup

Clone the repository:
```bash
git clone https://github.com/MikeMat22/rag-chat.git
cd rag-chat
```

Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Create `.env` file with your API keys:
```
GROQ_API_KEY=your_groq_api_key
PINECONE_API_KEY=your_pinecone_api_key
```

Run the application:
```bash
streamlit run app.py
```

## API Keys

Both services offer free tiers:

- Groq: https://console.groq.com (free, fast inference)
- Pinecone: https://www.pinecone.io (free tier includes 100k vectors)

## Pinecone index setup

Create an index in Pinecone with the following settings:
- Name: `rag-chat`
- Dimensions: `384`
- Metric: `cosine`
- Type: Serverless (AWS, us-east-1)

## Project structure

```
rag-chat/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── .gitignore
└── README.md
```

## Configuration

Key parameters can be adjusted in `app.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| EMBEDDING_MODEL | all-MiniLM-L6-v2 | HuggingFace model for embeddings |
| LLM_MODEL | llama-3.3-70b-versatile | Groq model for generation |
| chunk_size | 1000 | Characters per chunk |
| chunk_overlap | 200 | Overlap between chunks |
| k | 4 | Number of chunks retrieved per query |

## Deployment

The app can be deployed for free on Streamlit Community Cloud:

1. Push the code to GitHub
2. Go to share.streamlit.io
3. Connect your GitHub repository
4. Add secrets (GROQ_API_KEY, PINECONE_API_KEY) in the Streamlit dashboard
5. Deploy

## License

MIT