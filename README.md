# Rag-Based-ChatBot

rag_chatbot/
├── app.py                    # Flask app for local testing
├── ingest.py                 # Document loading and FAISS indexing
├── rag_chain.py              # LangChain RAG pipeline setup
├── lambda_handler.py         # AWS Lambda entry point
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and instructions
├── .env                      # Environment variables (optional)
├── docs/
│   └── knowledge.txt         # Source documents for retrieval
└── faiss_index/              # Saved FAISS index (auto-generated)
