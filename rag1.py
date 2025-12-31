import os
import re
import pytesseract
import streamlit as st
from pdf2image import convert_from_path
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# =========================
# 1. LOAD ENV & HF CLIENT

load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")

if not HF_API_KEY:
    st.error(" Missing HF_API_KEY in .env file")
    st.stop()

client = InferenceClient(
    api_key=HF_API_KEY,
    base_url="https://router.huggingface.co/v1"
)


# =========================
# 2. OCR: SCANNED PDF → TEXT

def extract_text_from_scanned_pdf(pdf_path: str) -> str:
    """Convert each PDF page to image, run OCR, return full text."""

    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    pages = convert_from_path(
        pdf_path,
        poppler_path=r"C:\poppler\poppler-25.11.0\Library\bin"
    )

    full_text = ""
    for i, page in enumerate(pages, start=1):
        text = pytesseract.image_to_string(page)
        full_text += f"\n\n=== PAGE {i} ===\n{text}"

    full_text = re.sub(r"\n{3,}", "\n\n", full_text)
    return full_text


# =========================
# 3. VECTOR DB (CHROMA)

def create_vector_db(text: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_text(text)

    docs = [Document(page_content=c) for c in chunks]

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    db = Chroma.from_documents(documents=docs, embedding=embeddings)
    return db


# =========================
# 4. PROMPT BUILDER

def build_prompt(question: str, context: str, history_text: str) -> str:
    return f"""
You are an AI assistant designed to help the user understand the content of a document.

Use the conversation history to maintain context, but ALWAYS ground your answer in the document text.

--- CONVERSATION HISTORY ---
{history_text}

--- DOCUMENT CONTEXT ---
{context}

USER QUESTION:
{question}

RULES:
- Use ONLY the context from the document.
- If the answer is not found in the document, say:
  "The document does not provide enough information."
- Keep the answer short, factual, and neutral.
- Use bullet points when helpful.
- Do NOT hallucinate or invent details.

ANSWER:
"""


# =========================
# 5. QWEN CALL

def ask_qwen_llm(prompt: str) -> str:

    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-72B-Instruct",
        messages=[
            {"role": "system", "content": "You are a safe and accurate assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=500,
        temperature=0
    )

    return response.choices[0].message["content"].strip()


# =========================
# 6. ANSWER QUERY

def answer_query(question: str, history: list, db: Chroma) -> str:
    """RAG pipeline for chat."""

    # Retrieval
    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content for d in docs])

    # Build chat history
    history_text = ""
    for msg in history[-6:]:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        history_text += f"{role}: {msg['content']}\n"

    # Build full prompt
    prompt = build_prompt(question, context, history_text)

    # Ask LLM
    return ask_qwen_llm(prompt)


# =========================
# STREAMLIT UI

st.title(" AI Document Chatbot – RAG + Qwen")
st.write("Upload a document and chat with it using Retrieval-Augmented Generation (RAG).")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:

    # Save temporarily
    pdf_path = "uploaded.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.info("Extracting text from PDF (OCR)... ⏳")
    text = extract_text_from_scanned_pdf(pdf_path)

    st.success("Text extracted!")

    st.info("Building vector database... ⏳")
    db = create_vector_db(text)
    st.success("Vector DB ready! Ask your questions below.")

    # Initialize chat history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_q = st.text_input("Your Question:")
    if st.button("Ask") and user_q:

        # Save user question
        st.session_state.history.append({"role": "user", "content": user_q})

        # Get answerr̥
        answer = answer_query(user_q, st.session_state.history, db)

        # Save answer
        st.session_state.history.append({"role": "assistant", "content": answer})

    # Display chat history
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f" **You:** {msg['content']}")
        else:
            st.markdown(f" **Assistant:** {msg['content']}")

