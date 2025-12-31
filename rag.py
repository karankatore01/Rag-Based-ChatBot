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
from langchain_huggingface import HuggingFaceEmbeddings



# STREAMLIT CONFIG (FIX TORCH BUG)

st.set_page_config(page_title="RAG Chatbot", layout="centered")



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


# 2. OCR FUNCTION

def extract_text_from_pdf(pdf_path: str) -> str:
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


# 3. VECTOR DB (CACHED)

@st.cache_resource(show_spinner=False)
def build_vector_db(text: str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)
    docs = [Document(page_content=c) for c in chunks]

    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        encode_kwargs={"normalize_embeddings": True}
    )

    return Chroma.from_documents(docs, embeddings)



# 4. PROMPT BUILDER

def build_prompt(question: str, context: str, history_text: str) -> str:
    return f"""
You are an AI assistant helping a user understand a document.

Use conversation history for continuity, but answer ONLY using the document context.

--- CHAT HISTORY ---
{history_text}

--- DOCUMENT CONTEXT ---
{context}

QUESTION:
{question}

RULES:
- Use ONLY document context.
- If answer not found, say:
  "The document does not provide enough information."
- Be concise and factual.
- Do NOT hallucinate.

ANSWER:
"""



# 5. FAST QWEN CALL

def ask_qwen(prompt: str) -> str:
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",  # 🚀 FAST MODEL
        messages=[
            {"role": "system", "content": "You are a safe and accurate RAG assistant."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=300,
        temperature=0,
        top_p=1
    )

    return response.choices[0].message["content"].strip()



# 6. RAG ANSWER FUNCTION

def answer_query(question: str, history: list, db: Chroma) -> str:

    docs = db.similarity_search(question, k=3)
    context = "\n\n".join([d.page_content[:600] for d in docs])

    history_text = ""
    for msg in history[-6:]:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        history_text += f"{role}: {msg['content']}\n"

    prompt = build_prompt(question, context, history_text)
    return ask_qwen(prompt)



# 7. UI STYLING
st.markdown("""
<style>
.main {background-color: #0E1117;}

.user-bubble {
    background-color: #1F6FEB;
    padding: 12px 16px;
    border-radius: 14px;
    color: white;
    margin-left: auto;
    margin-bottom: 10px;
    width: fit-content;
}

.assistant-bubble {
    background-color: #30363D;
    padding: 12px 16px;
    border-radius: 14px;
    color: #E6EDF3;
    margin-right: auto;
    margin-bottom: 10px;
    width: fit-content;
}

.stTextInput input {
    background-color: #161B22;
    color: white;
}

.stButton button {
    background-color: #238636;
    color: white;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)


# 8. STREAMLIT APP

st.title("📄 Fast Document Chatbot (RAG + Qwen)")
st.write("Upload a PDF and chat with it in real time.")

uploaded_pdf = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_pdf:
    pdf_path = "temp.pdf"
    with open(pdf_path, "wb") as f:
        f.write(uploaded_pdf.read())

    st.info("🔍 Extracting text (OCR)...")
    text = extract_text_from_pdf(pdf_path)
    st.success("Text extracted!")

    st.info("⚡ Building vector database...")
    db = build_vector_db(text)
    st.success("Ready! Ask your questions.")

    if "history" not in st.session_state:
        st.session_state.history = []

    user_q = st.text_input("Ask a question")

    if st.button("Ask") and user_q:
        st.session_state.history.append({"role": "user", "content": user_q})
        answer = answer_query(user_q, st.session_state.history, db)
        st.session_state.history.append({"role": "assistant", "content": answer})

    st.markdown("### 💬 Chat")
    for msg in st.session_state.history:
        if msg["role"] == "user":
            st.markdown(f"<div class='user-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='assistant-bubble'>{msg['content']}</div>", unsafe_allow_html=True)
