import streamlit as st
from PyPDF2 import PdfReader
import os
from dotenv import load_dotenv
import logging

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# --- Basic Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

# --- App Secrets & Keys ---
LOGIN_PASSWORD = os.getenv("APP_PASSWORD")
API_KEY = os.getenv("GOOGLE_API_KEY")

# --- Page Configuration ---
st.set_page_config(
    page_title="PDF Chat AI",
    page_icon="📄",
    layout="wide"
)

# --- 1. LOGIN LOGIC ---
def login_page():
    st.title("🔐 PDF Chat AI - Login")
    st.markdown("Please enter the password to access the application.")

    with st.form("login_form"):
        password = st.text_input("Password", type="password", placeholder="Enter password...")
        submitted = st.form_submit_button("Login", type="primary", use_container_width=True)

        if submitted:
            if password == LOGIN_PASSWORD:
                st.session_state.authenticated = True
                st.success("✅ Login successful!")
                st.rerun()
            else:
                st.error("❌ Invalid password. Please try again.")

# --- 2. PDF PROCESSING LOGIC ---
@st.cache_data
def extract_pdf_text(pdf_file):
    """Extracts text from an uploaded PDF file."""
    if not pdf_file:
        return ""
    text = ""
    try:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Error extracting text: {e}")
        return ""

def create_text_chunks(text):
    """Splits text into manageable chunks."""
    if not text:
        return []
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return text_splitter.split_text(text)

@st.cache_resource
def create_vector_store(_text_chunks):
    """Creates a FAISS vector store from text chunks."""
    if not _text_chunks:
        return None
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=API_KEY)
        vector_store = FAISS.from_texts(_text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Failed to create knowledge base: {e}")
        return None

@st.cache_resource
def create_rag_chain(_vector_store, selected_model):
    """Creates the RAG chain for answering questions."""
    if not _vector_store:
        return None
    prompt_template = """
You are a helpful and knowledgeable AI assistant. You are answering questions about a document.
Your behavior should be friendly, like ChatGPT, and you should keep the conversation going even if the answer is not in the document.

Guidelines:
1. If the answer is found in the provided CONTEXT, answer concisely and clearly.
2. If the answer is NOT found, politely say you couldn't find it in the document, but offer to help with related or general information.
3. Always maintain a conversational tone, and avoid making up document-specific facts.
4. If the question is unrelated to the document, briefly clarify this but still give a friendly and useful response.

---
CONTEXT:
{context}

---
QUESTION:
{input}

---
ANSWER:
"""
    try:
        model = ChatGoogleGenerativeAI(
            model=selected_model,
            temperature=0.2,
            google_api_key=API_KEY
        )
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "input"])
        document_chain = create_stuff_documents_chain(model, prompt)
        retriever = _vector_store.as_retriever()
        rag_chain = create_retrieval_chain(retriever, document_chain)
        return rag_chain
    except Exception as e:
        st.error(f"Failed to create AI chain: {e}")
        return None

# --- 3. MAIN APPLICATION LOGIC ---
def main():
    if not API_KEY:
        st.error("⚠️ Google API Key not found! Please set the GOOGLE_API_KEY environment variable.")
        st.stop()

    # --- Session State Init ---
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "pdf_file" not in st.session_state:
        st.session_state.pdf_file = None
    if "rag_chain" not in st.session_state:
        st.session_state.rag_chain = None
    if "conversation_history" not in st.session_state:
        st.session_state.conversation_history = []
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "gemini-1.5-flash"

    # --- Authentication ---
    if not st.session_state.authenticated:
        login_page()
        return

    # --- Sidebar Controls ---
    with st.sidebar:
        st.header("Controls")
        st.write(f"Logged in as: **User**")
        
        model_choice = st.selectbox(
            "Select Model",
            ["Gemini 1.5 Flash", "Gemini 2.0 Flash", "Gemini 2.5 Flash"],
            index=0
        )
        model_map = {
            "Gemini 1.5 Flash": "gemini-1.5-flash",
            "Gemini 2.0 Flash": "gemini-2.0-flash",
            "Gemini 2.5 Flash": "gemini-2.5-flash"
        }
        st.session_state.selected_model = model_map[model_choice]

        if st.button("🚪 Logout", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        st.divider()
        st.info("Upload a PDF in the main area to begin.")

    # --- Main UI ---
    st.title("📄 Simple PDF Chat")

    # STEP 1: Upload PDF
    if st.session_state.pdf_file is None:
        st.header("Step 1: Upload your PDF")
        uploaded_file = st.file_uploader(
            "Choose a single PDF file to analyze.",
            type="pdf",
            accept_multiple_files=False
        )
        if uploaded_file:
            st.session_state.pdf_file = uploaded_file
            st.session_state.pdf_name = uploaded_file.name
            st.rerun()

    # STEP 2: Process Document
    elif st.session_state.rag_chain is None:
        st.header("Step 2: Process the Document")
        st.info(f"File to be processed: **{st.session_state.pdf_name}**")

        if st.button("🚀 Process PDF", type="primary", use_container_width=True):
            with st.spinner("Analyzing document... This may take a moment."):
                raw_text = extract_pdf_text(st.session_state.pdf_file)
                if not raw_text:
                    st.error("Could not extract text from PDF.")
                    st.session_state.pdf_file = None
                    st.rerun()
                    return

                text_chunks = create_text_chunks(raw_text)
                if not text_chunks:
                    st.error("Could not create text chunks.")
                    st.session_state.pdf_file = None
                    st.rerun()
                    return

                vector_store = create_vector_store(text_chunks)
                if not vector_store:
                    st.error("Failed to create the knowledge base.")
                    st.session_state.pdf_file = None
                    st.rerun()
                    return

                rag_chain = create_rag_chain(vector_store, st.session_state.selected_model)
                if rag_chain:
                    st.session_state.rag_chain = rag_chain
                    st.success("✅ Document processed successfully!")
                    st.balloons()
                    st.rerun()
                else:
                    st.error("Failed to create the AI chat chain.")
                    st.session_state.pdf_file = None
                    st.rerun()

    # STEP 3: Chat
    else:
        st.header(f"Step 3: Chat with '{st.session_state.pdf_name}'")

        # Show chat history
        for message in st.session_state.conversation_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about the document..."):
            st.session_state.conversation_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("🤖 Thinking..."):
                    try:
                        response = st.session_state.rag_chain.invoke({"input": prompt})
                        answer = response.get("answer", "I couldn't find an answer in the document.")
                        st.markdown(answer)
                        st.session_state.conversation_history.append({"role": "assistant", "content": answer})
                    except Exception as e:
                        error_message = f"An error occurred: {e}"
                        st.error(error_message)
                        st.session_state.conversation_history.append({"role": "assistant", "content": error_message})

if __name__ == "__main__":
    main()
