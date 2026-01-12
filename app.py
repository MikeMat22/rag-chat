import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from pypdf import PdfReader
from pinecone import Pinecone, ServerlessSpec
import hashlib

load_dotenv()

# ==============================================================================
# CONFIG
# ==============================================================================
PINECONE_INDEX_NAME = "rag-chat"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.3-70b-versatile"

# ==============================================================================
# PAGE SETUP
# ==============================================================================
st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄",
    layout="wide"
)

# Custom CSS for clean look
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .upload-box {
        border: 2px dashed #ccc;
        border-radius: 8px;
        padding: 20px;
        text-align: center;
        margin-bottom: 20px;
    }
    .stats-box {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    .source-box {
        background-color: #f1f3f4;
        border-left: 3px solid #1a73e8;
        padding: 10px;
        margin: 5px 0;
        font-size: 0.9em;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# SESSION STATE
# ==============================================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

if "total_chunks" not in st.session_state:
    st.session_state.total_chunks = 0


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def extract_text_from_pdf(file) -> str:
    """Extract text from uploaded PDF file."""
    try:
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        st.error(f"Failed to read PDF: {str(e)}")
        return ""


def extract_text_from_txt(file) -> str:
    """Extract text from uploaded TXT file."""
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to read TXT file: {str(e)}")
        return ""


def get_file_hash(file) -> str:
    """Generate hash for file to track duplicates."""
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()[:8]


def process_documents(files) -> tuple[list[str], list[dict]]:
    """Process uploaded files and return chunks with metadata."""
    all_text = ""
    file_info = []
    
    for file in files:
        file_hash = get_file_hash(file)
        
        if file.name.endswith(".pdf"):
            text = extract_text_from_pdf(file)
            file_type = "PDF"
        elif file.name.endswith(".txt"):
            text = extract_text_from_txt(file)
            file_type = "TXT"
        elif file.name.endswith(".md"):
            text = extract_text_from_txt(file)
            file_type = "Markdown"
        else:
            continue
            
        if text:
            all_text += text + "\n\n"
            file_info.append({
                "name": file.name,
                "type": file_type,
                "size": f"{file.size / 1024:.1f} KB",
                "hash": file_hash
            })
    
    if not all_text.strip():
        return [], []
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = splitter.split_text(all_text)
    return chunks, file_info


@st.cache_resource
def get_embeddings():
    """Get cached embedding model."""
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def init_pinecone():
    """Initialize Pinecone client and create index if needed."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API key not found. Please set PINECONE_API_KEY in .env file.")
        return None
        
    pc = Pinecone(api_key=api_key)
    
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    
    return pc.Index(PINECONE_INDEX_NAME)


def create_vector_store(chunks: list[str]):
    """Create Pinecone vector store from text chunks."""
    embeddings = get_embeddings()
    index = init_pinecone()
    
    if index is None:
        return None
    
    vector_store = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=PINECONE_INDEX_NAME
    )
    
    return vector_store


def get_conversation_chain(vector_store):
    """Create conversational retrieval chain."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not found. Please set GROQ_API_KEY in .env file.")
        return None
        
    llm = ChatGroq(
        model=LLM_MODEL,
        temperature=0.3,
        api_key=api_key
    )
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 4}),
        memory=st.session_state.memory,
        return_source_documents=True
    )
    
    return chain


def clear_pinecone_index():
    """Delete all vectors from the index."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
    except Exception:
        pass


# ==============================================================================
# MAIN UI
# ==============================================================================
st.title("Document Q&A")
st.caption("Upload documents and ask questions about their content")

# Sidebar
with st.sidebar:
    st.header("Documents")
    
    uploaded_files = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="Supported formats: PDF, TXT, Markdown"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.status("Processing...", expanded=True) as status:
                st.write("Extracting text from documents...")
                chunks, file_info = process_documents(uploaded_files)
                
                if not chunks:
                    st.error("No text could be extracted from the uploaded files.")
                else:
                    st.write(f"Created {len(chunks)} text chunks")
                    st.write("Generating embeddings and storing in database...")
                    
                    vector_store = create_vector_store(chunks)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_files = file_info
                        st.session_state.total_chunks = len(chunks)
                        status.update(label="Processing complete", state="complete")
                    else:
                        status.update(label="Processing failed", state="error")
    
    # Show processed files
    if st.session_state.processed_files:
        st.divider()
        st.subheader("Processed Files")
        
        for f in st.session_state.processed_files:
            st.text(f"{f['name']}")
            st.caption(f"{f['type']} | {f['size']}")
        
        st.divider()
        st.caption(f"Total chunks: {st.session_state.total_chunks}")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        if st.button("Clear All", use_container_width=True):
            clear_pinecone_index()
            st.session_state.vector_store = None
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.rerun()
    
    st.divider()
    st.caption("Built with LangChain, Pinecone, and Groq")


# ==============================================================================
# CHAT INTERFACE
# ==============================================================================

# Show placeholder if no documents
if st.session_state.vector_store is None:
    st.info("Upload and process documents to start asking questions.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("View sources"):
                for i, source in enumerate(message["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.markdown(f'<div class="source-box">{source[:500]}...</div>', 
                               unsafe_allow_html=True)

# Chat input
if prompt := st.chat_input("Ask a question about your documents"):
    
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first.")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            chain = get_conversation_chain(st.session_state.vector_store)
            
            if chain is None:
                st.error("Failed to initialize the conversation chain.")
            else:
                with st.spinner("Searching documents..."):
                    try:
                        response = chain.invoke({"question": prompt})
                        answer = response["answer"]
                        sources = [doc.page_content for doc in response["source_documents"]]
                        
                        st.markdown(answer)
                        
                        with st.expander("View sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.markdown(f'<div class="source-box">{source[:500]}...</div>', 
                                           unsafe_allow_html=True)
                        
                        # Save message with sources
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error(f"Error: {error_msg}")