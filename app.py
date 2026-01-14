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
    page_title="DocMind",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load external CSS
def load_css(file_path):
    with open(file_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")


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
    try:
        return file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Failed to read file: {str(e)}")
        return ""


def get_file_hash(file) -> str:
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()[:8]


def process_documents(files) -> tuple[list[str], list[dict]]:
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
            file_type = "MD"
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
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def init_pinecone():
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        st.error("Pinecone API key not configured.")
        return None
        
    pc = Pinecone(api_key=api_key)
    existing_indexes = [idx.name for idx in pc.list_indexes()]
    
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    
    return pc.Index(PINECONE_INDEX_NAME)


def create_vector_store(chunks: list[str]):
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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("Groq API key not configured.")
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
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(PINECONE_INDEX_NAME)
        index.delete(delete_all=True)
    except Exception:
        pass


# ==============================================================================
# SIDEBAR
# ==============================================================================
with st.sidebar:
    st.markdown('<div class="brand-title">DocMind</div>', unsafe_allow_html=True)
    st.markdown('<div class="brand-subtitle">Intelligent Document Analysis</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Upload</div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drop files here",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("Process Documents", type="primary", use_container_width=True):
            with st.status("Processing...", expanded=True) as status:
                st.write("Extracting content...")
                chunks, file_info = process_documents(uploaded_files)
                
                if not chunks:
                    st.error("No content could be extracted.")
                else:
                    st.write(f"Creating {len(chunks)} vectors...")
                    vector_store = create_vector_store(chunks)
                    
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.processed_files = file_info
                        st.session_state.total_chunks = len(chunks)
                        status.update(label="Ready", state="complete")
                    else:
                        status.update(label="Failed", state="error")
    
    if st.session_state.processed_files:
        st.markdown('<div class="section-header">Documents</div>', unsafe_allow_html=True)
        
        for f in st.session_state.processed_files:
            st.markdown(f'''
                <div class="file-item">
                    <div class="file-name">{f['name']}</div>
                    <div class="file-meta">{f['type']} · {f['size']}</div>
                </div>
            ''', unsafe_allow_html=True)
        
        st.markdown(f'''
            <div class="stats-container">
                <div class="stat-box">
                    <div class="stat-value">{len(st.session_state.processed_files)}</div>
                    <div class="stat-label">Files</div>
                </div>
                <div class="stat-box">
                    <div class="stat-value">{st.session_state.total_chunks}</div>
                    <div class="stat-label">Chunks</div>
                </div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('<div class="section-header">Actions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.rerun()
    
    with col2:
        if st.button("Reset All", use_container_width=True):
            clear_pinecone_index()
            st.session_state.vector_store = None
            st.session_state.messages = []
            st.session_state.memory.clear()
            st.session_state.processed_files = []
            st.session_state.total_chunks = 0
            st.rerun()


# ==============================================================================
# MAIN CONTENT
# ==============================================================================
if st.session_state.vector_store is None:
    st.markdown("""
        <div class="welcome-container">
            <div class="welcome-icon">◆</div>
            <h1>DocMind</h1>
            <p class="welcome-text">
                Upload your documents and start asking questions. 
                Your data is processed securely and never stored permanently.
            </p>
        </div>
    """, unsafe_allow_html=True)
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            if message["role"] == "assistant" and "sources" in message:
                with st.expander("View sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(
                            f'<div class="source-item"><strong>Source {i}</strong><br>{source[:400]}...</div>', 
                            unsafe_allow_html=True
                        )

if prompt := st.chat_input("Ask a question about your documents..."):
    
    if st.session_state.vector_store is None:
        st.warning("Please upload and process documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            chain = get_conversation_chain(st.session_state.vector_store)
            
            if chain is None:
                st.error("Failed to initialize.")
            else:
                with st.spinner(""):
                    try:
                        response = chain.invoke({"question": prompt})
                        answer = response["answer"]
                        sources = [doc.page_content for doc in response["source_documents"]]
                        
                        st.markdown(answer)
                        
                        with st.expander("View sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(
                                    f'<div class="source-item"><strong>Source {i}</strong><br>{source[:400]}...</div>', 
                                    unsafe_allow_html=True
                                )
                        
                        st.session_state.messages.append({
                            "role": "assistant", 
                            "content": answer,
                            "sources": sources
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")