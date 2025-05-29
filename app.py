import os
import time
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based solely on the provided context. If the context does not contain enough information to answer the question fully, state that clearly and provide a concise summary of the relevant information available. For questions about the main topic, summarize the primary focus of the document.

    <context>
    {context}
    </context>

    Question: {input}
    Answer:
    """
)

def create_vector_embeddings():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")
        
        st.session_state.loader = PyPDFDirectoryLoader("research-papers")
        st.session_state.docs = st.session_state.loader.load()
        
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        st.session_state.vectors = FAISS.from_documents(st.session_state.split_docs, st.session_state.embeddings)


st.title("Document Query Tool Using Ollama 📊")
user_prompt = st.text_input("Enter your query from the research paper")

if st.button("Create Vector Database"):
    create_vector_embeddings()
    st.write("Your vector database has been created! Enter your query now.")

if user_prompt:
    doc_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    rag_chain = create_retrieval_chain(retriever, doc_chain)

    start_time = time.process_time()
    response = rag_chain.invoke({"input": user_prompt})
    end_time = time.process_time()
    st.write(f"Time taken: {end_time - start_time:.2f} seconds")
    st.write(response['answer'])

    st.write("Other relevant results: 📑")
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')