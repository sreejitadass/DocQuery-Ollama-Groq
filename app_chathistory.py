import streamlit as st
import os
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv

load_dotenv()
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

embeddings = OllamaEmbeddings(model="all-minilm:l6-v2")

st.title("Chat with Multiple PDF Documents! 📄💬")
st.write("Upload your PDF documents to start chatting with them.")
groq_api_key = st.text_input("Enter your Groq API Key", type="password")

if groq_api_key:
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="gemma2-9b-it")
    session_id = st.text_input("Enter a session ID to maintain chat history", value="default_session")

    if 'store' not in st.session_state:
        st.session_state.store = {}

    uploaded_files = st.file_uploader("Select any PDF files", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        documents = []
        for file in uploaded_files:
            temppdf = f"./temp.pdf"
            with open(temppdf, 'wb') as f:
                f.write(file.getvalue())
                file_name = file.name

            loader = PyPDFLoader(temppdf)
            docs = loader.load()
            documents.extend(docs)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        split_docs = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(documents=split_docs, embedding=embeddings)
        retriever = vector_store.as_retriever()

        contexualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, "
            "just reformulate it if needed and otherwise return it as is."
        )

        contexualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contexualize_q_system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        history_aware_retriever = create_history_aware_retriever(llm, retriever, contexualize_q_prompt)

        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer "
            "the question. If you don't know the answer, say that you "
            "don't know. Use three sentences maximum and keep the "
            "answer concise."
            "\n\n"
            "{context}"
        )

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{input}"),
            ]
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

        def get_session_history(session_id: str) -> BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id] = ChatMessageHistory()
            return st.session_state.store[session_id]

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain, get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer"
        )

        user_input = st.text_input("Ask a question about the documents:")
        if user_input:
            session_history = get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input, "chat_history": session_history.messages},
                config={
                    "configurable": {"session_id": session_id}
                },
            )
            st.success("Assistant: " + response["answer"])

else:
    st.warning("Please enter your Groq API Key to use the chat functionality.")
    st.stop()