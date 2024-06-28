import streamlit as st
from llm_chains import load_normal_chain, load_pdf_chat_chain
from utils import get_timestamp, load_config, get_avatar
from langchain.chains import LLMChain
from pydantic.v1 import BaseSettings
from pdf_handler import add_documents_to_db, extract_text_from_pdf
from html_templates import css
from database_operations import load_last_k_text_messages, save_text_message, load_messages, get_all_chat_history_ids, delete_chat_history
import sqlite3
import time

config = load_config()

@st.cache_resource
def load_chain():
    if st.session_state.pdf_chat:
        print("Loading PDF chat chain")
        return load_pdf_chat_chain()
    return load_normal_chain()

def toggle_pdf_chat():
    st.session_state.pdf_chat = True
    clear_cache()

def get_session_key():
    if st.session_state.session_key == "new_session":
        st.session_state.new_session_key = get_timestamp()
        return st.session_state.new_session_key
    return st.session_state.session_key

def delete_chat_session_history():
    delete_chat_history(st.session_state.session_key)
    st.session_state.session_index_tracker = "new_session"

def clear_cache():
    st.cache_resource.clear()

def split_questions(text):
    questions = text.split('?')
    return [q.strip() + '?' for q in questions if q.strip()]

def display_chat_history(chat_container, session_key):
    with chat_container:
        chat_history_messages = load_messages(session_key)
        for message in chat_history_messages:
            with st.chat_message(name=message["sender_type"], avatar=get_avatar(message["sender_type"])):
                if message["message_type"] == "text":
                    st.write(message["content"])

def main():
    st.write(css, unsafe_allow_html=True)
    response_time = 0

    if "db_conn" not in st.session_state:
        st.session_state.session_key = "new_session"
        st.session_state.new_session_key = None
        st.session_state.session_index_tracker = "new_session"
        st.session_state.db_conn = sqlite3.connect(config["chat_sessions_database_path"], check_same_thread=False)
        st.session_state.pdf_uploader_key = 1
        st.session_state.question_uploader_key = 0

    if st.session_state.session_key == "new_session" and st.session_state.new_session_key:
        st.session_state.session_index_tracker = st.session_state.new_session_key
        st.session_state.new_session_key = None

    st.sidebar.title("Chat Sessions")
    chat_sessions = ["new_session"] + get_all_chat_history_ids()
    index = chat_sessions.index(st.session_state.session_index_tracker)
    st.sidebar.selectbox("Select a chat session", chat_sessions, key="session_key", index=index)

    pdf_toggle_col, _ = st.sidebar.columns(2)
    pdf_toggle_col.toggle("PDF Chat", key="pdf_chat", value=False)

    delete_chat_col, clear_cache_col = st.sidebar.columns(2)
    delete_chat_col.button("Delete Chat Session", on_click=delete_chat_session_history)
    clear_cache_col.button("Clear Cache", on_click=clear_cache)

    chat_container = st.container()
    user_input = st.chat_input("Type your message here", key="user_input")

    uploaded_pdf = st.sidebar.file_uploader(
        "Upload a PDF file", accept_multiple_files=True,
        key=st.session_state.pdf_uploader_key, type=["pdf"], on_change=toggle_pdf_chat
    )

    questions_uploaded = st.sidebar.file_uploader(
        "Upload your questions", key=st.session_state.question_uploader_key, on_change=toggle_pdf_chat
    )

    if uploaded_pdf:
        with st.spinner("Processing PDF..."):
            vector_db = add_documents_to_db(uploaded_pdf)
            st.session_state.pdf_uploader_key += 2

    if questions_uploaded and not user_input:
        with st.spinner("Processing questions..."):
            st.session_state.question_uploader_key += 2
            text = extract_text_from_pdf(questions_uploaded)
            questions = split_questions(text)
            llm_chain = load_chain()

            while questions:
                question = questions.pop(0)
                llm_answer = llm_chain.run(
                    user_input=question,
                    chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"])
                )

                save_text_message(get_session_key(), "human", question)
                save_text_message(get_session_key(), "ai", llm_answer)

                if st.session_state.session_key != "new_session" or st.session_state.new_session_key:
                    display_chat_history(chat_container, get_session_key())
        
        questions_uploaded = None

    if user_input and not questions_uploaded:
        start_time = time.time()
        llm_chain = load_chain()
        llm_answer = llm_chain.run(
            user_input=user_input,
            chat_history=load_last_k_text_messages(get_session_key(), config["chat_config"]["chat_memory_length"])
        )
        response_time = time.time() - start_time

        save_text_message(get_session_key(), "human", user_input)
        save_text_message(get_session_key(), "ai", llm_answer)

        if st.session_state.session_key != "new_session" or st.session_state.new_session_key:
            display_chat_history(chat_container, get_session_key())
        user_input = None

    #if st.session_state.session_key == "new_session" and st.session_state.new_session_key:
    #    st.rerun()

if __name__ == "__main__":
    main()
