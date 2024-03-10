# ----- importing dependencies -----
import os
import time
import tempfile
import streamlit as st

from dotenv                               import load_dotenv
from langchain_groq                       import ChatGroq
from langchain.chains                     import RetrievalQA
from langchain_text_splitters             import CharacterTextSplitter
from langchain_community.embeddings       import HuggingFaceEmbeddings
from langchain_community.vectorstores     import FAISS
from langchain_community.document_loaders import PyPDFLoader

from langchain_google_genai               import ChatGoogleGenerativeAI
from langchain_experimental.text_splitter import SemanticChunker

# ----- loading .env file -----
load_dotenv()

# ----- defining constants -----
embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
gemini_api_key = os.getenv("GEMINI_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")
llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="mixtral-8x7b-32768")
# llm = ChatGoogleGenerativeAI(model="gemini-pro",temperature=0.3,google_api_key=gemini_api_key,convert_system_message_to_human=True)
# text_splitter = SemanticChunker(embedding_model, breakpoint_threshold_type="percentile")

# ----- loading pdf file and making chunks -----
@st.cache_data(show_spinner=False)
def load_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    file_name = os.path.splitext(pdf_file.name)[0]
    loader = PyPDFLoader(temp_file_path)
    data = loader.load_and_split()
    docs = text_splitter.split_documents(data)
    os.remove(temp_file_path)
    return docs, file_name

# ----- creating vector database -----
def get_vectordb(docs,embedding_model):
    db = FAISS.from_documents(docs, embedding_model)
    return db

# ----- creating retriever using vectordb -----
def get_retriever(db):
    retriever = db.as_retriever()
    return retriever

# ----- creating retrieval chain -----
def create_retrieval_chain(llm,retriever):
    qa = RetrievalQA.from_llm(llm=llm, retriever=retriever)
    return qa

# ----- streaming response of the chatbot -----
def stream_data(response):
    for word in response.split(" "):
        yield word + " "
        time.sleep(0.03)

# ----- defining main function -----
def main():

    # --- setting page configuration ---
    st.set_page_config("Dokchat 2.0",":robot_face:","centered")

    # --- hiding header and footer ---
    hide_streamlit_style = """
                <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

    # --- creating sidebar ---
    with st.sidebar:
        col1, col2 = st.columns([2,5])
        with col1:
            st.image("dokchat 2.0.png",width=80)
        with col2:
            st.markdown("<h1>Dokchat 2.0</h1>",unsafe_allow_html=True)
        st.write("---")

        # --- initializing history session state and displaying history ---
        st.header("Your history:")
        if "history" not in st.session_state:
            st.session_state.history = []
        for i in range(len(st.session_state.history)):
            book = st.session_state.history[len(st.session_state.history)-i-1]
            st.write(book)

        st.write("---")

        # --- uploading pdf file from the user ---
        pdf_file = st.file_uploader("Upload your pdf file here...",accept_multiple_files=False)

    if pdf_file is None or pdf_file == " ":
        st.title(":wave: Hey! I am Dokchat 2.0")
        st.header("Kindly upload a pdf file and start a little chat.")
    else:
        st.title(":robot_face: Dokchat 2.0")
        st.write("---")
        # --- initializing chat session state and displaying chat ---
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        user_input = st.chat_input("Write your query here...")
        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            docs, file_name = load_pdf(pdf_file)

            if file_name not in st.session_state.history:
                st.session_state.history.append(file_name)

            # --- generating response ---
            with st.spinner("Getting answer for you..."):
                db = get_vectordb(docs, embedding_model)
                retriever = get_retriever(db)
                qa = create_retrieval_chain(llm, retriever)
                result = qa.invoke({"query": user_input})
                response = result["result"]

            with st.chat_message("assistant"):
                st.write_stream(stream_data(response))
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()