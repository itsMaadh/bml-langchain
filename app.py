import os
import streamlit as st

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI

from dotenv import load_dotenv

# Env handling
load_dotenv()
os.environ.get("OPENAI_API_KEY")


@st.cache_resource()
def load_data_and_create_vectors():
    # Load the data
    loader = UnstructuredHTMLLoader("pages/faq.html")
    html_data = loader.load()

    # Split the data into chunks
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(html_data)

    # Create the vector database
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(texts, embedding=embeddings, persist_directory=".")
    vectordb.persist()

    return vectordb


# Create the conversational chain and load the data
vectordb = load_data_and_create_vectors()
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
pdf_qa = ConversationalRetrievalChain.from_llm(
    OpenAI(temperature=0.1), vectordb.as_retriever(), memory=memory
)

# App framework
st.title("💵 Bank of Maldives GPT")
st.markdown(
    "This is a demo of the conversational retrieval chain. The model is trained on the <a href='https://www.bankofmaldives.com.mv/faqs' target='_blank'>Bank of Maldives FAQ</a> page only. The model is not perfect, but it`s a good start. Try asking questions like 'When is the bank open?' or 'How much would the MobilePay fee be for 700 USD?'",
    unsafe_allow_html=True,
)
prompt = st.text_input("Here to answer your banking related questions..")

# Run the chain
if prompt:
    result = pdf_qa({"question": prompt})
    st.write(result["answer"])
