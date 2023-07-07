import requests
from bs4 import BeautifulSoup
import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Get the OpenAI API key from Streamlit secrets
openai_api_key = st.secrets["API_KEYS"]["openai"]

def generate_response(url, query_text):
    # Send HTTP request to URL
    response = requests.get(url)
    # Parse HTML from response
    soup = BeautifulSoup(response.text, 'html.parser')
    # Extract text from parsed HTML
    documents = [soup.get_text()]
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.create_documents(documents)
    # Select embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    # Create a vectorstore from documents
    db = Chroma.from_documents(texts, embeddings)
    # Create retriever interface
    retriever = db.as_retriever()
    # Create QA chain
    qa = RetrievalQA.from_chain_type(llm=OpenAI(openai_api_key=openai_api_key), chain_type='stuff', retriever=retriever)
    return qa.run(query_text)

# Page title
st.set_page_config(page_title='🦜🔗 Ask the Doc App')
st.title('🦜🔗 Ask the Doc App')

# URL input
url = st.text_input('Enter URL:', placeholder = 'Please provide a URL.', value='')

# Query text
query_text = st.text_input('Enter your question:', placeholder = 'Please provide a short summary.', disabled=not url)

# Form input and query
result = []
if url and query_text:
    with st.spinner('Calculating...'):
        response = generate_response(url, query_text)
        result.append(response)

if len(result):
    st.info(response)
