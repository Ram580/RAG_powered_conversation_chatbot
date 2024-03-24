import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv

import os
import google.generativeai as genai

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory 
from langchain.chains import ConversationalRetrievalChain


# Parsing the uploaded documents and creating a single text blob
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

# creating chunks of the text blob
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)
    return chunks

# Create Embeddings using Huggingface Embeddings
import sentence_transformers
from langchain.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


# Load environment variables to get Pinecone API Key and env
load_dotenv()
# Access the value of PINECONE_API_KEY
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")

# initialize pinecone
import pinecone
# initialize pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "rag-chatbot" # put in the name of your pinecone index here


#  Function that indexes documents into Pinecone
from langchain.vectorstores import Pinecone

# Load the data into pinecone database
def get_vector_store(text_chunks):
   #docsearch = Pinecone.from_texts(chunked_data, embeddings, index_name=index_name)
   index = Pinecone.from_texts([t for t in text_chunks], embeddings, index_name=index_name)
   return index

