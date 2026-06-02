# =========================================================
# NOTE FOR EVALUATOR
# =========================================================
# This project originally uses IBM watsonx.ai APIs as shown
# in the assignment instructions.
#
# However, IBM Cloud currently requires international card
# verification for account activation in my region.
#
# As a student, I do not have access to an international
# credit card, so I implemented the same RAG architecture
# using Google's Gemini API as an alternative.
#
# The overall pipeline remains identical:
# PDF Loader -> Text Splitter -> Embeddings ->
# Vector Database -> Retriever -> LLM -> Gradio UI
# =========================================================


# =========================================================
# Imports
# =========================================================
import os
os.environ["USE_TF"] = "0"

import warnings
from dotenv import load_dotenv

import gradio as gr

from langchain_google_genai import ChatGoogleGenerativeAI

from langchain_community.document_loaders import PyPDFLoader

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Chroma

from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.chains import RetrievalQA


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

warnings.filterwarnings("ignore")

def get_llm():

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.5
    )

    return llm



def document_loader(file):

    loader = PyPDFLoader(file)

    loaded_document = loader.load()

    return loaded_document


def text_splitter(data):

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_documents(data)

    return chunks


def embedding_model():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )

    return embeddings


def vector_database(chunks):

    embeddings = embedding_model()

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vectordb


def retriever(file):

    loaded_data = document_loader(file)

    chunks = text_splitter(loaded_data)

    vectordb = vector_database(chunks)

    retriever = vectordb.as_retriever()

    return retriever



def retriever_qa(file, query):

    llm = get_llm()

    retriever_obj = retriever(file)

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever_obj,
        return_source_documents=False
    )

    response = qa.invoke(query)

    return response["result"]


rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(
            label="Upload PDF File",
            file_count="single",
            file_types=[".pdf"],
            type="filepath"
        ),

        gr.Textbox(
            label="Input Query",
            lines=2,
            placeholder="Ask questions from the PDF..."
        )
    ],

    outputs=gr.Textbox(label="Answer"),

    title="PDF Question Answering Bot",

    description="Upload a PDF and ask questions using Gemini-powered RAG."
)


rag_application.launch(
    server_name="127.0.0.1",
    server_port=7860,
    inbrowser=True
)