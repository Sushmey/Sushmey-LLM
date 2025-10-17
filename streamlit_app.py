import streamlit as st
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline
from langchain.llms.base import LLM
from google.generativeai import configure, GenerativeModel
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
import os
import tempfile
from langchain_community.document_loaders import UnstructuredXMLLoader
from langchain_core.document_loaders.base import BaseLoader
import pypdf
import docx
import pandas as pd
from bs4 import BeautifulSoup
from langchain_community.document_loaders import PyPDFDirectoryLoader

## Incase you want to use dotenv for env vars
# load_dotenv()
# configure(api_key=os.getenv("GEMINI_API_KEY"))

configure(api_key=st.secrets["GEMINI_API_KEY"])

# Gemini wrapper as a LangChain-compatible LLM
class GeminiLLM(LLM):
    model: GenerativeModel

    def _call(self, prompt: str, stop=None):
        response = self.model.generate_content(prompt)
        return response.text

    @property
    def _llm_type(self):
        return "gemini"

class MixedFileTypeLoader(BaseLoader):
    def __init__(self, file_path):
        self.file_path = file_path

    def load(self):
        file_extension = os.path.splitext(self.file_path)[1].lower()
        print(f"file_extension is ------------ {file_extension} and file path {os.path.splitext(self.file_path)}")
        if file_extension == '.pdf':
            return self._load_pdf()
        elif file_extension in ['.doc', '.docx']:
            return self._load_word()
        elif file_extension == '.txt':
            return TextLoader(
                file_path = self.file_path,
                ).load()
        elif file_extension == '.html':
            return self._load_html()
        elif file_extension == '.csv':
            return self._load_csv()
        elif file_extension =='.xml':
         return UnstructuredXMLLoader(self.file_path).load()
        elif file_extension in ['', '/']:
         return DirectoryLoader(
    path = self.file_path,
    loader_cls = PyPDFLoader).load()
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

    def _load_pdf(self):
        with open(self.file_path, 'rb') as f:
            reader = pypdf.PdfReader(f)
            text = ''
            for page in reader.pages:
                text = f"{text}{page.extract_text()}'\n'"
            return text.strip()

    def _load_word(self):
        doc = docx.Document(self.file_path)
        text = '\n'.join([para.text for para in doc.paragraphs])
        return text.strip()

    def _load_txt(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            return f.read().strip()

    def _load_html(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            soup = BeautifulSoup(f, 'html.parser')
            return soup.get_text().strip()

    def _load_csv(self):
        df = pd.read_csv(self.file_path)
        return df.to_string(index=False)


@st.cache_resource
def initial_retrieval_qa():  
 # Might have to change the glob regex to look for different kinds of files
 mixed_loader = MixedFileTypeLoader(file_path = 'Documents/')

 try:
  doc = mixed_loader.load()
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0);
  if(isinstance(doc,str)):
   doc = text_splitter.create_documents([doc])
  docs = text_splitter.split_documents(doc)
  print(type(docs))
  encoder =  HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
  db = FAISS.from_documents(documents=docs, embedding=encoder)
  retriever = db.as_retriever(search_kwargs = {"k":10})

  gemini_model = GenerativeModel("gemini-2.0-flash")
  llm = GeminiLLM(model=gemini_model)
  qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
  return qa
 except:
  raise

# Show title and description.
st.title("Get to know Sushmey!")
st.write(
 """Ask questions about Sushmey to know him better! 
 """

)

# Ask the user for a question via `st.text_area`.
question = st.text_area(
    "",
    placeholder="Who is Sushmey?",
    max_chars=400)

if question:
 answer = qa.run(question)
 # Stream the response to the app using `st.write_stream`.
 st.write(answer)










