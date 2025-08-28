from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma

from .llm import VECTOR_STORE

import glob
import os
from dotenv import find_dotenv, load_dotenv
import getpass

list_of_files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc_files", "pdf", "*"))

load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

class PDFVectorizer:
    def __init__(self, list_of_files):
        self.list_of_files = list_of_files
        self.docs = [PyPDFLoader("doc_files/" + doc).load() for doc in list_of_files]
        # flatted list
        self.documents = [item for sublist in self.docs for item in sublist]
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
        )
        self.documents = self.text_splitter.split_documents(self.documents)
        self.vector_store = VECTOR_STORE.add_documents(self.documents)

    def __call__(self) -> Chroma:
        return self.vector_store

if __name__ == "__main__":
    vectorizer = PDFVectorizer(list_of_files)
    vectorizer()