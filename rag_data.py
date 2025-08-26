from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_gigachat import GigaChatEmbeddings

from chromadb.config import Settings

import os
from dotenv import find_dotenv, load_dotenv
import getpass

list_of_files = [f for f in os.listdir(os.path.abspath(__file__) + 'doc_files/')]

# [
# 'Положение Банка России от 30 января 2025 г N 851 П Об установлении обязательных .pdf',
# 'Приказ ФАПСИ от 13 июня 2001 г N 152 Об утверждении Инструкции об организации и .pdf',
# 'Приказ ФСБ РФ от 27 декабря 2011 г N 795 Об утверждении Требований к форме квали.pdf',
# 'Приказ ФСБ РФ от 27 декабря 2011 г N 796 Об утверждении Требований к средствам э.pdf',
# 'Приказ ФСБ РФ от 9 февраля 2005 г N 66 Об утверждении Положения о разработке про.pdf',
# 'Федеральный закон от 27 июля 2006 г N 152 ФЗ О персональных данных с изменениями.pdf',
# 'Федеральный закон от 6 апреля 2011 г N 63 ФЗ Об электронной подписи с изменениям.pdf'
# ]


load_dotenv(find_dotenv())

if "GIGACHAT_CREDENTIALS" not in os.environ:
    os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

# loader = PyPDFLoader("doc_files/" + list_of_files[2])
# documents = loader.load()
docs = [PyPDFLoader(doc).load() for doc in list_of_files]

# flatted list
documents = [item for sublist in docs for item in sublist]
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)

documents = text_splitter.split_documents(documents)
print(f"Total documents: {len(documents)}")

embeddings = GigaChatEmbeddings(verify_ssl_certs=False)

vector_store = Chroma.from_documents(
    documents,
    embeddings,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory='./vector_db'
)

