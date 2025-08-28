from langchain_chroma import Chroma
from langchain_gigachat import GigaChatEmbeddings
from chromadb.config import Settings
from langchain_gigachat import GigaChat

EMBEDDINGS = GigaChatEmbeddings(verify_ssl_certs=False, scope='GIGACHAT_API_PERS')

VECTOR_STORE = Chroma(
    embedding_function=EMBEDDINGS,
    client_settings=Settings(anonymized_telemetry=False),
    persist_directory='./vector_db'
)

LLM = GigaChat(verify_ssl_certs=False, model="GigaChat-2-Max", temperature=0, scope='GIGACHAT_API_PERS')