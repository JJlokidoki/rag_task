import re
import logging
import psycopg2
import numpy as np
from typing import Any, List, Optional, Tuple, Dict
from psycopg2.extras import execute_values
from pgvector.psycopg2 import register_vector
from framework.RAG_tools.abstract.v1_0_0.abstract import VectorStore

class PGVectorStore(VectorStore):
    SAFE_NAME_PATTERN = re.compile(r'^[a-z][a-z0-9_]{0,62}$')
    DEFAULT_PROBES = 10
    DEFAULT_IVFFLAT_LISTS = 100
    DEFAULT_HNSW_M = 16
    DEFAULT_HNSW_EF_CONSTRUCTION = 64

    def __init__(
        self,
        connection_string: str,
        embedder: Any,
        table_name: str,
        embedding_dimension: Optional[int] = None,
        vector_type: str = 'auto',  # 'auto', 'vector', 'halfvec'. dimension of the vector < 2000, dimension of halfvec < 4000. 
        # EmbeddingsGigaR dimension  2560, nomic dimension 768 
        splitter: Optional[Any] = None,
        index_type: Optional[str] = 'ivfflat',
        lists: Optional[int] = None,
        probes: Optional[int] = None,
        m: Optional[int] = None,
        ef_construction: Optional[int] = None,
        recreate_table: bool = False
    ):
        self._validate_table_name(table_name)
        self.connection_string = connection_string
        self.embedder = embedder
        self.table_name = table_name.lower()
        self.splitter = splitter
        self.embedding_dimension = self._resolve_embedding_dimension(embedding_dimension)
        if vector_type.lower() == 'auto':
            # HALFVEC для больших эмбеддингов, VECTOR для остальных
            self.vector_type_in_db = 'HALFVEC' if self.embedding_dimension > 1024 else 'VECTOR'
            logging.info(f"Автоматический выбор типа вектора для размерности {self.embedding_dimension}: {self.vector_type_in_db}")
        elif vector_type.lower() in ['vector', 'halfvec']:
            self.vector_type_in_db = vector_type.upper()
        else:
            raise ValueError(f"Неподдерживаемый тип вектора: {vector_type}. Допустимы 'auto', 'vector', 'halfvec'.")
            
        # Класс операторов теперь зависит от выбранного типа
        self.pg_index_opclass = f'{self.vector_type_in_db.lower()}_cosine_ops'
        self.pg_operator_for_search = '<=>'
        self.probes = probes or self.DEFAULT_PROBES
        
        self.conn = None
        self._connect()
        self._initialize_db(
            recreate_table=recreate_table,
            index_type=index_type,
            lists=lists,
            m=m,
            ef_construction=ef_construction
        )

    def _validate_table_name(self, name: str):
        """Проверяет безопасность имени таблицы"""
        if not self.SAFE_NAME_PATTERN.match(name):
            raise ValueError(f"Недопустимое имя таблицы: {name}. Разрешены только строчные буквы, цифры и подчеркивания.")

    def _resolve_embedding_dimension(self, dimension: Optional[int]) -> int:
        """Определяет размерность эмбеддингов"""
        if dimension:
            return dimension
        
        if hasattr(self.embedder, 'dimension'):
            return self.embedder.dimension
        
        try:
            sample = self.embedder.encode(['sample'])[0]
            return len(sample)
        except Exception as e:
            logging.exception("Не удалось определить размерность эмбеддинга")
            raise ValueError("Необходимо явно указать embedding_dimension") from e

    def _connect(self):
        """Устанавливает подключение к БД с обработкой ошибок"""
        try:
            if self.conn is None or self.conn.closed:
                self.conn = psycopg2.connect(self.connection_string)
                register_vector(self.conn)
                logging.info(f"Успешное подключение к БД для таблицы '{self.table_name}'")
        except psycopg2.Error as e:
            logging.exception("Ошибка подключения к PostgreSQL")
            raise ConnectionError("Не удалось подключиться к БД") from e

    def _get_row_count(self) -> int:
        """Возвращает количество строк в таблице"""
        with self.conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self.table_name};")
            return cur.fetchone()[0] or 0
        return 0

    def _initialize_db(self, **kwargs):
        """Инициализирует таблицу и индексы в БД"""
        print(self.embedding_dimension)
        with self.conn.cursor() as cur:
            # Создание расширения и таблицы
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            
            if kwargs['recreate_table']:
                cur.execute(f"DROP TABLE IF EXISTS {self.table_name} CASCADE;")
                logging.info(f"Таблица {self.table_name} пересоздана")

            create_table_query = f"""
                CREATE TABLE IF NOT EXISTS {self.table_name} (
                    id SERIAL PRIMARY KEY,
                    text_content TEXT NOT NULL,
                    embedding {self.vector_type_in_db}({self.embedding_dimension})
                );
            """
            cur.execute(create_table_query)
            logging.info(f"Таблица {self.table_name} проверена/создана с типом колонки {self.vector_type_in_db}.")
            
            # Создание индекса
            self._create_index(cur, **kwargs)
            
        self.conn.commit()

    def _create_index(self, cursor, index_type: Optional[str], **params):
        """Создает векторный индекс с оптимальными параметрами"""
        if not index_type:
            logging.info("Создание индекса пропущено")
            return

        index_name = f"{self.table_name}_{index_type}_idx"
        cursor.execute("SELECT EXISTS(SELECT 1 FROM pg_indexes WHERE indexname = %s);", (index_name,))
        
        if cursor.fetchone()[0]:
            logging.info(f"Индекс {index_name} уже существует")
            return

        # Автоматическая настройка параметров индекса
        index_query = None
        if index_type == 'ivfflat':
            lists = params.get('lists') or self.DEFAULT_IVFFLAT_LISTS
            index_query = f"""
                CREATE INDEX {index_name} 
                ON {self.table_name} 
                USING ivfflat (embedding {self.pg_index_opclass}) 
                WITH (lists = {lists});
            """
        elif index_type == 'hnsw':
            m = params.get('m') or self.DEFAULT_HNSW_M
            ef_construction = params.get('ef_construction') or self.DEFAULT_HNSW_EF_CONSTRUCTION
            index_query = f"""
                CREATE INDEX {index_name} 
                ON {self.table_name} 
                USING hnsw (embedding {self.pg_index_opclass}) 
                WITH (m = {m}, ef_construction = {ef_construction});
            """
        
        if index_query:
            cursor.execute(index_query)
            logging.info(f"Создан индекс типа {index_type}")
        else:
            logging.warning(f"Неподдерживаемый тип индекса: {index_type}")

    def add(self, texts: List[str], metadatas: Optional[List[dict]] = None):
        """Добавляет тексты с эмбеддингами в хранилище"""
        if not texts:
            logging.debug("Пропущена пустая вставка")
            return

        try:
            embeddings = self.embedder.encode(texts)
            if len(embeddings) != len(texts):
                raise ValueError("Количество эмбеддингов не совпадает с количеством текстов")
        except Exception as e:
            logging.exception("Ошибка генерации эмбеддингов")
            return

        # Подготовка данных для вставки
        records = []
        for text, emb in zip(texts, embeddings):
            try:
                vector = np.array(emb, dtype=np.float32)
                records.append((text, vector))
            except Exception as e:
                logging.warning(f"Ошибка конвертации эмбеддинга: {str(e)}")

        # Пакетная вставка
        try:
            with self.conn.cursor() as cur:
                execute_values(
                    cur,
                    f"INSERT INTO {self.table_name} (text_content, embedding) VALUES %s",
                    records,
                    page_size=500
                )
            self.conn.commit()
            logging.info(f"Добавлено {len(records)} записей в {self.table_name}")
        except Exception as e:
            self.conn.rollback()
            logging.exception("Ошибка вставки данных")

    def get(self, query_text: str, k: int) -> List[Tuple[float, str]]:
        """Поиск похожих текстов по запросу"""
        if not query_text or k < 1:
            return []

        try:
            embedding = np.array(
                self.embedder.encode([query_text])[0],
                dtype=np.float32
            )
        except Exception as e:
            logging.exception("Ошибка генерации эмбеддинга запроса")
            return []

        try:
            with self.conn.cursor() as cur:
                # Автоматическая настройка поиска
                self._configure_search(cur)
                
                cur.execute(f"""
                    SELECT text_content, 
                           embedding {self.pg_operator_for_search} %s AS distance
                    FROM {self.table_name}
                    ORDER BY distance ASC
                    LIMIT %s;
                """, (embedding, k))
                
                return [(float(dist), text) for text, dist in cur.fetchall()]
        except Exception as e:
            logging.exception("Ошибка поиска в БД")
            return []

    def _configure_search(self, cursor):
        """Настраивает параметры поиска для текущей сессии"""
        if not self.probes:
            return
            
        cursor.execute("""
            SELECT am.amname
            FROM pg_index i
            JOIN pg_class c ON c.oid = i.indexrelid
            JOIN pg_am am ON am.oid = c.relam
            WHERE c.relname LIKE %s
            LIMIT 1;
        """, (f"{self.table_name}_%",))
        
        if (result := cursor.fetchone()) and result[0] == 'ivfflat':
            cursor.execute(f"SET LOCAL ivfflat.probes = {self.probes};")
            logging.debug(f"Установлено probes = {self.probes}")

    def delete(self, ids: Optional[List[int]] = None, delete_all: bool = False) -> bool:
        """Удаляет данные по ID или полностью очищает таблицу"""
        try:
            with self.conn.cursor() as cur:
                if delete_all:
                    cur.execute(f"TRUNCATE TABLE {self.table_name};")
                    logging.info(f"Таблица {self.table_name} очищена")
                    self.conn.commit()
                    return True
                
                if ids:
                    cur.execute(
                        f"DELETE FROM {self.table_name} WHERE id = ANY(%s);",
                        (ids,)
                    )
                    logging.info(f"Удалено {cur.rowcount} записей")
                    self.conn.commit()
                    return True
            return False
        except Exception as e:
            self.conn.rollback()
            logging.exception("Ошибка удаления данных")
            return False

    def close(self):
        """Безопасно закрывает соединение с БД"""
        if self.conn and not self.conn.closed:
            self.conn.close()
            logging.info(f"Соединение для {self.table_name} закрыто")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()