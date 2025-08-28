import pandas as pd
import os
import glob
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from .llm import VECTOR_STORE, EMBEDDINGS

from typing import List, Dict, Any, Optional
import hashlib

import os
from dotenv import find_dotenv, load_dotenv
import getpass


class XLSXLoader:
    """Класс для загрузки XLSX файлов в ChromaDB"""
    
    def __init__(self, vector_store: Optional[Chroma] = None, 
                 chunk_size: int = 1000, 
                 chunk_overlap: int = 200):
        """
        Инициализация загрузчика
        
        Args:
            vector_store: ChromaDB векторное хранилище
            chunk_size: Размер чанка в символах
            chunk_overlap: Перекрытие между чанками в символах
        """
        self.vector_store = vector_store or VECTOR_STORE
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Создаем text splitter с настраиваемыми параметрами
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_xlsx_to_documents(self, file_path: str, processing_mode: str = "rows") -> List[Document]:
        """
        Загружает XLSX файл и преобразует в Document объекты
        
        Args:
            file_path: Путь к XLSX файлу
            processing_mode: Режим обработки ('rows', 'columns', 'cells', 'sheets')
                - 'rows': каждая строка = отдельный документ
                - 'columns': каждая колонка = отдельный документ
                - 'cells': каждая ячейка = отдельный документ
                - 'sheets': каждый лист = отдельный документ
        
        Returns:
            Список Document объектов
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        # Загружаем все листы Excel файла
        excel_data = pd.read_excel(file_path, sheet_name=None)
        documents = []
        
        for sheet_name, df in excel_data.items():
            print(f"Обрабатываем лист: {sheet_name}")
            
            if processing_mode == "sheets":
                documents.extend(self._process_sheet_mode(df, sheet_name, file_path))
            elif processing_mode == "rows":
                documents.extend(self._process_rows_mode(df, sheet_name, file_path))
            elif processing_mode == "columns":
                documents.extend(self._process_columns_mode(df, sheet_name, file_path))
            elif processing_mode == "cells":
                documents.extend(self._process_cells_mode(df, sheet_name, file_path))
            else:
                raise ValueError(f"Неизвестный режим обработки: {processing_mode}")
        
        return documents
    
    def _process_sheet_mode(self, df: pd.DataFrame, sheet_name: str, file_path: str) -> List[Document]:
        """Обрабатывает весь лист как один документ"""
        content = f"Лист: {sheet_name}\n\n"
        
        # Добавляем заголовки
        content += "Заголовки: " + ", ".join(df.columns.astype(str)) + "\n\n"
        
        # Добавляем данные построчно
        for index, row in df.iterrows():
            row_data = []
            for col_name, value in row.items():
                if pd.notna(value):
                    row_data.append(f"{col_name}: {value}")
            
            if row_data:
                content += f"Строка {index + 1}: " + "; ".join(row_data) + "\n"
        
        return [Document(
            page_content=content,
            metadata={
                "source": file_path,
                "sheet_name": sheet_name,
                "type": "excel_sheet",
                "rows_count": len(df),
                "columns_count": len(df.columns)
            }
        )]
    
    def _process_rows_mode(self, df: pd.DataFrame, sheet_name: str, file_path: str) -> List[Document]:
        """Обрабатывает каждую строку как отдельный документ"""
        documents = []
        
        for index, row in df.iterrows():
            row_data = []
            for col_name, value in row.items():
                if pd.notna(value):
                    row_data.append(f"{col_name}: {value}")
            
            if row_data:
                content = f"Строка {index + 1} из листа '{sheet_name}':\n" + "\n".join(row_data)
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "sheet_name": sheet_name,
                        "row_index": index + 1,
                        "type": "excel_row"
                    }
                ))
        
        return documents
    
    def _process_columns_mode(self, df: pd.DataFrame, sheet_name: str, file_path: str) -> List[Document]:
        """Обрабатывает каждую колонку как отдельный документ"""
        documents = []
        
        for col_name in df.columns:
            values = df[col_name].dropna().astype(str).tolist()
            
            if values:
                content = f"Колонка '{col_name}' из листа '{sheet_name}':\n"
                content += "\n".join([f"- {value}" for value in values])
                
                documents.append(Document(
                    page_content=content,
                    metadata={
                        "source": file_path,
                        "sheet_name": sheet_name,
                        "column_name": col_name,
                        "values_count": len(values),
                        "type": "excel_column"
                    }
                ))
        
        return documents

    def _process_cells_mode(self, df: pd.DataFrame, sheet_name: str, file_path: str) -> List[Document]:
        """Обрабатывает каждую ячейку как отдельный документ"""
        documents = []
        
        for row_idx, row in df.iterrows():
            for col_name, value in row.items():
                if pd.notna(value) and str(value).strip():
                    content = f"Ячейка {col_name}:{row_idx + 1} из листа '{sheet_name}': {value}"
                    
                    documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": file_path,
                            "sheet_name": sheet_name,
                            "row_index": row_idx + 1,
                            "column_name": col_name,
                            "type": "excel_cell"
                        }
                    ))
        
        return documents
    
    def split_documents_into_chunks(self, documents: List[Document], 
                                   min_chunk_size: int = 50) -> List[Document]:
        """
        Разбивает документы на чанки используя text_splitter
        
        Args:
            documents: Список исходных документов
            min_chunk_size: Минимальный размер для чанкования (если меньше - не чанкуем)
        
        Returns:
            Список разбитых на чанки документов
        """
        print(f"🔪 Разбиваем документы на чанки (размер: {self.chunk_size}, перекрытие: {self.chunk_overlap})")
        
        chunked_documents = []
        original_count = len(documents)
        
        for doc in documents:
            # Если документ маленький, оставляем как есть
            if len(doc.page_content) <= min_chunk_size:
                chunked_documents.append(doc)
                continue
            
            # Разбиваем большие документы на чанки
            if len(doc.page_content) > self.chunk_size:
                chunks = self.text_splitter.split_documents([doc])
                
                # Добавляем информацию о чанках в метаданные
                for i, chunk in enumerate(chunks):
                    chunk.metadata.update({
                        "chunk_index": i,
                        "total_chunks": len(chunks),
                        "original_doc_length": len(doc.page_content),
                        "is_chunked": True
                    })
                
                chunked_documents.extend(chunks)
            else:
                # Документ подходящего размера - добавляем метаданные о том, что он не был разбит
                doc.metadata.update({
                    "chunk_index": 0,
                    "total_chunks": 1,
                    "original_doc_length": len(doc.page_content),
                    "is_chunked": False
                })
                chunked_documents.append(doc)
        
        print(f"📊 Исходных документов: {original_count}")
        print(f"📊 После чанкования: {len(chunked_documents)}")
        
        return chunked_documents
    
    def add_to_vector_store(self, file_path: str, processing_mode: str = "rows", 
                           enable_chunking: bool = True) -> Dict[str, Any]:
        """
        Добавляет XLSX файл в векторное хранилище с чанкованием
        
        Args:
            file_path: Путь к XLSX файлу
            processing_mode: Режим обработки
            enable_chunking: Включить чанкование документов
        
        Returns:
            Словарь с результатами операции
        """
        try:
            print(f"Загружаем файл: {file_path}")
            print(f"Режим обработки: {processing_mode}")
            print(f"Чанкование: {'включено' if enable_chunking else 'отключено'}")
            
            # Загружаем документы
            documents = self.load_xlsx_to_documents(file_path, processing_mode)
            
            if not documents:
                return {
                    "success": False,
                    "error": "Не удалось создать документы из файла",
                    "documents_count": 0,
                    "chunks_count": 0
                }
            
            print(f"📄 Создано документов: {len(documents)}")
            
            # Чанкование документов (если включено)
            if enable_chunking:
                documents = self.split_documents_into_chunks(documents)
            
            # Добавляем метаданные о файле
            file_hash = self._get_file_hash(file_path)
            for doc in documents:
                doc.metadata.update({
                    "file_hash": file_hash,
                    "processing_mode": processing_mode,
                    "chunking_enabled": enable_chunking,
                    "chunk_size": self.chunk_size if enable_chunking else None,
                    "chunk_overlap": self.chunk_overlap if enable_chunking else None,
                    "added_at": pd.Timestamp.now().isoformat()
                })
            
            # Добавляем в векторное хранилище
            print("📝 Добавляем документы в векторное хранилище...")
            
            self.vector_store.add_documents(documents)
            print("✅ Документы добавлены в основную коллекцию")
            
            return {
                "success": True,
                "documents_count": len(documents),
                "original_docs_count": len(self.load_xlsx_to_documents(file_path, processing_mode)) if enable_chunking else len(documents),
                "chunks_count": len(documents) if enable_chunking else 0,
                "file_path": file_path,
                "processing_mode": processing_mode,
                "chunking_enabled": enable_chunking,
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "file_hash": file_hash,
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "documents_count": 0,
                "chunks_count": 0
            }
    
    def _get_file_hash(self, file_path: str) -> str:
        """Вычисляет хеш файла для отслеживания изменений"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def preview_xlsx_structure(self, file_path: str) -> Dict[str, Any]:
        """
        Предварительный просмотр структуры XLSX файла
        
        Args:
            file_path: Путь к XLSX файлу
        
        Returns:
            Информация о структуре файла
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл {file_path} не найден")
        
        excel_data = pd.read_excel(file_path, sheet_name=None)
        structure = {
            "file_path": file_path,
            "sheets_count": len(excel_data),
            "sheets": {}
        }
        
        for sheet_name, df in excel_data.items():
            structure["sheets"][sheet_name] = {
                "rows_count": len(df),
                "columns_count": len(df.columns),
                "columns": list(df.columns),
                "sample_data": df.head(3).to_dict('records') if len(df) > 0 else []
            }
        
        return structure


def manual_main():
    load_dotenv(find_dotenv())

    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")
    
    """Основная функция для интерактивной загрузки XLSX файлов"""
    print("🔄 Загрузчик XLSX файлов в ChromaDB с чанкованием")
    print("=" * 60)
    
    # Запрашиваем путь к файлу
    file_path = input("Введите путь к XLSX файлу: ").strip()
    
    if not file_path:
        print("❌ Путь к файлу не указан")
        return
    
    # Настройка чанкования
    print("\n⚙️ Настройка чанкования:")
    chunk_size = input("Размер чанка (по умолчанию 1000): ").strip()
    chunk_size = int(chunk_size) if chunk_size.isdigit() else 1000
    
    chunk_overlap = input("Перекрытие чанков (по умолчанию 200): ").strip()
    chunk_overlap = int(chunk_overlap) if chunk_overlap.isdigit() else 200
    
    enable_chunking = input("Включить чанкование? (y/n, по умолчанию y): ").strip().lower()
    enable_chunking = enable_chunking != 'n'
    
    loader = XLSXLoader(
        vector_store=VECTOR_STORE,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    try:
        # Предварительный просмотр
        print("\n📋 Анализ структуры файла...")
        structure = loader.preview_xlsx_structure(file_path)
        
        print(f"\nФайл: {structure['file_path']}")
        print(f"Количество листов: {structure['sheets_count']}")
        
        for sheet_name, info in structure['sheets'].items():
            print(f"\nЛист '{sheet_name}':")
            print(f"  - Строк: {info['rows_count']}")
            print(f"  - Колонок: {info['columns_count']}")
            print(f"  - Колонки: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")
        
        # Выбор режима обработки
        print("\n🔧 Выберите режим обработки:")
        print("1. rows - каждая строка как отдельный документ")
        print("2. sheets - каждый лист как отдельный документ")
        print("3. columns - каждая колонка как отдельный документ")
        print("4. cells - каждая ячейка как отдельный документ")
        
        mode_choice = input("\nВведите номер (1-4) Default: 1: ").strip()
        mode_map = {"1": "rows", "2": "sheets", "3": "columns", "4": "cells"}

        if mode_choice not in mode_map:
            mode_choice = "1"
        processing_mode = mode_map[mode_choice]
        
        # Загрузка
        print(f"\n🚀 Загружаем файл в режиме '{processing_mode}'...")
        print(f"🔪 Параметры чанкования: размер={chunk_size}, перекрытие={chunk_overlap}")
        
        result = loader.add_to_vector_store(file_path, processing_mode, enable_chunking)
        
        if result["success"]:
            print("\n✅ Успешно загружено!")
            print(f"📊 Итоговых документов: {result['documents_count']}")
            
            if result.get('chunking_enabled'):
                print(f"📄 Исходных документов: {result.get('original_docs_count', 'N/A')}")
                print(f"🔪 Чанков создано: {result.get('chunks_count', 'N/A')}")
                print(f"⚙️ Размер чанка: {result.get('chunk_size')}")
                print(f"🔗 Перекрытие: {result.get('chunk_overlap')}")
            
            print(f"🔗 Файл: {result['file_path']}")
            print(f"🎯 Режим: {result['processing_mode']}")
        else:
            print(f"❌ Ошибка: {result['error']}")
            
    except Exception as e:
        print(f"❌ Ошибка: {e}")

def main():
    load_dotenv(find_dotenv())

    if "GIGACHAT_CREDENTIALS" not in os.environ:
        os.environ["GIGACHAT_CREDENTIALS"] = getpass.getpass("Введите ключ авторизации GigaChat API: ")

    # Get all files from doc_files/xls/ directory
    files = glob.glob(os.path.join(os.path.dirname(os.path.dirname(__file__)), "doc_files", "xls", "*"))

    loader = XLSXLoader(
        vector_store=VECTOR_STORE,
        chunk_size=1000,
        chunk_overlap=200
    )

    for file_path in files:
        print("\n📋 Анализ структуры файла...")
        structure = loader.preview_xlsx_structure(file_path)

        for sheet_name, info in structure['sheets'].items():
            print(f"\nЛист '{sheet_name}':")
            print(f"  - Строк: {info['rows_count']}")
            print(f"  - Колонок: {info['columns_count']}")
            print(f"  - Колонки: {', '.join(info['columns'][:5])}{'...' if len(info['columns']) > 5 else ''}")

        processing_mode = "rows"
        result = loader.add_to_vector_store(file_path, processing_mode)

        if result["success"]:
            print("\n✅ Успешно загружено!")
            print(f"📊 Итоговых документов: {result['documents_count']}")
            
            if result.get('chunking_enabled'):
                print(f"📄 Исходных документов: {result.get('original_docs_count', 'N/A')}")
                print(f"🔪 Чанков создано: {result.get('chunks_count', 'N/A')}")
                print(f"⚙️ Размер чанка: {result.get('chunk_size')}")
                print(f"🔗 Перекрытие: {result.get('chunk_overlap')}")
            
            print(f"🔗 Файл: {result['file_path']}")
            print(f"🎯 Режим: {result['processing_mode']}")
        else:
            print(f"❌ Ошибка: {result['error']}")


if __name__ == "__main__":
    main()