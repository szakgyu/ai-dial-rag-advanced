from enum import StrEnum

import psycopg2
from psycopg2.extras import RealDictCursor

from task.embeddings.embeddings_client import DialEmbeddingsClient
from task.utils.text import chunk_text


class SearchMode(StrEnum):
    EUCLIDIAN_DISTANCE = "euclidean"  # Euclidean distance (<->)
    COSINE_DISTANCE = "cosine"  # Cosine distance (<=>)


class TextProcessor:
    """Processor for text documents that handles chunking, embedding, storing, and retrieval"""

    def __init__(self, embeddings_client: DialEmbeddingsClient, db_config: dict):
        self.embeddings_client = embeddings_client
        self.db_config = db_config

    def _get_connection(self):
        """Get database connection"""
        return psycopg2.connect(
            host=self.db_config['host'],
            port=self.db_config['port'],
            database=self.db_config['database'],
            user=self.db_config['user'],
            password=self.db_config['password']
        )


    def _truncate_table(self):
        """Truncate the vectors table in the database"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                cursor.execute("TRUNCATE TABLE vectors")
                conn.commit()

    def _store_embeddings(self, document_name: str, chunks: list[str], embeddings: dict[int, list[float]]):
        """Store chunks and their corresponding embeddings in the database"""
        with self._get_connection() as conn:
            with conn.cursor() as cursor:
                for index, chunk in enumerate(chunks):
                    embedding_vector = embeddings[index]
                    cursor.execute(
                        "INSERT INTO vectors (document_name, text, embedding) VALUES (%s, %s, %s::vector)",
                        (document_name, chunk, embedding_vector)
                    )
                conn.commit()

    def process_text_file(self, file_path: str, chunk_size: int, overlap: int, dimensions: int, truncate_table: bool = False):
        """Process a text file by chunking, embedding, and storing in the database"""
        if truncate_table:
            self._truncate_table()

        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

        chunks = chunk_text(content, chunk_size, overlap)
        embeddings = self.embeddings_client.get_embeddings(chunks, dimensions)

        self._store_embeddings(file_path, chunks, embeddings)

    def search(self, search_mode: SearchMode, user_request: str, top_k: int, score_threshold: float, dimensions: int) -> list[str]:
        """Search for relevant context based on user request and return matching text chunks"""
        request_embedding = self.embeddings_client.get_embeddings([user_request], dimensions)[0]
        vector_string = f"[{','.join(map(str, request_embedding))}]"

        if search_mode == SearchMode.COSINE_DISTANCE:
            max_distance = 1.0 - score_threshold
        else:
            max_distance = float('inf') if score_threshold == 0 else (1.0 / score_threshold) - 1.0

        retrieved_chunks = []
        with self._get_connection() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(self._get_search_query(search_mode), (vector_string, vector_string, max_distance, top_k))
                results = cursor.fetchall()

                for row in results:
                    if search_mode == SearchMode.COSINE_DISTANCE:
                        similarity = 1.0 - row['distance']
                    else:
                        similarity = 1.0 / (1.0 + row['distance'])

                    print(f"---Similarity score: {similarity:.2f}---")
                    print(f"Data: {row['text']}\n")
                    retrieved_chunks.append(row['text'])

        return retrieved_chunks


    def _get_search_query(self, search_mode: SearchMode) -> str:
        return """SELECT text, embedding {mode} %s::vector AS distance
            FROM vectors
            WHERE embedding {mode} %s::vector <= %s
            ORDER BY distance
            LIMIT %s""".format(mode='<->' if search_mode == SearchMode.EUCLIDIAN_DISTANCE else '<=>')