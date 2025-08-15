"""RAG (Retrieval-Augmented Generation) system with Whoosh and semantic reranking."""

import os
import json
from typing import List, Dict, Any, Optional
from whoosh import fields, analysis
from whoosh.index import create_index, open_dir, exists_in
from whoosh.filedb.filestore import FileStorage
from whoosh.qparser import QueryParser
from whoosh.query import And, Term
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class RAGSystem:
    """RAG system for knowledge base search and retrieval."""
    
    def __init__(self):
        self.index_dir = os.path.join(config.DATA_DIR, "whoosh_index")
        self.embeddings_dir = config.EMBEDDINGS_DIR
        self.model = None
        self._load_sentence_transformer()
    
    def _load_sentence_transformer(self):
        """Load sentence transformer model for semantic similarity."""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load sentence transformer: {e}")
            self.model = None
    
    def _chunk_text(self, text: str, chunk_size: int = 250, overlap: int = 50) -> List[Dict[str, Any]]:
        """
        Chunk text into overlapping segments.
        
        Args:
            text: Input text to chunk
            chunk_size: Target size for each chunk (in words)
            overlap: Number of words to overlap between chunks
        
        Returns:
            List of chunk dictionaries with metadata
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            
            chunks.append({
                "text": chunk_text,
                "start_word": i,
                "end_word": min(i + chunk_size, len(words)),
                "chunk_id": len(chunks)
            })
            
            if i + chunk_size >= len(words):
                break
        
        return chunks

def build_index(kb_dir: str, index_dir: str) -> None:
    """
    Build Whoosh index from knowledge base directory.
    
    Args:
        kb_dir: Directory containing knowledge base files
        index_dir: Directory to store the index
    """
    rag = RAGSystem()
    logger.info(f"Building index from {kb_dir} to {index_dir}")
    
    # Define schema
    schema = fields.Schema(
        id=fields.ID(stored=True, unique=True),
        content=fields.TEXT(analyzer=analysis.StemmingAnalyzer(), stored=True),
        source=fields.TEXT(stored=True),
        chunk_id=fields.NUMERIC(stored=True)
    )
    
    # Create index directory
    os.makedirs(index_dir, exist_ok=True)
    
    # Create or open index
    try:
        storage = FileStorage(index_dir)
        if storage.index_exists():
            ix = open_dir(index_dir)
        else:
            ix = create_index(schema, index_dir)
    except:
        # Fallback: try to open, create if fails
        try:
            ix = open_dir(index_dir)
        except:
            ix = create_index(schema, index_dir)
    
    # Index documents
    writer = ix.writer()
    doc_count = 0
    
    if os.path.exists(kb_dir):
        for root, dirs, files in os.walk(kb_dir):
            for file in files:
                if file.endswith(('.txt', '.md', '.json')):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        # Chunk the document
                        chunks = rag._chunk_text(content)
                        
                        for chunk in chunks:
                            doc_id = f"{file}_{chunk['chunk_id']}"
                            writer.add_document(
                                id=doc_id,
                                content=chunk['text'],
                                source=file,
                                chunk_id=chunk['chunk_id']
                            )
                            doc_count += 1
                    
                    except Exception as e:
                        logger.error(f"Error indexing {file_path}: {e}")
    
    writer.commit()
    logger.info(f"Indexed {doc_count} chunks from knowledge base")
    
    # Generate embeddings for semantic search
    _generate_embeddings(index_dir, rag.model)

def _generate_embeddings(index_dir: str, model) -> None:
    """Generate embeddings for indexed documents."""
    if not model:
        logger.warning("No sentence transformer model available, skipping embeddings")
        return
    
    try:
        ix = open_dir(index_dir)
        embeddings = []
        doc_ids = []
        
        with ix.searcher() as searcher:
            for doc in searcher.all_docs():
                content = doc['content']
                doc_id = doc['id']
                
                # Generate embedding
                embedding = model.encode(content)
                embeddings.append(embedding)
                doc_ids.append(doc_id)
        
        if embeddings:
            embeddings_array = np.array(embeddings)
            embeddings_path = os.path.join(config.EMBEDDINGS_DIR, "doc_embeddings.npy")
            ids_path = os.path.join(config.EMBEDDINGS_DIR, "doc_ids.json")
            
            os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
            np.save(embeddings_path, embeddings_array)
            
            with open(ids_path, 'w') as f:
                json.dump(doc_ids, f)
            
            logger.info(f"Generated embeddings for {len(embeddings)} documents")
    
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")

def keyword_search(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """
    Perform keyword search using Whoosh.
    
    Args:
        query: Search query
        top_k: Number of results to return
    
    Returns:
        List of search results
    """
    rag = RAGSystem()
    index_dir = rag.index_dir
    
    if not os.path.exists(index_dir):
        logger.warning(f"Index directory {index_dir} does not exist")
        return []
    
    try:
        ix = open_dir(index_dir)
        parser = QueryParser("content", ix.schema)
        parsed_query = parser.parse(query)
        
        results = []
        with ix.searcher() as searcher:
            search_results = searcher.search(parsed_query, limit=top_k)
            
            for result in search_results:
                results.append({
                    "id": result['id'],
                    "content": result['content'],
                    "source": result['source'],
                    "score": result.score,
                    "chunk_id": result.get('chunk_id', 0)
                })
        
        logger.info(f"Keyword search returned {len(results)} results for query: {query}")
        return results
    
    except Exception as e:
        logger.error(f"Error in keyword search: {e}")
        return []

def semantic_rerank(query: str, snippets: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """
    Rerank search results using semantic similarity.
    
    Args:
        query: Original search query
        snippets: Results from keyword search
        top_k: Number of top results to return
    
    Returns:
        Reranked results
    """
    rag = RAGSystem()
    
    if not rag.model or not snippets:
        logger.warning("No sentence transformer model or empty snippets, returning original results")
        return snippets[:top_k]
    
    try:
        # Generate query embedding
        query_embedding = rag.model.encode(query)
        
        # Calculate similarities
        similarities = []
        for snippet in snippets:
            content_embedding = rag.model.encode(snippet['content'])
            similarity = np.dot(query_embedding, content_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(content_embedding)
            )
            similarities.append(similarity)
        
        # Sort by similarity
        sorted_indices = np.argsort(similarities)[::-1]
        reranked_results = []
        
        for i in sorted_indices[:top_k]:
            result = snippets[i].copy()
            result['semantic_score'] = float(similarities[i])
            reranked_results.append(result)
        
        logger.info(f"Semantic reranking returned {len(reranked_results)} results")
        return reranked_results
    
    except Exception as e:
        logger.error(f"Error in semantic reranking: {e}")
        return snippets[:top_k]