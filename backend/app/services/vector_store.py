"""Vector store implementation using FAISS."""

import os
import json
import numpy as np
from typing import List, Optional, Tuple
import faiss

from ..config import config
from ..utils.logger import get_logger

logger = get_logger(__name__)

class VectorStore:
    """FAISS-based vector store for embeddings."""
    
    def __init__(self, dimension: int = 384, index_type: str = "flat"):
        """
        Initialize vector store.
        
        Args:
            dimension: Embedding dimension (384 for all-MiniLM-L6-v2)
            index_type: Type of FAISS index ('flat', 'ivf')
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.id_to_doc = {}
        self.doc_to_id = {}
        self.next_id = 0
        
        self.index_path = os.path.join(config.EMBEDDINGS_DIR, "faiss_index.bin")
        self.metadata_path = os.path.join(config.EMBEDDINGS_DIR, "vector_metadata.json")
        
        self._initialize_index()
        self._load_metadata()
    
    def _initialize_index(self):
        """Initialize FAISS index."""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
        elif self.index_type == "ivf":
            # For larger datasets, use IVF
            nlist = 100  # number of clusters
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        # Load existing index if it exists
        if os.path.exists(self.index_path):
            try:
                self.index = faiss.read_index(self.index_path)
                logger.info(f"Loaded existing FAISS index with {self.index.ntotal} vectors")
            except Exception as e:
                logger.warning(f"Failed to load existing index: {e}, creating new one")
    
    def _load_metadata(self):
        """Load metadata mappings."""
        if os.path.exists(self.metadata_path):
            try:
                with open(self.metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.id_to_doc = {int(k): v for k, v in metadata.get('id_to_doc', {}).items()}
                    self.doc_to_id = metadata.get('doc_to_id', {})
                    self.next_id = metadata.get('next_id', 0)
                logger.info(f"Loaded metadata for {len(self.id_to_doc)} documents")
            except Exception as e:
                logger.warning(f"Failed to load metadata: {e}")
    
    def _save_metadata(self):
        """Save metadata mappings."""
        os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
        metadata = {
            'id_to_doc': self.id_to_doc,
            'doc_to_id': self.doc_to_id,
            'next_id': self.next_id
        }
        
        try:
            with open(self.metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _save_index(self):
        """Save FAISS index to disk."""
        try:
            os.makedirs(config.EMBEDDINGS_DIR, exist_ok=True)
            faiss.write_index(self.index, self.index_path)
            logger.info(f"Saved FAISS index with {self.index.ntotal} vectors")
        except Exception as e:
            logger.error(f"Failed to save index: {e}")

def upsert(ids: List[str], vectors: np.ndarray) -> None:
    """
    Insert or update vectors in the store.
    
    Args:
        ids: List of document IDs
        vectors: Array of embedding vectors
    """
    store = VectorStore()
    
    if len(ids) != vectors.shape[0]:
        raise ValueError("Number of IDs must match number of vectors")
    
    # Normalize vectors for cosine similarity
    faiss.normalize_L2(vectors)
    
    # Convert document IDs to internal IDs
    internal_ids = []
    for doc_id in ids:
        if doc_id in store.doc_to_id:
            # Update existing
            internal_id = store.doc_to_id[doc_id]
        else:
            # Add new
            internal_id = store.next_id
            store.doc_to_id[doc_id] = internal_id
            store.id_to_doc[internal_id] = doc_id
            store.next_id += 1
        
        internal_ids.append(internal_id)
    
    # Add vectors to index
    store.index.add(vectors)
    
    # Save changes
    store._save_index()
    store._save_metadata()
    
    logger.info(f"Upserted {len(ids)} vectors to store")

def search(vec: np.ndarray, top_k: int) -> List[str]:
    """
    Search for similar vectors.
    
    Args:
        vec: Query vector
        top_k: Number of results to return
    
    Returns:
        List of document IDs
    """
    store = VectorStore()
    
    if store.index.ntotal == 0:
        logger.warning("Vector store is empty")
        return []
    
    # Normalize query vector
    vec = vec.reshape(1, -1).astype(np.float32)
    faiss.normalize_L2(vec)
    
    try:
        # Search
        scores, indices = store.index.search(vec, min(top_k, store.index.ntotal))
        
        # Convert internal IDs back to document IDs
        result_ids = []
        for idx in indices[0]:
            if idx != -1 and idx in store.id_to_doc:
                result_ids.append(store.id_to_doc[idx])
        
        logger.info(f"Vector search returned {len(result_ids)} results")
        return result_ids
    
    except Exception as e:
        logger.error(f"Error in vector search: {e}")
        return []