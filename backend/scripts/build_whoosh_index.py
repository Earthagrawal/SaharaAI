#!/usr/bin/env python3
"""Script to build Whoosh index and embeddings from knowledge base."""

import os
import sys
import argparse

# Add app directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.core.rag import build_index
from app.config import config
from app.utils.logger import get_logger

logger = get_logger(__name__)

def main():
    """Build Whoosh index from knowledge base."""
    parser = argparse.ArgumentParser(description='Build search index from knowledge base')
    parser.add_argument('--kb-dir', 
                       default=os.path.join(config.DATA_DIR, 'knowledge_base'),
                       help='Knowledge base directory path')
    parser.add_argument('--index-dir',
                       default=os.path.join(config.DATA_DIR, 'whoosh_index'),
                       help='Output index directory path')
    parser.add_argument('--rebuild', action='store_true',
                       help='Rebuild index even if it exists')
    
    args = parser.parse_args()
    
    logger.info(f"Building Whoosh index from {args.kb_dir}")
    logger.info(f"Output directory: {args.index_dir}")
    
    # Check if knowledge base exists
    if not os.path.exists(args.kb_dir):
        logger.error(f"Knowledge base directory not found: {args.kb_dir}")
        return 1
    
    # Check if index already exists
    if os.path.exists(args.index_dir) and not args.rebuild:
        logger.warning(f"Index directory already exists: {args.index_dir}")
        logger.info("Use --rebuild to overwrite existing index")
        return 0
    
    try:
        # Build the index
        build_index(args.kb_dir, args.index_dir)
        logger.info("Index building completed successfully!")
        
        # Show index statistics
        if os.path.exists(args.index_dir):
            index_files = os.listdir(args.index_dir)
            logger.info(f"Created {len(index_files)} index files")
            
            # Check embeddings
            embeddings_dir = config.EMBEDDINGS_DIR
            if os.path.exists(embeddings_dir):
                embedding_files = [f for f in os.listdir(embeddings_dir) if f.endswith(('.npy', '.json'))]
                logger.info(f"Generated {len(embedding_files)} embedding files")
        
        return 0
        
    except Exception as e:
        logger.error(f"Failed to build index: {e}")
        return 1

if __name__ == '__main__':
    sys.exit(main())