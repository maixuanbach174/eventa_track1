import json
import numpy as np
import faiss
import os
from typing import Dict, List, Tuple

def build_index_for_articles(passage_table: list, text_embeddings: np.ndarray, save_dir: str = "faiss_indices"):

    os.makedirs(save_dir, exist_ok=True)
    
    article_passages = {}
    
    for i, passage in enumerate(passage_table):
        article_id = passage['article_id']
        if article_id not in article_passages:
            article_passages[article_id] = []
        article_passages[article_id].append({
            'original_index': i,
            'passage_data': passage
        })
    
    print(f"Found {len(article_passages)} articles")
    
    for article_id, passages in article_passages.items():
        print(f"Building index for article {article_id} with {len(passages)} passages...")

        passage_indices = [p['original_index'] for p in passages]
        article_embeddings = text_embeddings[passage_indices]
        passage_list = [p['passage_data'] for p in passages]

        dimension = article_embeddings.shape[1]
        index = faiss.IndexFlatIP(dimension)  

        faiss.normalize_L2(article_embeddings)
        index.add(article_embeddings.astype(np.float32))

        index_path = os.path.join(save_dir, f"article_{article_id}.index")
        faiss.write_index(index, index_path)
        
        # Save passage data (ordered to match FAISS index results)
        passages_path = os.path.join(save_dir, f"article_{article_id}_passages.json")
        with open(passages_path, 'w', encoding='utf-8') as f:
            json.dump({
                'article_id': article_id,
                'passages': passage_list,
                'num_passages': len(passage_list)
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Saved index and {len(passage_list)} passages for article {article_id}")
    
    article_list_path = os.path.join(save_dir, "article_list.json")
    with open(article_list_path, 'w', encoding='utf-8') as f:
        json.dump({
            'articles': list(article_passages.keys()),
            'total_articles': len(article_passages)
        }, f)
    
    print(f"All indices saved to {save_dir}/")
    print(f"Files created:")
    print(f"- article_list.json (list of all articles)")
    for article_id in article_passages.keys():
        print(f"- article_{article_id}.index (FAISS index)")
        print(f"- article_{article_id}_passages.json (passage data)")

def load_article_index(article_id: int, save_dir: str = "faiss_indices") -> Tuple[faiss.Index, List[Dict]]:

    index_path = os.path.join(save_dir, f"article_{article_id}.index")
    passages_path = os.path.join(save_dir, f"article_{article_id}_passages.json")
    
    if not os.path.exists(index_path) or not os.path.exists(passages_path):
        raise FileNotFoundError(f"Index files for article {article_id} not found in {save_dir}")
    
    index = faiss.read_index(index_path)
    
    # Load passage data
    with open(passages_path, 'r', encoding='utf-8') as f:
        passages_data = json.load(f)
    
    return index, passages_data['passages']

def get_available_articles(save_dir: str = "faiss_indices") -> List[int]:
    """Get list of available article IDs."""
    article_list_path = os.path.join(save_dir, "article_list.json")
    
    if not os.path.exists(article_list_path):
        return []
    
    with open(article_list_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data.get('articles', [])

# Example usage:
if __name__ == "__main__":
    # Load your data
    passage_table_path = "saved_index/passage_table.json"
    text_embeddings_path = "saved_index/text_embeddings.npy"
    
    with open(passage_table_path, "r", encoding="utf-8") as f:
        passage_table = json.load(f)
    
    text_embeddings = np.load(text_embeddings_path)
    
    # Build and save all indices with passage data
    build_index_for_articles(passage_table, text_embeddings, save_dir="faiss_indices")
    
    # Example: Load and use index for article 0
    try:
        index, passages = load_article_index(0, "faiss_indices")
        print(f"Loaded index for article 0:")
        print(f"- FAISS index with {index.ntotal} passages")
        print(f"- Stored {len(passages)} passage objects")
        print(f"- First passage text: {passages[0].get('text', '')}...")
        
        # Show available articles
        available = get_available_articles("faiss_indices")
        print(f"Available articles: {available}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")