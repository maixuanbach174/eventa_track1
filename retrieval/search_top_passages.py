from PIL import Image
from typing import List, Dict
import numpy as np
from retrieval.build_index import load_article_index
import torch
from transformers import CLIPProcessor, CLIPModel
import gc


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def init_clip_model():
    device = get_device()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()
    return processor, model

@torch.no_grad()
def embed_image(image_path: str, device: str, processor: CLIPProcessor, model: CLIPModel) -> np.ndarray:
    
    img = Image.open(image_path).convert("RGB")
    inputs = processor(images=[img], return_tensors="pt")
    # Move inputs to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    emb = model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)

    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        gc.collect()
        
    return emb.cpu().numpy().astype("float32")

def search_passages_in_article(image_embedding: np.ndarray, article_id: str, top_k: int = 3, save_dir: str = "faiss_indices") -> List[Dict]:

    try:
        faiss_index, passages = load_article_index(article_id, save_dir)
        
        scores, indices = faiss_index.search(image_embedding.astype(np.float32), top_k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx != -1 and idx < len(passages):  # Valid result
                passage = passages[idx]
                results.append({
                    'rank': i + 1,
                    'similarity_score': float(score),
                    'passage_id': passage.get('passage_id', idx),
                    'text': passage.get('text', ''),
                    'article_id': passage.get('article_id', article_id),
                })
        
        return results
        
    except FileNotFoundError as e:
        print(f"Error loading article {article_id}: {e}")
        return []

if __name__ == "__main__":
    device = get_device()
    processor, model = init_clip_model()
    image_path = "dataset/database_images/b64507f291c22d96.jpg"
    image_embedding = embed_image(image_path, device, processor, model)
    print(image_embedding.shape)
    print(search_passages_in_article(image_embedding, "30f174bb655d403a", 1))
