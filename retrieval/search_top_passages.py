from PIL import Image
from typing import List, Dict
import numpy as np
from retrieval.build_index import load_article_index
import torch
from transformers import CLIPProcessor, CLIPModel
import gc
import pandas as pd
from tqdm import tqdm
import os

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
    image_dir = "dataset/val/pub_images/"
    df = pd.read_csv("submission.csv")
    required_columns = ['query_id', 'article_id_1', 'generated_caption']
    if not all(col in df.columns for col in required_columns):
        print("Error: submission.csv missing required columns")
        exit(1)

    generated_captions = []

    for index, row in tqdm(df.iterrows(), desc="Retrieving passages for caption"):
        query_id = row['query_id']
        article_id_1 = row['article_id_1']
        image_path = f"dataset/val/pub_images/{query_id}.jpg"
        # if(len(generated_captions) == 10):
        #     break
        if not os.path.exists(image_path):
            print(f"Warning: Image not found for query_id {query_id} at {image_path}")
            caption = ""
            generated_captions.append(caption)
            continue
        image_embedding = embed_image(image_path, device, processor, model)
        results = search_passages_in_article(image_embedding, str(article_id_1), top_k=2)
        if len(results) >= 2:
            caption = results[0]['text'] + results[1]['text']
        elif len(results) == 1:
            print(f"Warning: Only one result found for query_id {query_id}, article_id {article_id_1}")
            caption = results[0]['text']
        else:
            print(f"Warning: No results found for query_id {query_id}, article_id {article_id_1}")
            caption = ""
        
        generated_captions.append(caption)
    
    df['generated_caption'] = generated_captions
    # for i in range(10):
    #     df['generated_caption'][i] = generated_captions[i]
    df.to_csv("submission.csv", index=False)
    print("Updated submission.csv successfully")
