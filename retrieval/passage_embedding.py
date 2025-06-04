import gc
import json
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

def get_device():
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"

def init_clip_model():
    device = get_device()
    print(f"Using device: {device}")
    
    processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    return processor, model

def load_passage_table(passage_table_path: str) -> List[Dict]:
    with open(passage_table_path, "r", encoding="utf-8") as f:
        return json.load(f)

@torch.no_grad()
def compute_text_embeddings(texts, processor, model, device, batch_size=32):
    embeddings = []
    
    # Ensure model is on the correct device
    model = model.to(device)
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        
        # Process text inputs
        inputs = processor(text=batch_texts,
                         padding=True,
                         truncation=True,
                         max_length=77,
                         return_tensors="pt")
        
        try:
            # Move inputs to device
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Compute embeddings
            emb = model.get_text_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            
            # Move to CPU and collect
            embeddings.append(emb.cpu().numpy())
            
        except RuntimeError as e:
            print(f"Error processing batch {i}: {str(e)}")
            # If MPS error occurs, try processing on CPU
            if device == "mps":
                print("Falling back to CPU for this batch...")
                model = model.cpu()
                inputs = {k: v.cpu() for k, v in inputs.items()}
                emb = model.get_text_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.numpy())
                model = model.to(device)  # Move back to MPS
        
        # Clear memory
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            gc.collect()
            torch.mps.empty_cache()
    
    return np.vstack(embeddings).astype("float32")

def save_text_embeddings(text_embeddings: np.ndarray, text_embeddings_path: str):
    np.save(text_embeddings_path, text_embeddings)

def load_text_embeddings(text_embeddings_path: str) -> np.ndarray:
    return np.load(text_embeddings_path)

@torch.no_grad()
def verify_embeddings(text_embeddings: np.ndarray, text_sample: str, processor, model, device) -> bool:
    """Verify embeddings by computing cosine similarity between saved and new embeddings."""
    # Compute new embedding for the sample text
    inputs = processor(text=text_sample,
                      padding=True,
                      truncation=True,
                      max_length=77,
                      return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    new_emb = model.get_text_features(**inputs)
    new_emb = new_emb / new_emb.norm(dim=-1, keepdim=True)
    new_emb = new_emb.cpu().numpy()
    
    # Compute cosine similarity
    similarity = np.dot(text_embeddings[99], new_emb.flatten())
    print(f"Cosine similarity with first embedding: {similarity:.4f}")
    
    # Check shapes
    print(f"Saved embedding shape: {text_embeddings[0].shape}")
    print(f"New embedding shape: {new_emb.shape}")
    
    # High similarity threshold (they should be very similar but might not be exactly equal)
    return similarity > 0.99

if __name__ == "__main__":
    # Initialize model and device
    processor, clip_model = init_clip_model()
    device = get_device()
    print(f"Device: {device}")
    
    # Load and process passages
    passage_table = load_passage_table("saved_index/passage_table.json")
    print(f"Loaded {len(passage_table)} passages")
    
    # Process first 100 passages
    all_texts = [row["text"] for row in passage_table][:100]
    print(f"Processing {len(all_texts)} texts")
    
    # Compute embeddings
    text_embeddings = compute_text_embeddings(all_texts, processor, clip_model, device)
    print(f"Computed embeddings shape: {text_embeddings.shape}")
    
    # Save embeddings
    save_text_embeddings(text_embeddings, "saved_index/text_embeddings.npy")
    print("Saved embeddings to saved_index/text_embeddings.npy")
    
    # Load saved embeddings
    loaded_embeddings = load_text_embeddings("saved_index/text_embeddings.npy")
    print(f"Loaded embeddings shape: {loaded_embeddings.shape}")