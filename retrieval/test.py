import json, spacy, faiss, torch, numpy as np, os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from spacy.tokens import Doc
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
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

def process_article_chunk(args):
    nlp, art_chunk = args
    results = []
    # Disable components we don't need for sentence splitting
    with nlp.select_pipes(enable=["sentencizer"]):
        for art_id, text in art_chunk:
            current, length = [], 0
            # Use nlp.pipe for batch processing within each process
            for doc in nlp.pipe([text], batch_size=1):
                for sent in doc.sents:
                    s = sent.text.strip()
                    if not s:
                        continue
                    wcount = len(s.split())
                    if length + wcount > 120 and current:
                        results.append((art_id, " ".join(current)))
                        current, length = [s], wcount
                    else:
                        current.append(s)
                        length += wcount
                if current:
                    results.append((art_id, " ".join(current)))
    return results

@torch.no_grad()
def compute_text_embeddings(texts, processor, model, device, batch_size=32):
    embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Computing embeddings"):
        batch_texts = texts[i:i + batch_size]
        inputs = processor(text=batch_texts,
                         padding=True,
                         truncation=True,
                         max_length=77,
                         return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Compute embeddings
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        # Move to CPU and convert to numpy
        embeddings.append(emb.cpu().numpy())
        
        # Clear GPU memory
        if device == "cuda":
            torch.cuda.empty_cache()
        elif device == "mps":
            gc.collect()
    
    return np.vstack(embeddings).astype("float32")

def build_and_save_index():
    # Initialize CLIP model in the main process
    device = get_device()
    processor, clip_model = init_clip_model()
    print(f"Using device: {device}")
    
    # 2) Load JSON of articles
    with open("dataset/database.json", "r", encoding="utf-8") as f:
        articles_json = json.load(f)

    # 3) Build passage table using parallel processing
    print("Loading spaCy model...")
    nlp = spacy.load("en_core_web_sm")
    # Add sentencizer for faster processing
    if not "sentencizer" in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    
    # Prepare article chunks for parallel processing
    articles = [(art_id, info["content"]) for art_id, info in articles_json.items()]
    num_cores = max(1, multiprocessing.cpu_count() - 1)  # Leave one core free
    chunk_size = len(articles) // num_cores
    if chunk_size == 0:
        chunk_size = 1
    article_chunks = [articles[i:i + chunk_size] for i in range(0, len(articles), chunk_size)]
    
    print(f"Processing articles using {num_cores} cores...")
    passage_table = []
    pid_counter = 0
    
    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = [executor.submit(process_article_chunk, (nlp, chunk)) for chunk in article_chunks]
        
        # Process results as they complete
        for future in tqdm(futures, total=len(futures), desc="Processing articles"):
            chunk_results = future.result()
            for art_id, text in chunk_results:
                passage_table.append({
                    "passage_id": f"{art_id}-p{pid_counter}",
                    "article_id": art_id,
                    "text": text
                })
                pid_counter += 1

    print(f"Built passage pool of size {len(passage_table)}")

    # 4) Zero-shot embed all passages via get_text_features
    # extract first 1000 passages
    # save passage table to json file
    # with open("saved_index/passage_table.json", "w", encoding="utf-8") as f:
    #     json.dump(passage_table, f, ensure_ascii=False, indent=4)

    all_texts = [row["text"] for row in passage_table][:100]
    
    # Compute embeddings with optimized GPU usage
    text_embeddings = compute_text_embeddings(all_texts, processor, clip_model, device)
    faiss.normalize_L2(text_embeddings)

    # 5) Build FAISS IndexFlatIP
    print("Building FAISS index...")
    d = text_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(text_embeddings)

    # Save the index, embeddings, and passage information
    print("Saving index and metadata...")
    os.makedirs("saved_index", exist_ok=True)
    faiss.write_index(index, "saved_index/passage_index.faiss")
    np.save("saved_index/text_embeddings.npy", text_embeddings)
    
    # Save passage IDs and article IDs
    passage_ids = [row["passage_id"] for row in passage_table]
    article_ids = [row["article_id"] for row in passage_table]
    np.save("saved_index/passage_ids.npy", np.array(passage_ids))
    np.save("saved_index/article_ids.npy", np.array(article_ids))
    print("Saved index and embeddings to saved_index/")

def load_index():
    # Load the saved index and IDs
    index = faiss.read_index("saved_index/passage_index.faiss")
    passage_ids = np.load("saved_index/passage_ids.npy").tolist()
    article_ids = np.load("saved_index/article_ids.npy").tolist()
    return index, passage_ids, article_ids

class ImageRetriever:
    def __init__(self):
        self.device = get_device()
        print(f"Using device: {self.device}")
        self.processor, self.model = init_clip_model()
        self.index = None
        self.passage_ids = None
        self.article_ids = None
    
    @torch.no_grad()
    def embed_image(self, image_path: str) -> np.ndarray:
        img = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=[img], return_tensors="pt")
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        emb = self.model.get_image_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
        
        # Clear GPU memory if needed
        if self.device == "cuda":
            torch.cuda.empty_cache()
        elif self.device == "mps":
            gc.collect()
            
        return emb.cpu().numpy().astype("float32")
    
    def load_if_needed(self):
        if self.index is None:
            self.index, self.passage_ids, self.article_ids = load_index()
    
    def retrieve(self, image_path: str, topn=10):
        self.load_if_needed()
        topk = topn * 10
        img_emb = self.embed_image(image_path)
        faiss.normalize_L2(img_emb)
        
        D_scores, I_indices = self.index.search(img_emb, topk)

        retrieved = []
        for rank, p_idx in enumerate(I_indices[0]):
            pid = self.passage_ids[p_idx]
            aid = self.article_ids[p_idx]
            score = float(D_scores[0][rank])
            retrieved.append({"rank": rank, "passage_id": pid, "article_id": aid, "score": score})

        # Collapse to top-N unique articles
        seen = set()
        top_articles = []
        for entry in retrieved:
            a_id = entry["article_id"]
            sc = entry["score"]
            if a_id not in seen:
                seen.add(a_id)
                top_articles.append({"article_id": a_id, "score": sc})
            if len(top_articles) == topn:
                break

        return retrieved, top_articles

if __name__ == "__main__":
    # Set up proper multiprocessing start method for macOS
    if os.name != 'nt':  # Not Windows
        multiprocessing.set_start_method('spawn')
    
    # Check if index exists, if not, build it
    if not os.path.exists("saved_index/passage_index.faiss"):
        print("Building and saving index...")
        build_and_save_index()
    
    # Demo on one of the images
    demo_img_id = "291f6e8211d92889"
    demo_img_path = f"dataset/database_images/{demo_img_id}.jpg"

    retriever = ImageRetriever()
    retrieved_passages, top_articles = retriever.retrieve(demo_img_path, topn=10)
    
    print("Zero-Shot Retrieved Passages:")
    for x in retrieved_passages:
        print(x)

    print("\nZero-Shot Top 10 Articles:")
    for a in top_articles:
        print(a)
