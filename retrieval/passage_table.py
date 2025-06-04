from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing
import spacy
from tqdm import tqdm
import os
from typing import List, Tuple, Dict

def init_spacy():
    """Initialize spaCy model with only the sentencizer component."""
    nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser", "ner", "lemmatizer", "textcat"])
    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    return nlp

def process_article_chunk(chunk_data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    """Process a chunk of articles in parallel.
    
    Args:
        chunk_data: List of tuples (article_id, text)
    Returns:
        List of tuples (article_id, processed_text)
    """
    try:
        nlp = init_spacy()
        results = []
        
        for art_id, text in chunk_data:
            current_sentences = []
            current_length = 0
            
            # Process the text in smaller chunks if it's too long
            doc = nlp(text)
            
            for sent in doc.sents:
                sentence = sent.text.strip()
                if not sentence:
                    continue
                    
                word_count = len(sentence.split())
                
                if current_length + word_count > 120 and current_sentences:
                    # Join current sentences and add to results
                    results.append((art_id, " ".join(current_sentences)))
                    current_sentences = [sentence]
                    current_length = word_count
                else:
                    current_sentences.append(sentence)
                    current_length += word_count
            
            # Add any remaining sentences
            if current_sentences:
                results.append((art_id, " ".join(current_sentences)))
                
        return results
    except Exception as e:
        print(f"Error in process_article_chunk: {str(e)}")
        return []

class PassageTable:
    def __init__(self, database_path: str):
        """Initialize PassageTable with the database file.
        
        Args:
            database_path: Path to the JSON database file
        """
        with open(database_path, "r", encoding="utf-8") as f:
            self.articles_json = json.load(f)
    
    def build_passage_table(self, max_articles: int = None) -> List[Dict]:
        """Build passage table from articles.
        
        Args:
            max_articles: Maximum number of articles to process. If None, process all.
        Returns:
            List of passage dictionaries.
        """
        # Extract articles
        articles = [(art_id, info["content"]) for art_id, info in self.articles_json.items()]
        
        if max_articles is not None:
            articles = articles[:max_articles]
        
        # Set up multiprocessing
        if os.name != 'nt':  # Not Windows
            try:
                multiprocessing.set_start_method('spawn', force=True)
            except RuntimeError:
                pass  # Already set
        
        # Calculate chunks
        num_cores = max(1, multiprocessing.cpu_count() - 1)
        chunk_size = max(1, len(articles) // num_cores)
        article_chunks = [
            articles[i:i + chunk_size] 
            for i in range(0, len(articles), chunk_size)
        ]
        
        print(f"Processing {len(articles)} articles using {num_cores} cores...")
        print(f"Chunk size: {chunk_size}, Number of chunks: {len(article_chunks)}")
        
        passage_table = []
        pid_counter = 0
        
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_cores) as executor:
            # Submit all chunks for processing
            futures = [
                executor.submit(process_article_chunk, chunk)
                for chunk in article_chunks
            ]
            
            # Process results as they complete
            for future in tqdm(futures, total=len(futures), desc="Processing articles"):
                try:
                    chunk_results = future.result()
                    for art_id, text in chunk_results:
                        passage_table.append({
                            "passage_id": f"{art_id}-p{pid_counter}",
                            "article_id": art_id,
                            "text": text
                        })
                        pid_counter += 1
                except Exception as e:
                    print(f"Error processing chunk: {str(e)}")
                    continue
        
        print(f"Built passage pool of size {len(passage_table)}")
        return passage_table
    
    def load_passage_table(self, passage_table_path: str) -> List[Dict]:
        """Load passage table from file.
        
        Args:
            passage_table_path: Path to the JSON passage table file
        Returns:
            List of passage dictionaries
        """
        with open(passage_table_path, "r", encoding="utf-8") as f:
            return json.load(f)
    
    def save_passage_table(self, passage_table: List[Dict], passage_table_path: str):
        """Save passage table to file.
        
        Args:
            passage_table: List of passage dictionaries
            passage_table_path: Path to save the JSON passage table
        """
        os.makedirs(os.path.dirname(passage_table_path), exist_ok=True)
        with open(passage_table_path, "w", encoding="utf-8") as f:
            json.dump(passage_table, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    passage_table = PassageTable("dataset/database.json")
    passages = passage_table.build_passage_table(max_articles=1000)
    passage_table.save_passage_table(passages, "saved_index/passage_table.json")
    print(f"Saved {len(passages)} passages to saved_index/passage_table.json")