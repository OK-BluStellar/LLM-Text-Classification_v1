from typing import List, Dict, Any, Callable
from .ollama_client import OllamaClient
from .data_processor import DataProcessor


class TextClassifier:
    def __init__(self, ollama_client: OllamaClient, data_processor: DataProcessor):
        self.ollama_client = ollama_client
        self.data_processor = data_processor
        self.is_running = False
        
    def classify_batch(self, prompt: str, progress_callback: Callable = None) -> List[Dict[str, Any]]:
        texts = self.data_processor.get_texts()
        ids = self.data_processor.get_ids()
        
        if not texts:
            return []
        
        results = []
        total = len(texts)
        
        for idx, (text_id, text) in enumerate(zip(ids, texts)):
            if not self.is_running:
                break
                
            result = self.ollama_client.classify_text(text, prompt)
            
            result_entry = {
                'ID': text_id,
                'テキスト': text,
                '分類結果': result['classification'],
                'LLM応答': result['raw_response']
            }
            
            results.append(result_entry)
            
            if progress_callback:
                progress_callback(idx + 1, total, result_entry)
        
        return results
    
    def start(self):
        self.is_running = True
    
    def stop(self):
        self.is_running = False
