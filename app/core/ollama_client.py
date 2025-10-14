import ollama
from typing import Dict, Any
import time


class OllamaClient:
    def __init__(self, host: str, model: str, temperature: float = 0.3, 
                 max_tokens: int = 512, top_p: float = 0.9, timeout: int = 300):
        self.host = host
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.timeout = timeout
        self.client = ollama.Client(host=host)
        
    def check_connection(self) -> bool:
        try:
            self.client.list()
            return True
        except Exception as e:
            print(f"Ollama connection error: {e}")
            return False
    
    def get_installed_models(self) -> Dict[str, Any]:
        try:
            models = self.client.list()
            model_list = []
            for m in models.get('models', []):
                model_list.append({
                    'name': m.get('name', ''),
                    'size': self._format_size(m.get('size', 0)),
                    'modified': m.get('modified_at', '')
                })
            return {
                'success': True,
                'models': model_list,
                'count': len(model_list)
            }
        except Exception as e:
            return {
                'success': False,
                'models': [],
                'count': 0,
                'error': str(e)
            }
    
    def _format_size(self, size_bytes: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
    
    def pull_model(self, model_name: str = None) -> Dict[str, Any]:
        try:
            target_model = model_name if model_name else self.model
            print(f"Pulling model: {target_model}")
            self.client.pull(target_model)
            return {
                'success': True,
                'message': f"モデル '{target_model}' のダウンロードが完了しました"
            }
        except Exception as e:
            return {
                'success': False,
                'message': f"モデルのダウンロードに失敗しました: {str(e)}"
            }
    
    def ensure_model(self) -> bool:
        try:
            models = self.client.list()
            model_names = [m['name'] for m in models.get('models', [])]
            
            if self.model not in model_names:
                print(f"Model {self.model} not found. Pulling...")
                self.client.pull(self.model)
                print(f"Model {self.model} pulled successfully")
            
            return True
        except Exception as e:
            print(f"Error ensuring model: {e}")
            return False
    
    def classify_text(self, text: str, prompt: str) -> Dict[str, Any]:
        try:
            full_prompt = f"""以下の文章を読んで、質問に答えてください。

文章: {text}

質問: {prompt}

回答は「はい」または「いいえ」のみで答えてください。それ以外の説明は不要です。"""

            response = self.client.generate(
                model=self.model,
                prompt=full_prompt,
                options={
                    'temperature': self.temperature,
                    'num_predict': self.max_tokens,
                    'top_p': self.top_p,
                }
            )
            
            answer = response['response'].strip()
            
            classification = 1 if 'はい' in answer.lower() or 'yes' in answer.lower() else 0
            
            return {
                'classification': classification,
                'raw_response': answer,
                'success': True
            }
            
        except Exception as e:
            return {
                'classification': -1,
                'raw_response': str(e),
                'success': False
            }
    
    def update_config(self, model: str = None, temperature: float = None, 
                     max_tokens: int = None, top_p: float = None):
        if model:
            self.model = model
        if temperature is not None:
            self.temperature = temperature
        if max_tokens:
            self.max_tokens = max_tokens
        if top_p is not None:
            self.top_p = top_p
