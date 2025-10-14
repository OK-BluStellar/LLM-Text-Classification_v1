import pandas as pd
from typing import List, Dict, Any, Tuple
import io


class DataProcessor:
    def __init__(self):
        self.data = None
        self.results = None
        
    def load_from_file(self, file_obj) -> Tuple[bool, str]:
        try:
            if file_obj is None:
                return False, "ファイルが選択されていません"
            
            file_name = file_obj.name if hasattr(file_obj, 'name') else 'file'
            
            if file_name.endswith('.csv'):
                self.data = pd.read_csv(file_obj)
            elif file_name.endswith(('.xlsx', '.xls')):
                self.data = pd.read_excel(file_obj)
            else:
                return False, "サポートされていないファイル形式です。CSVまたはExcelファイルを選択してください。"
            
            if self.data.empty:
                return False, "ファイルにデータがありません"
            
            if 'ID' not in self.data.columns and 'id' not in self.data.columns:
                self.data.insert(0, 'ID', range(1, len(self.data) + 1))
            
            text_column = None
            for col in ['text', 'Text', 'TEXT', 'テキスト', '文章', 'content', 'Content']:
                if col in self.data.columns:
                    text_column = col
                    break
            
            if text_column is None:
                text_column = self.data.columns[1] if len(self.data.columns) > 1 else self.data.columns[0]
            
            self.data = self.data.rename(columns={text_column: 'text'})
            
            return True, f"データ読み込み成功: {len(self.data)}件"
            
        except Exception as e:
            return False, f"ファイル読み込みエラー: {str(e)}"
    
    def load_from_default(self, default_data: List[Dict]) -> Tuple[bool, str]:
        try:
            self.data = pd.DataFrame(default_data)
            self.data = self.data.rename(columns={'id': 'ID'})
            return True, f"デフォルトデータ読み込み成功: {len(self.data)}件"
        except Exception as e:
            return False, f"デフォルトデータ読み込みエラー: {str(e)}"
    
    def get_data(self) -> pd.DataFrame:
        return self.data
    
    def get_texts(self) -> List[str]:
        if self.data is None:
            return []
        return self.data['text'].tolist()
    
    def get_ids(self) -> List[Any]:
        if self.data is None:
            return []
        id_col = 'ID' if 'ID' in self.data.columns else 'id'
        return self.data[id_col].tolist()
    
    def save_results(self, results: List[Dict]) -> Tuple[bool, str]:
        try:
            self.results = pd.DataFrame(results)
            
            csv_buffer = io.StringIO()
            self.results.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
            csv_content = csv_buffer.getvalue()
            
            return True, csv_content
            
        except Exception as e:
            return False, f"結果保存エラー: {str(e)}"
    
    def get_results_dataframe(self) -> pd.DataFrame:
        return self.results if self.results is not None else pd.DataFrame()
