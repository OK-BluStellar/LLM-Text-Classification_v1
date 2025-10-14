import gradio as gr
from typing import Any, Tuple
from omegaconf import OmegaConf
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams['font.family'] = 'Noto Sans CJK JP'
from ..core.ollama_client import OllamaClient
from ..core.data_processor import DataProcessor
from ..core.classifier import TextClassifier


class GradioApp:
    def __init__(self, config: Any):
        self.config = config
        
        self.model_name = str(config.model.name)
        self.model_temperature = float(config.model.temperature)
        self.model_max_tokens = int(config.model.max_tokens)
        self.model_top_p = float(config.model.top_p)
        self.default_prompt = str(config.app.default_prompt)
        self.gradio_server_name = str(config.gradio.server_name)
        self.gradio_server_port = int(config.gradio.server_port)
        self.gradio_share = bool(config.gradio.share)
        
        self.ollama_client = OllamaClient(
            host=config.ollama.host,
            model=self.model_name,
            temperature=self.model_temperature,
            max_tokens=self.model_max_tokens,
            top_p=self.model_top_p,
            timeout=config.ollama.timeout
        )
        self.data_processor = DataProcessor()
        self.classifier = TextClassifier(self.ollama_client, self.data_processor)
        self.current_results = []
        
        default_data = OmegaConf.to_container(config.app.default_test_data, resolve=True)
        success, msg = self.data_processor.load_from_default(default_data)
        
    def create_interface(self) -> gr.Blocks:
        with gr.Blocks(
            title="医療文章分類システム",
            theme=gr.themes.Soft(),
            css="""
                .gradio-container {max-width: 100% !important;}
                @media (max-width: 768px) {
                    .gradio-container {padding: 10px !important;}
                }
            """
        ) as demo:
            gr.Markdown("# 医療文章分類システム")
            gr.Markdown("Ollama LLMを使用した医療アンケート文章の分類システム")
            
            with gr.Tabs():
                with gr.Tab("主要エリア"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("### 1. データ入力")
                            file_input = gr.File(
                                label="CSVまたはExcelファイルをアップロード",
                                file_types=[".csv", ".xlsx", ".xls"]
                            )
                            file_status = gr.Textbox(
                                label="ファイル読み込み状態",
                                value="デフォルトテストデータが読み込まれています",
                                interactive=False
                            )
                            
                            gr.Markdown("### 2. プロンプト入力")
                            prompt_input = gr.Textbox(
                                label="分類用プロンプト",
                                value=self.default_prompt,
                                placeholder="例: 手関節に痛みを感じているか?",
                                lines=3
                            )
                            
                            gr.Markdown("### 3. 分類実行")
                            classify_btn = gr.Button("分類実行", variant="primary", size="lg")
                            
                        with gr.Column(scale=1):
                            gr.Markdown("### 4. 処理状況")
                            progress_text = gr.Textbox(
                                label="進捗状況",
                                value="待機中",
                                interactive=False
                            )
                            
                            current_result = gr.Textbox(
                                label="現在の処理内容",
                                value="",
                                lines=5,
                                interactive=False
                            )
                            
                            gr.Markdown("### 5. 出力")
                            download_btn = gr.Button("CSVダウンロード", variant="secondary")
                            download_file = gr.File(label="結果ファイル", visible=False)
                            
                            gr.Markdown("### 6. 分類グラフ")
                            classification_chart = gr.Plot(label="分類結果の円グラフ")
                            
                            gr.Markdown("### 7. 結果表示")
                            results_table = gr.Dataframe(
                                label="分類結果",
                                interactive=False
                            )
                
                with gr.Tab("設定パネル"):
                    gr.Markdown("### モデル設定")
                    
                    with gr.Row():
                        model_name = gr.Textbox(
                            label="モデル名",
                            value=self.model_name,
                            placeholder="例: llama3.2:3b"
                        )
                        temperature = gr.Slider(
                            label="Temperature",
                            minimum=0.0,
                            maximum=1.0,
                            value=self.model_temperature,
                            step=0.1
                        )
                    
                    with gr.Row():
                        max_tokens = gr.Slider(
                            label="Max Tokens",
                            minimum=100,
                            maximum=2048,
                            value=self.model_max_tokens,
                            step=100
                        )
                        top_p = gr.Slider(
                            label="Top P",
                            minimum=0.0,
                            maximum=1.0,
                            value=self.model_top_p,
                            step=0.05
                        )
                    
                    update_config_btn = gr.Button("設定を更新", variant="primary")
                    config_status = gr.Textbox(
                        label="設定更新状態",
                        value="",
                        interactive=False
                    )
                    
                    gr.Markdown("### ダウンロード済みモデル")
                    
                    with gr.Row():
                        refresh_models_btn = gr.Button("モデルリストを更新", variant="secondary", size="sm")
                    
                    installed_models = gr.Dataframe(
                        label="インストール済みモデル",
                        headers=["モデル名", "サイズ", "更新日時"],
                        interactive=False,
                        wrap=True
                    )
                    
                    gr.Markdown("### モデルダウンロード")
                    gr.Markdown("推奨モデル: `gemma2:2b` (1.6GB), `llama3.2:3b` (2GB), `qwen2.5:3b` (2.3GB)")
                    
                    with gr.Row():
                        download_model_name = gr.Textbox(
                            label="ダウンロードするモデル名",
                            value=self.model_name,
                            placeholder="例: llama3.2:3b"
                        )
                        download_model_btn = gr.Button("モデルをダウンロード", variant="secondary")
                    
                    download_status = gr.Textbox(
                        label="ダウンロード状態",
                        value="",
                        interactive=False
                    )
                    
                    gr.Markdown("### システム状態")
                    connection_status = gr.Textbox(
                        label="Ollama接続状態",
                        value="未確認",
                        interactive=False
                    )
                    check_connection_btn = gr.Button("接続確認")
            
            file_input.upload(
                fn=self._handle_file_upload,
                inputs=[file_input],
                outputs=[file_status, results_table]
            )
            
            classify_btn.click(
                fn=self._handle_classification,
                inputs=[prompt_input],
                outputs=[progress_text, current_result, classification_chart, results_table]
            )
            
            download_btn.click(
                fn=self._handle_download,
                inputs=[],
                outputs=[download_file]
            )
            
            update_config_btn.click(
                fn=self._handle_config_update,
                inputs=[model_name, temperature, max_tokens, top_p],
                outputs=[config_status]
            )
            
            check_connection_btn.click(
                fn=self._check_connection,
                inputs=[],
                outputs=[connection_status]
            )
            
            download_model_btn.click(
                fn=self._handle_model_download,
                inputs=[download_model_name],
                outputs=[download_status, installed_models]
            )
            
            refresh_models_btn.click(
                fn=self._get_installed_models,
                inputs=[],
                outputs=[installed_models]
            )
            
            demo.load(
                fn=self._get_installed_models,
                inputs=[],
                outputs=[installed_models]
            )
            
        return demo
    
    def _handle_file_upload(self, file_obj) -> Tuple[str, Any]:
        success, msg = self.data_processor.load_from_file(file_obj)
        
        if success:
            df = self.data_processor.get_data()
            return msg, df
        else:
            return msg, None
    
    def _create_pie_chart(self, results: list) -> Any:
        try:
            if not results:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.pie([100], labels=['未分類'], colors=['#cccccc'], autopct='%1.1f%%', startangle=90)
                ax.set_title('分類結果（未処理）', fontsize=14, pad=20)
                plt.tight_layout()
                return fig
            
            classification_counts = {}
            for result in results:
                llm_response = result.get('LLM応答', 'エラー')
                classification = result.get('分類結果', -1)
                
                if classification == -1:
                    label = 'エラー'
                elif classification == 1:
                    label = 'はい'
                elif classification == 0:
                    label = 'いいえ'
                else:
                    label = str(classification)
                
                classification_counts[label] = classification_counts.get(label, 0) + 1
            
            labels = list(classification_counts.keys())
            sizes = list(classification_counts.values())
            colors = ['#2ecc71', '#e74c3c', '#95a5a6'][:len(labels)]
            
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax.set_title(f'分類結果（全{len(results)}件）', fontsize=14, pad=20)
            plt.tight_layout()
            return fig
        except Exception as e:
            print(f"Chart creation error: {e}")
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f'グラフ生成エラー\n{str(e)}', ha='center', va='center')
            return fig
    
    def _handle_classification(self, prompt: str):
        if not prompt:
            yield "エラー: プロンプトを入力してください", "", self._create_pie_chart([]), None
            return
        
        self.classifier.start()
        self.current_results = []
        
        data = self.data_processor.get_data()
        if data is None or len(data) == 0:
            yield "エラー: データが読み込まれていません", "", self._create_pie_chart([]), None
            return
        
        total = len(data)
        
        for idx, row in data.iterrows():
            text_id = row.get('ID', idx + 1)
            text = row.get('テキスト', row.get('text', ''))
            
            result = self.ollama_client.classify_text(text, prompt)
            
            result_dict = {
                'ID': text_id,
                'テキスト': text,
                '分類結果': result['classification'],
                'LLM応答': result['raw_response']
            }
            self.current_results.append(result_dict)
            
            current = len(self.current_results)
            progress_msg = f"処理中: {current}/{total}"
            
            current_msg = f"ID: {text_id}\nテキスト: {text}\n分類: {result['classification']}\nLLM応答: {result['raw_response']}"
            
            chart = self._create_pie_chart(self.current_results)
            
            import pandas as pd
            results_df = pd.DataFrame(self.current_results)
            
            yield progress_msg, current_msg, chart, results_df
        
        success, csv_content = self.data_processor.save_results(self.current_results)
        
        final_progress = f"完了: {total}/{total}"
        yield final_progress, current_msg, chart, self.data_processor.get_results_dataframe()
    
    def _handle_download(self) -> Any:
        try:
            if self.data_processor.results is None or len(self.data_processor.results) == 0:
                return gr.File(value=None, visible=False)
            
            csv_file = "/tmp/classification_results.csv"
            self.data_processor.results.to_csv(csv_file, index=False, encoding='utf-8-sig')
            
            return gr.File(value=csv_file, visible=True)
        except Exception as e:
            print(f"CSV download error: {e}")
            return gr.File(value=None, visible=False)
    
    def _handle_config_update(self, model: str, temp: float, tokens: int, top_p: float) -> str:
        try:
            self.ollama_client.update_config(
                model=model,
                temperature=temp,
                max_tokens=int(tokens),
                top_p=top_p
            )
            return f"設定が更新されました: モデル={model}, Temperature={temp}, MaxTokens={tokens}, TopP={top_p}"
        except Exception as e:
            return f"設定更新エラー: {str(e)}"
    
    def _check_connection(self) -> str:
        if self.ollama_client.check_connection():
            return "接続成功: Ollamaサーバーに接続できました"
        else:
            return "接続失敗: Ollamaサーバーに接続できません"
    
    def _get_installed_models(self) -> Any:
        result = self.ollama_client.get_installed_models()
        if result['success']:
            data = [[m['name'], m['size'], m['modified']] for m in result['models']]
            return data
        else:
            return []
    
    def _handle_model_download(self, model_name: str) -> Tuple[str, Any]:
        if not model_name:
            return "エラー: モデル名を入力してください", self._get_installed_models()
        
        result = self.ollama_client.pull_model(model_name)
        updated_list = self._get_installed_models()
        return result['message'], updated_list
    
    def launch(self):
        demo = self.create_interface()
        demo.launch(
            server_name=self.gradio_server_name,
            server_port=self.gradio_server_port,
            share=self.gradio_share
        )
