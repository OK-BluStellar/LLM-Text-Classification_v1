import gradio as gr
from typing import Any, Tuple
from omegaconf import OmegaConf
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
                            
                            gr.Markdown("### 6. 結果表示")
                            results_table = gr.Dataframe(
                                label="分類結果",
                                interactive=False,
                                wrap=True
                            )
                
                with gr.Tab("設定パネル"):
                    gr.Markdown("### モデル設定")
                    
                    with gr.Row():
                        model_name = gr.Textbox(
                            label="モデル名",
                            value=self.model_name,
                            placeholder="例: llama3:8b"
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
                outputs=[progress_text, current_result, results_table]
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
            
        return demo
    
    def _handle_file_upload(self, file_obj) -> Tuple[str, Any]:
        success, msg = self.data_processor.load_from_file(file_obj)
        
        if success:
            df = self.data_processor.get_data()
            return msg, df
        else:
            return msg, None
    
    def _handle_classification(self, prompt: str) -> Tuple[str, str, Any]:
        if not prompt:
            return "エラー: プロンプトを入力してください", "", None
        
        self.classifier.start()
        self.current_results = []
        
        def progress_callback(current: int, total: int, result: dict):
            self.current_results.append(result)
        
        results = self.classifier.classify_batch(prompt, progress_callback)
        
        if not results:
            return "エラー: データが読み込まれていません", "", None
        
        total = len(results)
        progress_msg = f"完了: {total}/{total}"
        
        last_result = results[-1] if results else {}
        current_msg = f"ID: {last_result.get('ID', '')}\nテキスト: {last_result.get('テキスト', '')}\n分類: {last_result.get('分類結果', '')}\nLLM応答: {last_result.get('LLM応答', '')}"
        
        success, csv_content = self.data_processor.save_results(results)
        
        return progress_msg, current_msg, self.data_processor.get_results_dataframe()
    
    def _handle_download(self) -> Any:
        if self.data_processor.results is None:
            return None
        
        csv_file = "/tmp/classification_results.csv"
        self.data_processor.results.to_csv(csv_file, index=False, encoding='utf-8-sig')
        
        return csv_file
    
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
    
    def launch(self):
        demo = self.create_interface()
        demo.launch(
            server_name=self.gradio_server_name,
            server_port=self.gradio_server_port,
            share=self.gradio_share
        )
