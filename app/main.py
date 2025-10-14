import hydra
from omegaconf import DictConfig, OmegaConf
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ui.gradio_app import GradioApp


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    print("=" * 60)
    print("医療文章分類システム起動中...")
    print("=" * 60)
    print("\n設定内容:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 60)
    
    app = GradioApp(cfg)
    
    print("\nGradio UIを起動しています...")
    print(f"アクセスURL: http://localhost:{cfg.gradio.server_port}")
    print("=" * 60)
    
    app.launch()


if __name__ == "__main__":
    main()
