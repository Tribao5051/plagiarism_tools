import os

from project_config import load_project_config, resolve_project_path

PROJECT_CONFIG = load_project_config()
MODEL_CONFIG = PROJECT_CONFIG["model"]
DEFAULT_CACHE_DIR = resolve_project_path(MODEL_CONFIG["cache_dir"])
os.environ.setdefault("HF_HOME", DEFAULT_CACHE_DIR)
os.environ.setdefault("HF_HUB_CACHE", os.path.join(DEFAULT_CACHE_DIR, "hub"))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", DEFAULT_CACHE_DIR)

import torch
from FlagEmbedding import BGEM3FlagModel

# 1. Định nghĩa device một lần duy nhất cho toàn bộ module
device = "cuda" if torch.cuda.is_available() else "cpu"


class ModelManager:
    _instance = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ModelManager, cls).__new__(cls)
        return cls._instance

    def get_model(self, model_name=None, cache_dir=None, revision=None):
        if self._model is None:
            model_name = model_name or MODEL_CONFIG["name"]
            cache_dir = cache_dir or DEFAULT_CACHE_DIR
            model_options = {}
            selected_revision = revision or MODEL_CONFIG.get("revision")
            if selected_revision:
                model_options["revision"] = selected_revision
            print(f"--- [STARTUP] Initializing BGE-M3 on {device.upper()} ---")

            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir, exist_ok=True)

            # 2. Truyền device vào cấu hình model
            # BGE-M3 tự động quản lý device, nhưng ta vẫn nên đảm bảo
            self._model = BGEM3FlagModel(
                model_name,
                use_fp16=(
                    device == "cuda"
                    and bool(MODEL_CONFIG.get("use_fp16_on_cuda", True))
                ),
                cache_dir=cache_dir,
                **model_options,
            )

            # Đưa model về đúng device
            if device == "cuda":
                self._model.model.to(device)

            print("--- [STARTUP] Model loaded and ready ---")
        return self._model


# Khởi tạo instance
model_manager = ModelManager()
