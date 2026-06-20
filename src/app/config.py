
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
import os
from dotenv import load_dotenv
load_dotenv()

CONFIG_FILE = Path(__file__).resolve()
APP_DIR = CONFIG_FILE.parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(exist_ok=True)

# Pydantic v2 configuration — load env vars from <PROJECT_ROOT>/.env
_settings_config = SettingsConfigDict(
    env_file=str(PROJECT_ROOT / ".env"),
    env_file_encoding="utf-8",
    extra="ignore",
)

class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # HuggingFace / LLM
    # ------------------------------------------------------------------
    model_config = _settings_config

    hf_token: str | None = Field(default=os.getenv("HF_TOKEN"), alias="HF_TOKEN")
    hf_write: str | None = Field(default=os.getenv("HF_TOKEN"), alias="HF_WRITE")
    huggingface_hub_api_token: str | None = Field(
        default=os.getenv("HF_TOKEN"), alias="HUGGINGFACEHUB_API_TOKEN"
    )

    # --- fallback to hf_token so callers can use token() consistently ---
    def token(self) -> str | None:
        return self.hf_token or self.huggingface_hub_api_token

    # ------------------------------------------------------------------
    # Model identifiers
    # ------------------------------------------------------------------
    specialist_model: str = Field(
        default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
        alias="SPECIALIST_MODEL",
    )
    generalist_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        alias="GENERALIST_MODEL",
    )
    router_model: str = Field(
        default="meta-llama/Llama-3.2-1B-Instruct",
        alias="ROUTER_MODEL",
    )

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------
    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_device: str = Field(default="cpu", alias="RERANKER_DEVICE")

    # ------------------------------------------------------------------
    # Query transformation (HyDE / rewriting)
    # ------------------------------------------------------------------
    query_transform_enabled: bool = Field(default=True, alias="QUERY_TRANSFORM_ENABLED")
    query_transform_model: str | None = Field(default=None, alias="QUERY_TRANSFORM_MODEL")

    # ------------------------------------------------------------------
    # Retrieval knobs
    # ------------------------------------------------------------------
    hybrid_candidate_multiplier: int = Field(default=4, alias="HYBRID_CANDIDATE_MULTIPLIER")

    # ------------------------------------------------------------------
    # Auth / JWT
    # ------------------------------------------------------------------
    jwt_secret_key: str = Field(
        default="change-me-in-production-use-a-real-secret", alias="JWT_SECRET_KEY"
    )
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")

    # ------------------------------------------------------------------
    # Database & storage
    # ------------------------------------------------------------------
    db_dir: Path = Field(
        default=PROJECT_ROOT / "data", alias="DB_DIR"
    )
    database_url: str = Field(default="", alias="DATABASE_URL")
    
    @model_validator(mode='after')
    def set_database_url(self):
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{self.db_dir}/app.db"
        return self

    # ------------------------------------------------------------------
    # Corpus / vector DB
    # ------------------------------------------------------------------
    local_data_path: Path = Field(default=PROJECT_ROOT / ".langchain_chroma", alias="LOCAL_DATA_PATH")
    chroma_collection_name: str = Field(default="big_token_corpus", alias="CHROMA_COLLECTION_NAME")

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_dir: Path = Field(default=PROJECT_ROOT/ "logs", alias="LOG_DIR")


# Global singleton — import this from anywhere in the app.
settings = Settings()


if __name__ == "__main__":
    # Quick sanity check when run directly.
    import json

    print(json.dumps(settings.model_dump(), indent=2, default=str))
