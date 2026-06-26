from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

CONFIG_FILE = Path(__file__).resolve()
APP_DIR = CONFIG_FILE.parent
PROJECT_ROOT = APP_DIR.parent.parent
DB_DIR = PROJECT_ROOT / "data"
DB_DIR.mkdir(exist_ok=True)



class Settings(BaseSettings):
    model_config = SettingsConfigDict(
    env_file=str(PROJECT_ROOT / ".env"),
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False
    )

    hf_token: str | None = Field(default=None, alias="HF_TOKEN")

    def token(self) -> str | None:
        return self.hf_token

    router_model: str = Field(
        default="meta-llama/Llama-3.1-8B-Instruct",
        alias="ROUTER_MODEL",
    )
    generation_model: str = Field(
        default="google/gemma-4-31B-it",
        alias="GENERATION_MODEL",
    )

    reranker_model: str = Field(
        default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL"
    )
    reranker_enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    reranker_device: str = Field(default="cuda", alias="RERANKER_DEVICE")

    # --- NER enrichment ---
    ner_enabled: bool = Field(default=True, alias="NER_ENABLED")
    ner_device: str = Field(default="cpu", alias="NER_DEVICE")
    ner_model_en: str = Field(default="dslim/bert-base-NER", alias="NER_MODEL_EN")
    ner_model_id: str = Field(default="cahya/bert-base-indonesian-NER", alias="NER_MODEL_ID")
    ner_batch_size: int = Field(default=32, alias="NER_BATCH_SIZE")

    query_transform_enabled: bool = Field(default=True, alias="QUERY_TRANSFORM_ENABLED")

    hybrid_candidate_multiplier: int = Field(
        default=2, alias="HYBRID_CANDIDATE_MULTIPLIER"
    )

    # Minimum cross-encoder relevance for a chunk to reach the LLM prompt.
    # 0.0 disables filtering; 1.0 keeps only top-scoring chunks. Operates on
    # the cross-encoder's normalised score (FlagReranker ``normalize=True``
    # range ≈ [0, 1]).
    rag_min_relevance: float = Field(default=0.5, alias="RAG_MIN_RELEVANCE")

    jwt_secret_key: str = Field(
        default="change-me-in-production-use-a-real-secret", alias="JWT_SECRET_KEY"
    )
    jwt_expire_minutes: int = Field(default=1440, alias="JWT_EXPIRE_MINUTES")

    db_dir: Path = Field(default=PROJECT_ROOT / "data", alias="DB_DIR")
    database_url: str = Field(default="", alias="DATABASE_URL")

    @model_validator(mode="after")
    def set_database_url(self):
        if not self.database_url:
            self.database_url = f"sqlite+aiosqlite:///{self.db_dir}/app.db"
        return self

    local_data_path: Path = Field(
        default=PROJECT_ROOT / ".langchain_chroma", alias="LOCAL_DATA_PATH"
    )
    chroma_collection_name: str = Field(
        default="big_token_corpus", alias="CHROMA_COLLECTION_NAME"
    )

    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")


settings = Settings()

if __name__ == "__main__":
    import json

    print(json.dumps(settings.model_dump(), indent=2, default=str))
