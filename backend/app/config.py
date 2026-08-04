"""
Global settings loaded from environment / .env file.
Copy .env.example to .env and fill in the values.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Absolute or relative path to the weights/ directory
    weights_dir: str = "weights"

    # Minimum score returned by /infer (threshold filtering is client-side)
    min_score_floor: float = 0.05

    # Path to the cloned D-FINE repo on the GPU box.
    # The loader does sys.path.insert(0, dfine_repo_path) to import src.core.
    dfine_repo_path: str = "/opt/D-FINE"

    # CORS origins — "*" is fine for single-box Option A deploy
    cors_origins: list[str] = ["*"]

    # Azure Document Intelligence credentials
    azure_document_intelligence_endpoint: str | None = None
    azure_document_intelligence_key: str | None = None

    @property
    def resolved_weights_dir(self) -> Path:
        p = Path(self.weights_dir)
        if not p.is_absolute():
            backend_dir = Path(__file__).resolve().parent.parent
            p = (backend_dir / p).resolve()
        return p

    @property
    def resolved_dfine_repo_path(self) -> Path:
        p = Path(self.dfine_repo_path)
        if not p.is_absolute():
            backend_dir = Path(__file__).resolve().parent.parent
            p = (backend_dir / p).resolve()
        return p


settings = Settings()

