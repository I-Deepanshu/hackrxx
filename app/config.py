from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    # Pinecone
    PINECONE_API_KEY: str = Field(..., env="PINECONE_API_KEY")
    PINECONE_ENVIRONMENT: str = Field(default="us-east4-eqdc4a", env="PINECONE_ENVIRONMENT")
    PINECONE_INDEX_NAME: str = Field(default="hackrx-index", env="PINECONE_INDEX_NAME")

    # HackRx team token
    HACKRX_TEAM_TOKEN: str = Field(..., env="HACKRX_TEAM_TOKEN")

    # Model settings
    MAX_CHUNK_TOKENS: int = Field(default=700, env="MAX_CHUNK_TOKENS")
    GROQ_API_KEY: str = Field(..., env="GROQ_API_KEY")
    GROQ_MODEL: str = Field(default="llama3-70b-8192", env="GROQ_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
