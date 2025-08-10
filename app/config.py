from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    DATABASE_URL: str
    PINECONE_API_KEY: str
    PINECONE_ENVIRONMENT: str
    PINECONE_INDEX_NAME: str
    HACKRX_TEAM_TOKEN: str
    MAX_CHUNK_TOKENS: int
    GROQ_API_KEY: str
    GROQ_MODEL: str

    class Config:
        env_file = ".env"

settings = Settings()
