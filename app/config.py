from pydantic import BaseModel
from dotenv import load_dotenv
import os

load_dotenv()  # loads .env if present

class Settings(BaseModel):
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str | None = os.getenv("OPENAI_MODEL")
    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")
    ncbi_email: str | None = os.getenv("NCBI_EMAIL")
    ncbi_tool: str  | None = os.getenv("NCBI_TOOL", "pubmed-gpt-app")

settings = Settings()
