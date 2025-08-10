from setuptools import setup, find_packages

setup(
    name="app",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "pydantic",
        "uvicorn[standard]",
        "python-dotenv",
        "requests",
        "pinecone-client",
        "python-docx",
        "PyMuPDF",
        "pydantic",
        "pydantic-settings",
        "pytest",
        "groq",
        "sqlalchemy",
        "psycopg2-binary",
        "alembic",
        "tiktoken"
    ],
)
