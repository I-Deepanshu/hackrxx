# hackrx-pinecone-groq-full
FastAPI scaffold with tiktoken chunking, PostgreSQL persistence, Alembic migrations, Groq LLM, and CI.

## Quickstart (local)
my name is deepanshu
1. Copy `.env.example` to `.env` and set values (GROQ_API_KEY required if you want LLM calls).
2. Create a Python virtualenv and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   ```
3. Start a local Postgres (e.g., with Docker):
   ```bash
   docker run --name hackrx-postgres -e POSTGRES_USER=user -e POSTGRES_PASSWORD=pass -e POSTGRES_DB=hackrx -p 5432:5432 -d postgres:14
   ```
4. Run migrations (alembic):
   ```bash
   alembic upgrade head
   ```
5. Run the app:
   ```bash
   uvicorn app.main:app --reload --port 8000
   ```
## Deploy on Railway
- Push repo to GitHub, create Railway project, link repo, add PostgreSQL plugin and set env vars.
