from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from app.config import settings

engine = create_engine(settings.DATABASE_URL, echo=False, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# Add this helper function
def safe_commit(db_session):
    try:
        db_session.commit()
    except SQLAlchemyError as e:
        db_session.rollback()
        if 'NUL' in str(e) or 'null character' in str(e):
            # Log the error and continue
            print(f"Warning: Skipping commit due to NUL character: {str(e)}")
        else:
            raise
