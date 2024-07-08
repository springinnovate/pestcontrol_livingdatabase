"""Database definitions for news articles and their classifications."""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_model_definitions import Base

DATABASE_URI = 'sqlite:///instance/living_database.db'
engine = create_engine(DATABASE_URI, echo=False)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)
