from sqlalchemy import create_engine
from database_model_definitions import Base
from database import DATABASE_URI

engine = create_engine(DATABASE_URI)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)

print("Database schema dropped and created successfully.")
