from database_model_definitions import Base
from sqlalchemy import create_engine
import database

engine = create_engine(database.DATABASE_URI)
Base.metadata.drop_all(engine)
Base.metadata.create_all(engine)
database.initialize_covariates()

print("Database schema dropped and created successfully.")
