"""Database definitions for news articles and their classifications."""
import os
import datetime
import shutil
import sqlite3

from database_model_definitions import Base, CovariateDefn, CovariateType, CovariateAssociation, STUDY_ID
from sqlalchemy import create_engine
from sqlalchemy import text
from sqlalchemy.orm import sessionmaker

if os.path.exists('/usr/local/data/live_database/living_database.db'):
    DATABASE_URI = 'sqlite:////usr/local/data/live_database/living_database.db'
else:
    DATABASE_URI = 'sqlite:///live_database/living_database.db'

#DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:password@db:5432/mydatabase')

engine = create_engine(DATABASE_URI, echo=False)

SessionLocal = sessionmaker(bind=engine)


def is_valid_sqlite_db(db_path):
    try:
        # Try to connect to the database
        conn = sqlite3.connect(db_path)
        conn.close()
        return True
    except sqlite3.Error as e:
        print(f'some kind of error on {db_path}: {e}')
        return False


def backup_db():
    db_path = DATABASE_URI.split('///')[-1]
    if not is_valid_sqlite_db(db_path):
        raise RuntimeError(f'{db_path} is invalid database somehow.')
    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    backup_db_path = f'{os.path.splitext(db_path)[0]}_backup_{timestamp}{os.path.splitext(db_path)[1]}'
    shutil.copy(db_path, backup_db_path)
    if not is_valid_sqlite_db(backup_db_path):
        raise RuntimeError(f'{backup_db_path} backup is invalid database somehow.')


def init_db():
    Base.metadata.create_all(bind=engine)
    initialize_covariates()


def initialize_covariates():
    session = SessionLocal()
    print('initalizing covariates')

    # display order, name, editable_name, association, queryable, always display, condition to display)
    # display_order, covariate_name, editable_name, covariate_type, covariate_association, queryable, always_display, condition, hidden, show_in_point_table, search_by_unique
    OTHER_COVARIATES = [
        (-1, 'doi', False, CovariateType.STRING.value, CovariateAssociation.STUDY.value, True, False, False, False, True),
        (1, 'year', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, False, False, True),
        (4, 'Response_variable', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (6, 'Units', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (7, 'Latin_name', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (8, 'Species', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (9, 'Genus', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (10, 'Family', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (11, 'Order', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (12, 'Class', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (13, 'Lifestage', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (14, 'Functional_type', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (15, 'ObserverID', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (16, 'Manager', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (17, 'Crop_latin_name', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (18, 'Crop_common_name', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (19, 'Sampling_method', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (20, 'Month', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (21, 'Day', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (22, 'Time', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
        (23, 'Replicate', False, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, False, False, True),
    ]

    covariates_to_add = []
    for (display_order, covariate_name, editable_name, covariate_type, covariate_association,
         queryable, always_display, hidden, show_in_point_table, search_by_unique) in OTHER_COVARIATES:
        covariates_to_add.append(
            CovariateDefn(
                display_order=display_order,
                name=covariate_name,
                editable_name=editable_name,
                covariate_type=covariate_type,
                covariate_association=covariate_association,
                queryable=queryable,
                always_display=always_display,
                hidden=hidden,
                show_in_point_table=show_in_point_table,
                search_by_unique=search_by_unique,
            ))

    for covariate in covariates_to_add:
        existing = session.query(CovariateDefn).filter(
            CovariateDefn.name.ilike(covariate.name)
        ).first()
        if not existing:
            print(f'adding {covariate.name}')
            session.add(covariate)
            session.commit()

    print('initalizing indexes')
    with engine.connect() as connection:
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariatevalue_covariate_defn_id ON covariate_value (covariate_defn_id);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariatedefn_id_key ON covariate_defn (id_key);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_name ON covariate_defn (name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_editable_name ON covariate_defn (editable_name);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_queryable ON covariate_defn (queryable);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_always_display ON covariate_defn (always_display);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_hidden ON covariate_defn (hidden);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_show_in_point_table ON covariate_defn (show_in_point_table);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_search_by_unique ON covariate_defn (search_by_unique);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_covariate_type ON covariate_defn (covariate_type);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariate_defn_covariate_association ON covariate_defn (covariate_association);"))
        connection.execute(text("CREATE INDEX IF NOT EXISTS ix_covariatevalue_covariate_defn_id_id_key ON covariate_value (covariate_defn_id, id_key);"))

    session.commit()
