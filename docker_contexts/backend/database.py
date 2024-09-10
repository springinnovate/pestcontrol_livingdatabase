"""Database definitions for news articles and their classifications."""
import os

from database_model_definitions import Base, CovariateDefn, CovariateType, CovariateAssociation, STUDY_ID
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'sqlite:///instance/living_database.db'
#DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:password@db:5432/mydatabase')

engine = create_engine(DATABASE_URI, echo=False)

SessionLocal = sessionmaker(bind=engine)


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
        print(covariate_name)
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
        print(f'adding covariate {covariate}')
        existing = session.query(CovariateDefn).filter_by(
            name=covariate.name).first()
        if not existing:
            print(f'adding {covariate.name}')
            session.add(covariate)
            session.commit()

    session.commit()
