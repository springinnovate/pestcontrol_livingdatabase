"""Database definitions for news articles and their classifications."""
import os

from database_model_definitions import Base, CovariateDefn, CovariateType, CovariateAssociation
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URI = 'sqlite:///instance/living_database.db'
#DATABASE_URI = os.getenv('DATABASE_URL', 'postgresql://postgres:password@db:5432/mydatabase')

engine = create_engine(DATABASE_URI, echo=False)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def initialize_covariates():
    session = SessionLocal()
    print('initalizing covariates')

    # display order, name, type, association, queryable, always display, condition to display)
    OTHER_COVARIATES = [
        (0, 'doi', False, CovariateType.STRING.value, CovariateAssociation.STUDY.value, True, False, None, False, False, False),
        (0, 'study_metadata', True, CovariateType.STRING.value, CovariateAssociation.STUDY.value, False, False, None, False, False, False),
        (0, 'study_id', False, CovariateType.STRING.value, CovariateAssociation.STUDY.value, False, True, None, False, True, True),
        (1, 'response_type', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'species', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'functional_type', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'crop_name', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'sampling_method', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'sampling_effort', True, CovariateType.INTEGER.value, CovariateAssociation.SAMPLE.value, False, False, None, False, False, False),
        (1, 'year', False, CovariateType.INTEGER.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, True),
        (1, 'TraitGenSpec', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, False),
        (1, 'SiteID', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, False, False, None, False, False, False),
        (1, 'SiteDesc', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, True, None, False, False, False),
        (1, 'AnnualPerennial', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'Organic', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'Tilling', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'LocalDiversity', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'InsecticidePlot', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'InsecticideFarm', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'ConfidenceInsecticide', True, CovariateType.FLOAT.value, CovariateAssociation.SAMPLE.value, False, False, None, False, False, False),
        (1, 'CropType', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'PollinatorDepend', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, True),
        (1, 'MeasureType', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, False),
        (1, 'RawAbundace', True, CovariateType.FLOAT.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, False),
        (1, 'AbundanceDuration', True, CovariateType.FLOAT.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, False),
        (1, 'Notes', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, None, False, False, False),
        (1, 'pest_class', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'pest_order', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'pest_family', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'pest_species', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'pest_sub_species', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'pest_life_stage', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_class', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_order', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_family', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_species', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_sub_species', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_morphospecies', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        (1, 'enemy_lifestage', True, CovariateType.STRING.value, CovariateAssociation.SAMPLE.value, True, False, {'depends_on': 'response_type', 'value': 'abundance'}, False, False, True),
        ]

    covariates_to_add = []
    for (display_order, covariate_name, editable_name, covariate_type, covariate_association,
         queryable, always_display, condition, hidden, show_in_point_table, search_by_unique) in OTHER_COVARIATES:
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
                condition=condition,
                hidden=hidden,
                show_in_point_table=show_in_point_table,
                search_by_unique=search_by_unique,
            ))

    for covariate in covariates_to_add:
        print(f'adding covariate {covariate}')
        existing = session.query(CovariateDefn).filter_by(
            name=covariate.name).first()
        if not existing:
            session.add(covariate)

    session.commit()
