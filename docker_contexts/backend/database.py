"""Database definitions for news articles and their classifications."""
from database_model_definitions import Base, CovariateDefn, CovariateType, CovariateAssociation
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker


DATABASE_URI = 'sqlite:///instance/living_database.db'
engine = create_engine(DATABASE_URI, echo=False)

SessionLocal = sessionmaker(bind=engine)


def init_db():
    Base.metadata.create_all(bind=engine)


def initialize_covariates():
    session = SessionLocal()
    print('initalizing covariates')

    OTHER_COVARIATES = [
        (0, 'doi', CovariateType.STRING, CovariateAssociation.STUDY, True, True, None),
        (0, 'study_metadata', CovariateType.STRING, CovariateAssociation.STUDY, True, True, None),
        (1, 'response_type', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'species', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'functional_type', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'crop_name', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'sampling_method', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'sampling_effort', CovariateType.INTEGER, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'year', CovariateType.INTEGER, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'TraitGenSpec', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'SiteID', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'SiteDesc', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'AnnualPerennial', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'Organic', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'Tilling', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'LocalDiversity', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'InsecticidePlot', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'InsecticideFarm', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'ConfidenceInsecticide', CovariateType.FLOAT, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'CropType', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'PollinatorDepend', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'MeasureType', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'RawAbundace', CovariateType.FLOAT, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'AbundanceDuration', CovariateType.FLOAT, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'Notes', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, None),
        (1, 'pest_class', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'pest_order', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'pest_family', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'pest_species', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'pest_sub_species', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'pest_life_stage', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_class', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_order', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_family', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_species', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_sub_species', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_morphospecies', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        (1, 'enemy_lifestage', CovariateType.STRING, CovariateAssociation.SAMPLE, True, True, {'depends_on': 'response_type', 'value': 'abundance'}),
        ]

    covariates_to_add = []
    for (display_order, covariate_name, covariate_type, covariate_association,
         queryable, always_display, condition) in OTHER_COVARIATES:
        covariates_to_add.append(
            CovariateDefn(
                display_order=display_order,
                name=covariate_name,
                covariate_type=covariate_type,
                covariate_association=covariate_association,
                queryable=queryable,
                always_display=always_display,
                condition=condition
            ))

    for covariate in covariates_to_add:
        print(f'adding covariate {covariate}')
        existing = session.query(CovariateDefn).filter_by(
            name=covariate.name).first()
        if not existing:
            session.add(covariate)

    session.commit()
