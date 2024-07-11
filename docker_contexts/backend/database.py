"""Database definitions for news articles and their classifications."""
from database_model_definitions import Base, CovariateDefn, CovariateType, RequiredState
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
    REQUIRED_COVARIATES = [
        ('response_type', CovariateType.STRING),
        ('species', CovariateType.STRING),
        ('functional_type', CovariateType.STRING),
        ('crop_name', CovariateType.STRING),
        ('sampling_method', CovariateType.STRING),
        ('sampling_effort', CovariateType.INTEGER),
        ('year', CovariateType.INTEGER),
        ]
    covariates_to_add = []
    for covariate_name, covariate_type in REQUIRED_COVARIATES:
        covariates_to_add.append(
            CovariateDefn(
                name=covariate_name,
                required=RequiredState.REQUIRED,
                covariate_type=covariate_type,
                condition=None,
            ))

    OTHER_COVARIATES = [
        ('TraitGenSpec', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('SiteID', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('SiteDesc', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('AnnualPerennial', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('Organic', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('Tilling', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('LocalDiversity', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('InsecticidePlot', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('InsecticideFarm', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('ConfidenceInsecticide', CovariateType.FLOAT, RequiredState.OPTIONAL, None),
        ('CropType', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('PollinatorDepend', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('MeasureType', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('RawAbundace', CovariateType.FLOAT, RequiredState.OPTIONAL, None),
        ('AbundanceDuration', CovariateType.FLOAT, RequiredState.OPTIONAL, None),
        ('Notes', CovariateType.STRING, RequiredState.OPTIONAL, None),
        ('pest_class', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_order', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_family', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_species', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_sub_species', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_life_stage', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_class', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_order', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_family', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_species', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_sub_species', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_morphospecies', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_lifestage', CovariateType.STRING, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ]

    for covariate_name, covariate_type, required_state, condition in OTHER_COVARIATES:
        covariates_to_add.append(
            CovariateDefn(
                name=covariate_name,
                required=required_state,
                covariate_type=covariate_type,
                condition=condition
            ))

    for covariate in covariates_to_add:
        print(f'adding covariate {covariate}')
        existing = session.query(CovariateDefn).filter_by(
            name=covariate.name).first()
        if not existing:
            session.add(covariate)

    session.commit()
