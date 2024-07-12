"""Database definitions for news articles and their classifications."""
from database_model_definitions import Base, CovariateDefn, CovariateType, CovariateAssociation, RequiredState
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
        ('response_type', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('species', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('functional_type', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('crop_name', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('sampling_method', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('sampling_effort', CovariateType.INTEGER, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('year', CovariateType.INTEGER, CovariateAssociation.SAMPLE, RequiredState.REQUIRED, None),
        ('TraitGenSpec', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('SiteID', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('SiteDesc', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('AnnualPerennial', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('Organic', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('Tilling', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('LocalDiversity', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('InsecticidePlot', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('InsecticideFarm', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('ConfidenceInsecticide', CovariateType.FLOAT, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('CropType', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('PollinatorDepend', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('MeasureType', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('RawAbundace', CovariateType.FLOAT, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('AbundanceDuration', CovariateType.FLOAT, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('Notes', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.OPTIONAL, None),
        ('pest_class', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_order', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_family', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_species', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_sub_species', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('pest_life_stage', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_class', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_order', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_family', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_species', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_sub_species', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_morphospecies', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ('enemy_lifestage', CovariateType.STRING, CovariateAssociation.SAMPLE, RequiredState.CONDITIONAL, {'depends_on': 'response_type', 'value': 'abundance'}),
        ]

    covariates_to_add = []
    for covariate_name, covariate_type, covariate_association, required_state, condition in OTHER_COVARIATES:
        covariates_to_add.append(
            CovariateDefn(
                name=covariate_name,
                required=required_state,
                covariate_type=covariate_type,
                covariate_association=covariate_association,
                condition=condition
            ))

    for covariate in covariates_to_add:
        print(f'adding covariate {covariate}')
        existing = session.query(CovariateDefn).filter_by(
            name=covariate.name).first()
        if not existing:
            session.add(covariate)

    session.commit()
