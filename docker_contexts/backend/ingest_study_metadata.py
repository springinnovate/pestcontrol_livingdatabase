import pandas as pd

from database import SessionLocal
from database_model_definitions import CovariateDefn, CovariateValue, Study
from database_model_definitions import CovariateAssociation


def ingest_study_metadata():
    table = pd.read_csv('base_data/BioControlDatabaseMetadata.csv')
    session = SessionLocal()
    valid_columns = []
    for column in table.columns:
        covariate_defn = session.query(CovariateDefn).filter(
            CovariateDefn.name == column,
            CovariateDefn.covariate_association == CovariateAssociation.STUDY.value).first()
        if covariate_defn is None:
            print(f'missing {column}')
        else:
            print(f'FOUND {column}')
            valid_columns.append((column, covariate_defn))
    for _, table_row in table.iterrows():
        print(table_row)
        study = session.query(Study).filter(
            Study.name == table_row['Study_ID']).first()
        if study is None:
            print(f'cannot find {table_row["Study_ID"]}')
            continue
        for column, covariate_defn in valid_columns:
            value = table_row[column]
            if pd.isna(value):
                continue

            covariate_value = session.query(CovariateValue).join(Study).filter(
                CovariateValue.covariate_defn == covariate_defn,
                Study.name == table_row['Study_ID']
            ).first()

            if covariate_value is not None:
                # chatgpt -- this is where i would ottherwise update covariate_value
                print(f'{column}: {table_row["Study_ID"]}: {covariate_value}')
                covariate_value.value = table_row[column]
            else:
                new_covariate = CovariateValue(
                    value=table_row[column],
                    covariate_defn=covariate_defn,
                    study=study)
                session.add(new_covariate)
    session.commit()
    session.close()


if __name__ == "__main__":
    print('update order')
    ingest_study_metadata()
