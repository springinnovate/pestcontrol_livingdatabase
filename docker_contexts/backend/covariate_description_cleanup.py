import pandas as pd
import sqlite3

from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import select, func, or_
from database_model_definitions import CovariateValue, CovariateDefn


def clean_whitespace(session: Session):
    covariate_values_with_spaces = session.execute(
        select(CovariateValue).where(
            func.trim(CovariateValue.value) != CovariateValue.value  # Finds rows where leading/trailing spaces exist
        )
    ).scalars().all()

    for cov_value in covariate_values_with_spaces:
        cleaned_value = cov_value.value.strip()
        if cleaned_value != cov_value.value:
            print(f"Fixing '{cov_value.value}' to '{cleaned_value}'")
            cov_value.value = cleaned_value

    print(f"Removed leading/trailing spaces from {len(covariate_values_with_spaces)} covariate values.")


def fix_values(session: Session, replacements: list[tuple[str, str]]):
    for original_value, fixed_value in replacements:
        covariate_values_to_fix = session.execute(
            select(CovariateValue).where(CovariateValue.value == original_value)
        ).scalars().all()

        for cov_value in covariate_values_to_fix:
            print(f"Changing '{original_value}' to '{fixed_value}'")
            cov_value.value = fixed_value

    session.commit()
    print(f"Fixed {len(replacements)} covariate values.")


def update_descriptions():
    table = pd.read_csv('covariate_description_table.csv')
    print(table)
    session = SessionLocal()
    for _, table_row in table.iterrows():
        covariate_defn = session.query(CovariateDefn).filter(CovariateDefn.name == table_row['Database_name'])
        if covariate_defn is None:
            print(f"ERROR: {table_row['Database_name']} not in table")
        for row in covariate_defn:
            print(f'old row: {row}')
            row.description = table_row["Description (for website)"]
            print(row.name)
            row.name = table_row["What we want to call it"]
            print(table_row["What we want to call it"])
            print(row.name)
            print(f'new row: {row}')

    session.commit()
    session.close()


def merge_and_delete(table_path):
    table = pd.read_csv(table_path)
    session = SessionLocal()
    for _, table_row in table.iterrows():
        print(table_row['Covariate_ID'])
        base_covariate_defn = session.query(CovariateDefn).filter(
            CovariateDefn.name == table_row['Covariate_ID']).first()
        base_covariate_id = base_covariate_defn.id_key
        base_covariate_value_query = session.query(CovariateValue).filter(
            CovariateValue.covariate_defn_id == base_covariate_id)
        base_covariate_values = base_covariate_value_query.all()
        merge_covariate_defn = session.query(CovariateDefn).filter(
            CovariateDefn.name == table_row['Merge Target']).first()

        if merge_covariate_defn:
            merge_covariate_id = merge_covariate_defn.id_key
            for base_value in base_covariate_values:
                merged_covariate_value = session.query(CovariateValue).filter(
                    or_(
                        CovariateValue.sample_id == base_value.sample_id,
                        CovariateValue.study_id == base_value.study_id
                    ),
                    CovariateValue.covariate_defn_id == merge_covariate_id).first()

                if merged_covariate_value:
                    merged_covariate_value.value += f' {base_covariate_defn.name}: {base_value.value}'
                else:
                    # didn't exist to begin with, make a new one
                    new_merged_covariate_value = CovariateValue(
                        sample_id=base_value.sample_id,
                        study_id=base_value.study_id,
                        value=f'{base_covariate_defn.name}: {base_value.value}',
                        covariate_defn_id=merge_covariate_id
                    )
                    session.add(new_merged_covariate_value)
                session.delete(base_value)
        else:
            base_covariate_value_query.delete()
        session.delete(base_covariate_defn)
    session.commit()
    session.close()


if __name__ == "__main__":
    # print('update descriptions')
    # update_descriptions()
    print('merge and delete covariate values')
    merge_and_delete('covariate_merge_table_v2.csv')
    print('all done')
