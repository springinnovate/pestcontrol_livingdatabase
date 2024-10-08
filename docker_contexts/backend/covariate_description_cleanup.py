import pandas as pd
from database import SessionLocal
from sqlalchemy.orm import Session
from sqlalchemy import select, func
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


def main():
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


if __name__ == "__main__":
    main()