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


if __name__ == "__main__":
    session = SessionLocal()
    clean_whitespace(session)
    # replacements = [
    #     ('incorrect_value1', 'correct_value1'),
    #     ('incorrect_value2', 'correct_value2'),
    #     # Add more (original, fixed) tuples here
    # ]
    # fix_values(session, replacements)
    session.commit()
    session.close()
