"""This is used in cases where we have
column_a - column_b

and there are multiple pairs of covariate a that
match covariate b in a sample

i.e. 'Crop_latin_name' -- 'Crop_common_name'

if there's samples where there is a covariate a but not a b
"""
import argparse
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from database import SessionLocal, init_db
from database_model_definitions import Sample, CovariateValue, CovariateDefn
from sqlalchemy import select, func
from sqlalchemy.orm import aliased


def main():
    init_db()
    session = SessionLocal()
    parser = argparse.ArgumentParser(description='covariate pairs')
    parser.add_argument('cov_name_a')
    parser.add_argument('cov_name_b')
    args = parser.parse_args()

    # Create aliases for the two covariates
    cov_a = aliased(CovariateValue)
    cov_b = aliased(CovariateValue)

    subquery = (
        select(cov_a.sample_id)
        .join(CovariateDefn, CovariateDefn.id_key == cov_a.covariate_defn_id)
        .filter(CovariateDefn.name == args.cov_name_a)
    )

    # Main query to get the values for cov_name_a and cov_name_b together
    query = (
        select(cov_a.value, cov_b.value)
        .join(CovariateDefn, CovariateDefn.id_key == cov_a.covariate_defn_id)
        .outerjoin(cov_b, cov_b.sample_id == cov_a.sample_id)  # Left outer join to include cases where cov_b may not exist
        .filter(CovariateDefn.name == args.cov_name_a)
        .filter(cov_b.covariate_defn.has(CovariateDefn.name == args.cov_name_b) | (cov_b.value == None))
        .filter(cov_a.sample_id.in_(subquery))
    ).distinct()

    results = session.execute(query).all()
    for col_a, col_b in sorted(results):
        print(f'{col_a}, {col_b}')


if __name__ == '__main__':
    main()
