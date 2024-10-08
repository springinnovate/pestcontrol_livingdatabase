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


def merge_and_delete():
    table = pd.read_csv('covariate_merge_table.csv')
    session = SessionLocal()
    for _, table_row in table.iterrows():
        print(table_row['Covariate_ID'])
        base_covariate_defn = session.query(CovariateDefn).filter(
            CovariateDefn.name == table_row['Covariate_ID']).first()
        base_covariate_id = base_covariate_defn.id_key
        base_covariate_values = session.query(CovariateValue).filter(
            CovariateValue.covariate_defn_id == base_covariate_id).all()
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
                    merged_covariate_value.value += base_value.value
                else:
                    # didn't exist to begin with, make a new one
                    new_merged_covariate_value = CovariateValue(
                        sample_id=base_value.sample_id,
                        study_id=base_value.study_id,
                        value=base_value.value,
                        covariate_defn_id=merge_covariate_id
                    )
                    session.add(new_merged_covariate_value)
                session.delete(base_value)
        session.delete(base_covariate_defn)
    session.commit()
    session.close()


def merge_and_delete_with_cudf():
    # Step 1: Read the covariate_merge_table.csv into a Pandas DataFrame
    table = pd.read_csv('covariate_merge_table.csv')

    # Step 2: Connect to your SQLite database
    conn = sqlite3.connect('instance/living_database.db')

    # Step 3: Load CovariateDefn and CovariateValue tables into Pandas DataFrames
    covariate_defn_df = pd.read_sql_query("SELECT * FROM CovariateDefn", conn)
    covariate_value_df = pd.read_sql_query("SELECT * FROM CovariateValue", conn)

    # Step 4: Convert Pandas DataFrames to cuDF DataFrames
    covariate_defn_cudf = cudf.DataFrame.from_pandas(covariate_defn_df)
    covariate_value_cudf = cudf.DataFrame.from_pandas(covariate_value_df)
    table_cudf = cudf.DataFrame.from_pandas(table)

    # Step 5: Merge table_cudf with covariate_defn_cudf to get IDs
    table_cudf = table_cudf.merge(
        covariate_defn_cudf[['id_key', 'name']],
        left_on='Covariate_ID', right_on='name', how='left'
    ).rename(columns={'id_key': 'base_covariate_id'})

    table_cudf = table_cudf.merge(
        covariate_defn_cudf[['id_key', 'name']],
        left_on='Merge Target', right_on='name', how='left'
    ).rename(columns={'id_key': 'merge_covariate_id'})

    # Step 6: Prepare covariate_value_cudf for merging
    # Fill NaN in sample_id and study_id with -1 to allow joins
    covariate_value_cudf['sample_id'] = covariate_value_cudf['sample_id'].fillna(-1)
    covariate_value_cudf['study_id'] = covariate_value_cudf['study_id'].fillna(-1)

    # Convert IDs to integers
    covariate_value_cudf['sample_id'] = covariate_value_cudf['sample_id'].astype('int64')
    covariate_value_cudf['study_id'] = covariate_value_cudf['study_id'].astype('int64')

    # Step 7: Iterate over table_cudf rows to perform merging
    for idx, table_row in table_cudf.iterrows():
        base_covariate_id = table_row['base_covariate_id']
        merge_covariate_id = table_row['merge_covariate_id']

        if cudf.isnull(merge_covariate_id):
            print(f"Merge target '{table_row['Merge Target']}' not found!")
            continue

        # Filter base_covariate_values
        base_values = covariate_value_cudf[
            covariate_value_cudf['covariate_defn_id'] == base_covariate_id
        ]

        # Prepare keys for joining
        base_values['join_key'] = base_values.apply(
            lambda row: (row['sample_id'], row['study_id']), axis=1
        )

        # Filter merge_covariate_values
        merge_values = covariate_value_cudf[
            covariate_value_cudf['covariate_defn_id'] == merge_covariate_id
        ]

        merge_values['join_key'] = merge_values.apply(
            lambda row: (row['sample_id'], row['study_id']), axis=1
        )

        # Step 8: Merge base_values with merge_values on join_key
        merged = base_values.merge(
            merge_values[['id_key', 'join_key', 'value']],
            on='join_key', how='left', suffixes=('_base', '_merge')
        )

        # Step 9: Update existing values
        existing_values = merged[~merged['id_key_merge'].isnull()]
        covariate_value_cudf.loc[
            covariate_value_cudf['id_key'].isin(existing_values['id_key_merge']),
            'value'
        ] = existing_values['value_base'] + existing_values['value_merge']

        # Step 10: Create new values where necessary
        new_values = merged[merged['id_key_merge'].isnull()]
        if not new_values.empty:
            new_entries = cudf.DataFrame({
                'covariate_defn_id': merge_covariate_id,
                'sample_id': new_values['sample_id'],
                'study_id': new_values['study_id'],
                'value': new_values['value_base']
            })
            covariate_value_cudf = cudf.concat([covariate_value_cudf, new_entries], ignore_index=True)

        # Step 11: Delete base_values from covariate_value_cudf
        covariate_value_cudf = covariate_value_cudf[
            covariate_value_cudf['id_key'].isin(base_values['id_key']) == False
        ]

        # Step 12: Delete base_covariate_defn from covariate_defn_cudf
        covariate_defn_cudf = covariate_defn_cudf[
            covariate_defn_cudf['id_key'] != base_covariate_id
        ]

    # Step 13: Write updated DataFrames back to SQLite
    # Convert cuDF DataFrames back to Pandas DataFrames
    covariate_value_df_updated = covariate_value_cudf.to_pandas()
    covariate_defn_df_updated = covariate_defn_cudf.to_pandas()

    # Write back to SQLite
    covariate_value_df_updated.to_sql('CovariateValue', conn, if_exists='replace', index=False)
    covariate_defn_df_updated.to_sql('CovariateDefn', conn, if_exists='replace', index=False)

    conn.close()


if __name__ == "__main__":
    print('update descriptions')
    update_descriptions()
    print('merge and delete covariate values')
    merge_and_delete()
    print('all done')
