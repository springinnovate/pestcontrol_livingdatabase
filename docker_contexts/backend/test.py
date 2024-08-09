from io import StringIO
import collections
import configparser
import csv
import hashlib
import json
import logging
import os
import pickle
import re
import sys
import zipfile

import numpy
from database import SessionLocal
from database_model_definitions import REQUIRED_STUDY_FIELDS, REQUIRED_SAMPLE_INPUT_FIELDS
from database_model_definitions import OBSERVATION, LATITUDE, LONGITUDE
from database_model_definitions import Study, Sample, Point, CovariateDefn, CovariateValue, CovariateType, CovariateAssociation, Geolocation
from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import url_for
from flask import send_file
from sqlalchemy import select, text
from sqlalchemy import distinct, func
from sqlalchemy.engine import Row
from sqlalchemy.sql import and_, or_, tuple_
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.sql.functions import GenericFunction
from sqlalchemy.types import String
from sqlalchemy.orm import subqueryload
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.orm import joinedload, contains_eager, selectinload

def to_dict(covariate_list):
    covariate_dict = collections.defaultdict(lambda: None)
    for covariate in covariate_list:
        covariate_dict[covariate.covariate_defn.name] = covariate.value
    return covariate_dict


def calculate_sample_display_table(session, query_to_filter):
    pre_covariate_display_query = (
        session.query(
            CovariateDefn.name,
            CovariateDefn.always_display,
            CovariateDefn.hidden,
            CovariateDefn.condition)
        .filter(
            or_(CovariateDefn.covariate_association == CovariateAssociation.SAMPLE.value,
                CovariateDefn.show_in_point_table == 1))
        .order_by(
            CovariateDefn.display_order,
            func.lower(CovariateDefn.name)))

    unique_values_per_covariate = collections.defaultdict(set)
    sample_covariate_list = []
    for index, (sample, study) in enumerate(query_to_filter):
        if index % 1000 == 0:
            print(f'sample display table index {index}')
        sample_covariates = sample.covariates
        study_covariates = [
            cov for cov in study.covariates
            if cov.covariate_defn.show_in_point_table == 1
        ]
        all_covariates = sample_covariates + study_covariates
        sample_covariate_list.append((sample, all_covariates))

        for covariate in all_covariates:
            if not isinstance(covariate.value, str) and (
                    covariate.value is None or numpy.isnan(covariate.value)):
                continue
            if isinstance(covariate.value, str):
                if covariate.value == 'null':
                    continue
                try:
                    # see if it could be numeric, if so just put true
                    float(covariate.value)
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(True)
                except ValueError:
                    unique_values_per_covariate[
                        covariate.covariate_defn.name].add(
                            covariate.value.lower())
            else:
                # it's a numeric, just note it's defined
                unique_values_per_covariate[
                    covariate.covariate_defn.name].add(True)

    # get all possible conditions
    covariate_display_order = []

    for name, always_display, hidden, condition in pre_covariate_display_query:
        if hidden:
            continue
        if condition is None or condition == 'null':
            if always_display or unique_values_per_covariate[name]:
                covariate_display_order.append(name)

        elif condition['value'].lower() in unique_values_per_covariate[
                condition['depends_on']]:
            covariate_display_order.append(name)

    display_table = []
    for sample, covariate_list in sample_covariate_list:
        covariate_dict = to_dict(covariate_list)
        display_table.append([
            covariate_dict[name]
            for name in covariate_display_order])
    return covariate_display_order, display_table

def main():
    session = SessionLocal()
    sample_query = (
        session.query(Sample, Study)
        .join(Sample.study)
        .join(Sample.point)
        .options(selectinload(Sample.covariates))
    ).yield_per(1000)

    sample_covariate_display_order, sample_table = calculate_sample_display_table(
        session, sample_query)
    sample_covariate_display_order = [
        OBSERVATION, LATITUDE, LONGITUDE] + sample_covariate_display_order
    sample_table_io = StringIO(newline='')

    writer = csv.writer(sample_table_io)
    writer.writerow(sample_covariate_display_order)

    for index, (sample_row, row) in enumerate(zip(sample_query, sample_table)):
        if index % 1000 == 0:
            print(f'on index {index}')
        row = [
            str(sample_row[0].observation),
            str(sample_row[0].point.latitude),
            str(sample_row[0].point.longitude),] + row
        clean_row = [x if x is not None else 'None' for x in row]
        writer.writerow(clean_row)

    sample_table_io.seek(0)
    with open('all_data.csv', 'w', newline='') as file:
        file.write(sample_table_io.getvalue())


if __name__ == '__main__':
    main()