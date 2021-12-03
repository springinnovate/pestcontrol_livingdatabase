"""
Convert XML to csv based off the files I found here.

https://www.juntadeandalucia.es/datosabiertos/portal/dataset/raif
"""
import argparse
import glob
import os

import xmltodict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='xml to csv')
    parser.add_argument(
        'xml_paths', nargs='+', type=str, help=(
            'path or patterns to xml files'))
    args = parser.parse_args()

    for xml_pattern in args.xml_paths:
        for xml_path in glob.glob(xml_pattern):
            print(f'processing {xml_path}')
            csv_path = f'{os.path.splitext(os.path.basename(xml_path))[0]}.csv'
            if os.path.exists(csv_path):
                print(f'{csv_path} exists, skipping')
                continue
            with open(xml_path, 'rb') as xml_file:
                doc = xmltodict.parse(xml_file.read())

            for key in doc['dataroot'].keys():
                if not key.startswith('@'):
                    doc = doc['dataroot'][key]
                    break
            columns = set(k for e in doc for k in e.keys())

            with open(csv_path, 'w', encoding="utf-8") as csv_file:
                csv_file.write(','.join(columns))
                csv_file.write('\n')

                for element in doc:
                    line_to_write = ','.join([
                            element[column] if column in element else ''
                            for column in columns])
                    csv_file.write(line_to_write)
                    csv_file.write('\n')
