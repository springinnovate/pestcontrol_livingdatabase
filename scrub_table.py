"""Replace a column with user defined table values."""
import argparse
import concurrent
import logging
import os
import sys

from itertools import islice
from clean_table import _process_line

logging.basicConfig(
    level=logging.DEBUG,
    format=(
        '%(asctime)s (%(relativeCreated)d) %(levelname)s %(name)s'
        ' [%(funcName)s:%(lineno)d] %(message)s'),
    stream=sys.stdout)
LOGGER = logging.getLogger(__name__)


def batch(iterable, size):
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            return
        yield chunk


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(description=(
        'replace column in CSV with replacement values in a second table.'))
    parser.add_argument('table_path', help='Path to table.')
    args = parser.parse_args()

    scrubbed_file = open(
        f'''scrubbed_{os.path.basename(os.path.splitext(
            args.table_path)[0])}.csv''', 'wb')
    scrubbed_file.write(b'\xEF\xBB\xBF')
    missing_letter_set = set()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        print(f'process reading {args.table_path}')
        with open(args.table_path, 'rb') as table_file:
            n_lines = len(['x' for line in table_file])
            print(f'lines in table: {n_lines}')
            table_file.seek(0)
            processed_lines = []
            for lines_batch in batch(table_file, 1000):
                n_lines -= 1000
                processed_lines = list(
                    executor.map(_process_line, lines_batch))
                print(f'{n_lines} left to process')

            for line_no, line in enumerate(processed_lines):
                missing_letter_set.update([
                    word.replace('"', '')
                    for element in line.decode('utf-8').split(',') if '_' in element
                    for word in element.split(' ')
                    if '_' in word])
                scrubbed_file.write(line)


if __name__ == '__main__':
    main()
