import configparser
import glob
import os
import re


def sanitize_filename(filename):
    # Remove or replace invalid characters
    # You can expand the set of characters based on your requirements
    sanitized = re.sub(r'[\\/:*?"<>|]', '_', filename)

    # Remove leading and trailing periods, as some OSs may have issues with them
    sanitized = sanitized.strip('.')

    return sanitized


with open('viewer_template.js', 'r') as jsfile:
    js_raw = jsfile.read()

# Iterate over all .ini files in the directory
for ini_file in glob.glob('*.ini'):
    config = configparser.ConfigParser()
    config.read(ini_file)
    print(ini_file)
    section_id = os.path.splitext(os.path.basename(ini_file))[0]
    print(config.sections())
    # Extract values from the .ini file (adjust the section names as needed)
    valid_years = str([str(x) for x in eval(config[section_id]['VALID_YEARS'])])
    dataset_id = config[section_id]['gee_dataset']
    band_id = config[section_id]['band_name']
    try:
        dataset_name = config[section_id]['dataset_name']
    except KeyError:
        dataset_name = dataset_id
    try:
        image_only = config[section_id]['image_only']
    except KeyError:
        image_only = 'false'

    # Replace placeholders in the template
    js_content = js_raw.replace('$VALID_YEARS$', valid_years)
    js_content = js_content.replace('$DATASET_NAME$', dataset_name)
    js_content = js_content.replace('$DATASET_ID$', dataset_id)
    js_content = js_content.replace('$BAND_ID$', band_id)
    js_content = js_content.replace('$IMAGE_ONLY$', image_only)
    js_content += f'\n{dataset_name}\n{section_id}.js\n'

    # Write to new .js file (named according to the dataset)
    with open(sanitize_filename(f'{section_id}.js'), 'w') as js_file:
        js_file.write(js_content)
