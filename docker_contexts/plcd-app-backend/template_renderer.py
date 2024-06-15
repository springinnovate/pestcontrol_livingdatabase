from jinja2 import Template
from database_model_definitions import STUDY_LEVEL_VARIABLES
from database_model_definitions import BASE_FIELDS


def main():
    # Read the template file directly
    with open('templates/living_database_study.jinja', 'r') as file:
        template_content = file.read()

    living_database_template = Template(template_content)

    # Define the variables to fill in the template

    covariate_names = [
        'LC1Gau250',
        'LC2Gau250',
        'LC3Gau250',
        'LC4Gau250',
        'LC5Gau250',
        'LC6Gau250',
        'LC7Gau250',
        'LC8Gau250',
        'LC9Gau250',
        'LC1Gau750',
        'LC2Gau750',
        'LC3Gau750',
        'LC4Gau750',
        'LC5Gau750',
        'LC6Gau750',
        'LC7Gau750',
        'LC8Gau750',
        'LC9Gau750',
        'LC1Gau1250',
        'LC2Gau1250',
        'LC3Gau1250',
        'LC4Gau1250',
        'LC5Gau1250',
        'LC6Gau1250',
        'LC7Gau1250',
        'LC8Gau1250',
        'LC9Gau1250',
        'ExclosureTreatment',
        'MeasureType',
        'CropType',
        'AnnualPerennial',
        'Organic',
        'LocalDiversity',
        'InsecticidePlot',
        'InsecticideFarm',
        'ConfidenceInsecticide',
        'Tilling',
        'SiteDesc',
        'NPerEffortPred',
        'Stand_ActPred',
        'NPerEffortDamage',
        'StandActDam',
        'NaturalLandProp250',
        'NaturalLandProp750',
        'NaturalLandProp1250',
        'AgProp250',
        'AgProp750',
        'AgProp1250',
        'GrasslandProp250',
        'GrasslandProp750',
        'GrasslandProp1250',
        'PollinatorDepend',
        'CoverType250',
        'CoverType750',
        'CoverType1250',
        'TraitGenSpec',
    ]

    variables = {
        'study_level_variables': STUDY_LEVEL_VARIABLES,
        'headers': BASE_FIELDS + [f'Covariate_{val}' for val in covariate_names],
    }

    # Render the template with variables
    output = living_database_template.render(variables)

    # Print the output or use it as needed
    print(output)


if __name__ == '__main__':
    main()
