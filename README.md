# Pest Control Living Database (PCLD)

The Pest Control Living Database (PCLD) is a USDA-funded initiative ([project #1023888](https://portal.nifa.usda.gov/web/crisprojectpages/1023888-fact-cyber-infrastructure-for-landscape-impacts-on-biocontrol.html)) that integrates agricultural pest observation data, biological traits of insects, and satellite-based Earth observation resources into a comprehensive, interactive data resource. Designed to streamline data-driven analysis for agricultural pest management, PCLD enables scientists, researchers, and growers to leverage data science and remote sensing technologies to predict pest dynamics, ultimately guiding agricultural decision-making and stewardship worldwide.

## Key Resources and Features

The Pest Control Living Database provides:

- **Over 100,000 pest-related observations** capturing insect activity, abundance, and impacts on crop yields.
- **Integrated remote sensing datasets** tailored for agricultural sampling locations.
- **Detailed insect traits data** to facilitate ecological and agricultural research.
- **Visualizations** of remote sensing datasets and agricultural data.
- **Standardized templates** for organizing and contributing data.

[Explore the detailed slideshow for the PCLD here](https://docs.google.com/presentation/d/1iGTxeFV1Zp3VcniMeiz6-uucHDSGvgQEDGn2xgWbEJ4/edit?usp=sharing).

## Directory Structure Overview

```
├── app
│ ├── dataset_defns # Definitions for datasets used by the database
│ ├── live_database # Primary database files and configuration
│ ├── secrets # Sensitive configuration and authentication details
│ ├── templates # HTML templates for web interface
│ └── pycache # Python compiled files cache
├── data # Miscellaneous data for database initialization or reference
├── gee_apps # JavaScript apps deployed on Google Earth Engine
└── llm_trait_pipeline # Standalone LLM pipeline for automated pest trait discovery
```
- `docker-compose.yml`: Main Docker configuration, orchestrating all core services.

## Contributing Data

We invite contributions of datasets containing information on pest abundance, natural enemies, parasitism/predation rates, or pest-related crop damage. Ideal datasets include over 100 farm-years of observations. Minimum data requirements are:

- Crop sampled
- Sampling date
- Metric type (e.g., pest abundance, predation rate)
- Measurement (per sampling unit)
- Management unit or unique farm ID
- Insect identification (if known)
- Geographical coordinates (if shareable)
- Sampling methodology (metadata)

Submit completed datasets using provided templates to [Richard Sharp](mailto:rich@springinnovate.org).

## Project Team

This project is managed by an interdisciplinary team including:

- Becky Chaplin-Kramer ([rchaplin@umn.edu](mailto:rchaplin@umn.edu))
- Colleen Miller ([Colleen Miller](mailto:mill5773@umn.edu))
- Danny Karp ([dkarp@ucdavis.edu](mailto:dkarp@ucdavis.edu))
- Richard Sharp ([rich@springinnovate.org](mailto:rich@springinnovate.org))

For general inquiries or further details, please contact the project leads above.
