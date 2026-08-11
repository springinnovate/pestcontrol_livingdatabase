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
- `docker-compose.yml`: Production and Coolify configuration for the web app,
  Celery worker, and private Redis service.
- `docker-compose.dev.yml`: Local-development overrides that bind-mount the
  application source for rapid iteration.

## Coolify deployment

The production Compose stack is designed to be deployed as a Git-backed
Docker Compose resource in Coolify. Coolify supplies the shared reverse proxy
and TLS certificates, so this repository does not publish host ports or run a
second Traefik instance.

### Prepare persistent files

Create an application-owned directory on the Coolify server:

```text
/srv/apps/pcld/
|-- data/
|   `-- live_database/
`-- secrets/
    `-- service-account-key.json
```

Copy the existing contents of `app/live_database/` (or the contents exported
from the existing `pcld_data` volume) into
`/srv/apps/pcld/data/live_database/`. Copy the Google service-account key to
`/srv/apps/pcld/secrets/service-account-key.json`. Keep both paths out of Git,
restrict the credential file's permissions, and make the files readable by
the containers.

In Coolify, define these environment variables for the Compose resource:

```dotenv
PCLD_DATA_PATH=/srv/apps/pcld/data
PCLD_GOOGLE_CREDENTIALS_PATH=/srv/apps/pcld/secrets/service-account-key.json
```

Create the resource from this Git repository using the Docker Compose build
pack and `/docker-compose.yml`. Assign the following domain to the `app`
service:

```text
https://pcld.ecoshard.org:5000
```

Port `5000` is the web service's internal container port. Coolify serves the
public application on normal HTTPS port 443. Redis remains available only to
the web and worker services as `redis:6379` on the private Compose network.

### Local development

Copy `.env.example` to `.env`, ensure the referenced local data and credential
paths exist, and start Compose with the development override:

```bash
docker compose -f docker-compose.yml -f docker-compose.dev.yml up --build
```

The override bind-mounts `./app` for source changes. The production stack does
not mount the repository into its containers; application code is copied into
the image during the build.

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
