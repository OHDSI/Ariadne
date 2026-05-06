# Ariadne <a href="https://github.com/OHDSI/ariadne"><img src="docs/assets/logo.png" align="right" height="138" alt="Ariadne website" /></a>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Ariadne** is a Python toolkit to harmonize source vocabularies into the OHDSI Standardized Vocabularies.

It currently supports three workflows:

- **Conditions:** exact matching and hierarchy matching (for identifying a parent concept when no exact standard match is available).
- **Drugs:** prepares source codes for inserting into the OHDSI Boiler software as described [here](https://github.com/OHDSI/Vocabulary-v5.0/wiki/Community-contribution.-Drug-vocabularies).
- **Procedures:** exact source-to-standard concept mapping for procedure vocabularies.

The toolkit is designed to be extensible, allowing for the addition of new workflows (e.g., measurements, devices) and mapping strategies as needed.


## Features

* **Clean-up:** normalizes source terms per mapping rules, removing non-essential information.
* **Verbatim term mapping:** maps terms that (almost) exactly match standard concepts using normalization (lowercasing, punctuation removal, stemming).
* **Embedding vector search:** retrieves semantically similar standard concepts as mapping candidates.
* **LLM-assisted exact mapping:** selects the best standard concept from candidates.
* **Hierarchy matching (Conditions):** finds parent concepts for unmatched terms in the standard hierarchy.
* **Evaluation:** compares outputs against gold-standard mappings.

## Documentation

Comprehensive documentation is available on the [Ariadne Documentation Site](https://ohdsi.github.io/Ariadne/).


##  Installation

### Prerequisites
* Python **3.12** or higher.

### Install from Source
```bash
git clone [https://github.com/OHDSI/ariadne.git](https://github.com/OHDSI/ariadne.git)
cd ariadne

pip install -e .
```

## Configuration

Ariadne uses environment variables for database connections and API keys.

1. Copy the example configuration:
    ```bash
    cp .env.example .env
    ``` 

2. Edit the `.env` file to set your database connection details and API keys. **Do not commit this file to version control.** We recommend moving it to a secure location outside the project directory.


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.