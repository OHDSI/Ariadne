# Ariadne <a href="https://github.com/OHDSI/ariadne"><img src="docs/assets/logo.png" align="right" height="138" alt="Ariadne website" /></a>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Ariadne** is a Python toolkit for mapping source terminologies to standard concepts in the OHDSI Vocabulary. 

## Features

* **Verbatim term mapping:** maps terms that (almost) exactly match standard concepts. Using term normalization techniques like lowercasing, punctuation removal, and word stemming.
* **Embedding vector search:** leverages embedding language models to find semantically similar standard concepts for source terms.
* **Exact term mapping:** using reasoning LLMs to find exact matches in the vocabulary.
* **Evaluation:** uses golden standard mappings to evaluate mapping performance.

# Documentation

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

2. Edit the `.env` file to set your database connection details and API keys. **Do not commit this file to version control.**


## License

This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.