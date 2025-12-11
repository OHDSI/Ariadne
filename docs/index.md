# Ariadne <a href="https://github.com/OHDSI/ariadne"><img src="assets/logo.png" align="right" height="138" alt="Ariadne website" /></a>

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)

**Ariadne** is a Python toolkit for mapping source terminologies to standard concepts in the OHDSI Vocabulary.

It includes the following modules:
- **term_cleanup:** for cleaning source terms, removing uninformative parts such as "not otherwise specified".
- **verbatim_mapping:** for mapping source terms that (almost) exactly match standard concepts, using term normalization techniques like lowercasing, punctuation removal, and word stemming.
- **vector_search:** for leveraging embedding language models to find semantically similar standard concepts for source terms.
- **llm_mapping:** for using reasoning LLMs to find exact matches in the vocabulary.
- **evaluation:** for evaluating mapping performance using golden standard mappings.
- **utils:** for common utility functions, including managing the configuration file.


