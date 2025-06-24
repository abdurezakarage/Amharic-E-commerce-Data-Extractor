# Ethiopian E-commerce Data Extractor

A comprehensive data ingestion and preprocessing system for Ethiopian Telegram e-commerce channels. This system collects, processes, and stores messages from multiple Ethiopian e-commerce Telegram channels with special handling for Amharic text.
## Project Structure


Amharic-E-commerce-Data-Extractor/
├── config/
│   └── __init__.py
├── data/
│   ├── labeled/
│   │   └── labeled_conll_output.txt
│   ├── processed/
│   │   ├── processed_telegram_data.csv
│   │   └── processed_telegram_data.json
│   └── raw/
│       ├── telegram_data.csv
│       └── photos/
│           └── ... (downloaded images)
├── docs/
├── examples/
│   └── __init__.py
├── labeled/
│   └── labeled_conll_output.txt
├── notebooks/
│   ├── conll_Labling.ipynb
│   ├── Fine_Tune_.ipynb
│   ├── preprocess.ipynb
│   └── __init__.py
├── requirements.txt
├── scraping_session.session
├── scripts/
│   └── __init__.py
├── src/
│   ├── core/
│   │   └── __init__.py
│   ├── models/
│   │   └── __init__.py
│   ├── services/
│   │   ├── amharic_trainer.py
│   │   ├── label_conll.py
│   │   ├── preproccesser.py
│   │   ├── telegram_scrapper.py
│   │   └── __init__.py
│   └── utils/
│       └── __init__.py
├── tests/
│   ├── integration/
│   │   └── __init__.py
│   ├── unit/
│   │   └── __init__.py
│   └── __init__.py
├── README.md
├── Makefile
├── pyproject.toml
└── .gitignore

# Services Overview

This directory contains the core scripts for data scraping, preprocessing, labeling, model training, and vendor analysis for Amharic e-commerce Telegram data.

## Notebooks Overview

The notebooks directory contains Jupyter notebooks that demonstrate and support the main data pipeline, including preprocessing, manual labeling, and model fine-tuning.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request



