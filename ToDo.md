## Package structure
easyner/
├── __init__.py             # Main package exports
├── __main__.py             # Entry point for direct execution
├── api/                    # Public interfaces
│   ├── __init__.py         # Exports user-facing APIs
│   └── cli.py              # Command-line interface definition
├── core/                   # Core business logic
│   ├── __init__.py
│   ├── config/             # Configuration management
│   │   ├── __init__.py
│   │   ├── generator.py
│   │   ├── manager.py
│   │   └── validator.py
│   ├── models/             # Data models and domain objects
│   │   ├── __init__.py
│   │   └── entities.py
│   └── pipeline/           # Processing pipeline components
│       ├── __init__.py
│       ├── ner/            # Named Entity Recognition
│       │   ├── __init__.py
│       │   ├── processor.py
│       │   ├── factory.py
│       │   └── transformer_based/
│       │       ├── __init__.py
│       │       ├── biobert.py
│       │       └── inference.py
│       ├── analysis/       # Analysis components
│       │   ├── __init__.py
│       │   └── statistics.py
│       ├── splitter/       # Sentence splitting
│       │   ├── __init__.py
│       │   └── processor.py
│       └── merger/         # Entity merging
│           ├── __init__.py
│           └── processor.py
├── data/                   # Data processing
│   ├── __init__.py
│   ├── loaders/            # Data loading modules
│   │   ├── __init__.py
│   │   ├── pubmed.py
│   │   ├── cord.py
│   │   └── text.py
│   └── parsers/            # Data parsing modules
│       ├── __init__.py
│       └── json.py
├── io/                     # I/O operations
│   ├── __init__.py
│   ├── factory.py
│   ├── handlers/           # I/O handlers
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── file.py
│   │   └── database.py
│   └── database/           # Database operations
│       ├── __init__.py
│       ├── schemas/        # SQL schema definitions
│       │   ├── __init__.py
│       │   ├── articles.sql
│       │   ├── sentences.sql
│       │   └── entities.sql
│       └── handlers/
│           ├── __init__.py
│           └── duckdb.py
├── utils/                  # Utility functions
│   ├── __init__.py
│   ├── file.py
│   ├── text.py
│   └── validation.py
└── config/                 # Configuration files
    ├── __init__.py
    └── schema.json
