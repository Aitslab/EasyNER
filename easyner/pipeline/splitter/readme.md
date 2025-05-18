# SentenceSplitter module
This module is responsible for splitting text into sentences. It uses different tokenizers to achieve this, and it can load data from various sources and write the output to different formats.

Loosely based on the original implementation in EasyNer pipline written by Rafsan Ahmed.

Written by Carl Ollvik Aasa

## Code structure

Data flows loader → processor(splitter) → writer

Behavior of each component is defined in the config file.
- loader: loads the data from the source
- processor: processes the data, splits into sentences
- writer: writes the data to the destination

SentenceSplitter/
├── tokenizers/
│   ├── TokenizerBase (abstract)
│   ├── SpacyTokenizer
│   └── NLTKTokenizer
├── loaders/
│   ├── DataLoaderBase (abstract)
│   ├── StandardLoader
│   └── PubMedLoader
├── writers/
│   ├── OutputWriterBase (abstract)
│   └── JSONWriter
├── SplitterProcessor
└── SplitterRunner
