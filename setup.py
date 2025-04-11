from setuptools import find_packages, setup

setup(
    name="easyner",
    version="0.1.0",
    packages=find_packages(),
    description="NER processing tools",
    author="",
    author_email="",
    install_requires=[
        # Add your dependencies here
        "spacy",
    ],
)
