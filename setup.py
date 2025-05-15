from setuptools import setup, find_packages

setup(
    name="exam-generator",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.5.0",
        "numpy>=1.20.0",
        "openpyxl>=3.0.0",
        "python-docx>=0.8.11",
        "PyYAML>=6.0",
        "jsonschema>=4.0.0",
        "sentence-transformers>=2.2.0",
        "scikit-learn>=1.0.0"
    ],
    entry_points={
        "console_scripts": [
            "exam-generator=main:main"
        ]
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="A tool to generate exams from question banks with semantic diversity",
    keywords="exam, question bank, semantic similarity, education",
    python_requires=">=3.8",
)