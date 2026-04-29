from pathlib import Path

from setuptools import find_packages, setup


ROOT = Path(__file__).resolve().parent
LONG_DESCRIPTION = (ROOT / "USAGE.md").read_text(encoding="utf-8")

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
    include_package_data=True,
    author="Felix Schulz",
    description="Generate exams from Excel question banks with exact type ratios, semantic diversity, and LaTeX/PDF output",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    keywords="exam, question bank, semantic similarity, education",
    python_requires=">=3.8",
)
