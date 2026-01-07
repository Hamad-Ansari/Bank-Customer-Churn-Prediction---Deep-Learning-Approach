"""
Setup script for bank churn prediction package
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="bank-churn-prediction",
    version="1.0.0",
    author="Hammad Zahid",
    author_email="mrhammadzahi24@gmail.com",
    description="A machine learning project for predicting bank customer churn",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hamad-Ansari/bank-churn-prediction",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.991",
        ],
        "docs": [
            "mkdocs>=1.4.0",
            "mkdocs-material>=9.0.0",
            "mkdocstrings[python]>=0.20.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "bank-churn-train=src.model_training:main",
            "bank-churn-predict=src.predict:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["config/*.yaml", "data/*.csv"],
    },
)
