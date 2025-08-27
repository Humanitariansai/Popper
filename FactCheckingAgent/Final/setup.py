#!/usr/bin/env python3
"""
Setup script for Scientific Fact Checking System
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="scientific-fact-checker",
    version="1.0.0",
    author="Scientific Fact Checking Team",
    author_email="contact@example.com",
    description="A comprehensive AI-powered tool for extracting, validating, and fact-checking scientific statements",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/scientific-fact-checker",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Text Processing :: Linguistic",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
            "pre-commit>=2.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "scientific-fact-checker=scientific_fact_checker:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.md", "*.json"],
    },
    keywords="fact-checking, scientific, AI, machine-learning, research, validation",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/scientific-fact-checker/issues",
        "Source": "https://github.com/yourusername/scientific-fact-checker",
        "Documentation": "https://github.com/yourusername/scientific-fact-checker#readme",
    },
)
