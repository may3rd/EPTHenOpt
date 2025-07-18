"""
Setup script for the EPTHenOpt package.

This script is used to install the EPTHenOpt package, making it available
in the Python environment and creating a command-line entry point.
"""
from setuptools import setup, find_packages

# Read the contents of your README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the contents of your requirements file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# --- ADDED: Define development dependencies ---
dev_requirements = [
    'sphinx',
    'sphinx_rtd_theme', # A popular, clean theme for Sphinx
    'pytest'
]

setup(
    name="EPTHenOpt",
    version="0.8.0",  # Version bump
    author="Maetee Lorprajuksiri",
    author_email="26008353@pttgcgroup.com",
    description="A package for Heat Exchanger Network (HEN) Synthesis and Optimization.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/may3rd/EPTHenOpt",
    packages=find_packages(exclude=["tests*"]),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
    # --- ADDED: extras_require for development install ---
    extras_require={
        'dev': dev_requirements
    },
    entry_points={
        'console_scripts': [
            'run_hen_problem=EPTHenOpt.run_problem:cli',
        ],
    },
)
