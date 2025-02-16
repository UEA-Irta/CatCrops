#!/usr/bin/env python
#
# This file is part of SEN-ET project.

from setuptools import setup

# Llegeix les dependències des del fitxer requirements.txt
def parse_requirements(filename):
    with open(filename, 'r') as req_file:
        return [
            line.split("/")[-1].split("@")[0] if line.startswith("git") else line.strip()
            for line in req_file
            if line.strip() and not line.startswith('#')
        ]

SHORT_DESCRIPTION = ("A remote sensing-based library for crop classification using Sentinel-2 imagery and the "
                     "Transformer model for time series analysis.")

# Càrrega de les dependències
requirements = parse_requirements('requirements.txt')

setup(
    name='catcrops',
    version='1.0',
    packages=['catcrops'],
    url='https://github.com/UEA-Irta/CatCrops',
    license='GPL',
    author='Jordi Gené Mola, Magí Pàmies Sans',
    author_email='jordi.gene@irta.cat',
    maintainer='Magí Pàmies Sans',
    contcat_email='jordi.gene@irta.cat',
    description=SHORT_DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,  # Dependències des del requirements.txt
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Agricultural Science",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"]
)
