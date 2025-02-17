#!/usr/bin/env python
#
# This file is part of CatCrops project.

from setuptools import setup, find_packages

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

# Extra requeriments
extras_require_dict={
    "gee": ["earthengine-api", "geemap", "geopy","rtree"]
}

setup(
    name='catcrops',
    version='1.0',
    packages=find_packages(include=['catcrops', 'catcrops.*']),
    url='https://github.com/UEA-Irta/CatCrops',
    license='GPL-3.0',
    author='Jordi Gené Mola',
    author_email='jordi.gene@irta.cat',
    maintainer='Magí Pàmies Sans',
    maintainer_email='magi.pamies@irta.cat',
    contcat_email='jordi.gene@irta.cat, magi.pamies@irta.cat',
    description=SHORT_DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,  # Dependències des del requirements.txt
    extras_require=extras_require_dict,
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: GPU :: NVIDIA CUDA",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: GIS",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Natural Language :: English",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"
    ],
    keywords=[
        "crop classification",
        "Sentinel-2",
        "deep learning",
        "Transformer",
        "GIS",
        "agriculture",
        "remote sensing",
        "time series",
        "machine learning",
        "Google Earth Engine",
        "DUN"
    ]
)
