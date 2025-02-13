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

SHORT_DESCRIPTION = 'Llibreria per executar el CatCrops'

# Càrrega de les dependències
requirements = parse_requirements('requirements.txt')

setup(
    name='catcrops',
    version='1.0',
    packages=['catcrops'],
    #packages=find_packages(),
    url='https://github.com/UEA-Irta/CatCrops',
    license='GPL',
    author="IRTA - Us Eficient de l'Aigua",
    author_email='irta.remote.sensing@gmail.com',
    maintainer='magipamies',
    maintainer_email='magipamies@gmail.com',
    contact="IRTA - Us Eficient de l'Aigua",
    contcat_email='irta.remote.sensing@gmail.com',
    description=SHORT_DESCRIPTION,
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    install_requires=requirements,  # Dependències des del requirements.txt
    python_requires='>=3.10',
    classifiers=[
        "Development Status :: Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Agricultural Science",
        "Topic :: Scientific/Engineering :: Hydrology",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3"],
    keywords=['SEN-ET', 'Evapotranspiration',
              'Sentinel-2', 'Sentinel-3', 'SLSTR', 'Remote Sensing']
)
