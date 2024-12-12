"""ColorfulBirds setup script."""

import os
from setuptools import setup, find_packages

# Load the required dependencies from the requirements file
with open("requirements.txt") as requirements_file:
    install_requires = requirements_file.read().splitlines()

setup(
   name='colourfulbirds',
   packages = find_packages(),   
   install_requires = install_requires
)
