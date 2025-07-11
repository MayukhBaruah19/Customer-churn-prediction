import os 
from setuptools import setup, find_packages
from typing import List

HYPEN_E_DOT='-e .'

def get_requirements(filename='requirements.txt') -> List[str]: 
    """Read requirements.txt and return a list of packages."""
    with open(filename, 'r') as f:
        lines = f.readlines()
        requirements = [
            line.strip() for line in lines
            if line.strip() and not line.startswith('#')
        ]
        if HYPEN_E_DOT in requirements:
            requirements.remove(HYPEN_E_DOT)
    return requirements


setup(
    name='red_wine_quality_prediction',
    version='0.1.0',
    author='Mayukh',
    author_email='mayukhbaeuah91@gmail.com',
    description='is projrct is about predicting the quality of red wine using machine learning techniques.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/MayukhBaruah19/Red-wine-quality-prediction',
    packages=find_packages(),
    install_requires=get_requirements()  # Automatically finds all sub-packages
)