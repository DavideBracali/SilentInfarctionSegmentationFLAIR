#!/usr/bin/python
# -*- coding: utf-8 -*-

try:
    from setuptools import setup
    from setuptools import find_packages
except ImportError:
    from distutils.core import setup
    from distutils.core import find_packages

import os

__author__ = ['Davide Bracali']
__email__ = ['davide.bracali@studio.unibo.it']

def get_requires(requirements_filename):
    '''
    What packages are required for this module to be executed?

    Parameters
    ----------
    requirements_filename : str
        filename of requirements (e.g requirements.txt)

    Returns
    -------
    requirements : list
        list of required packages
    '''
    with open(requirements_filename, 'r') as fp:
        requirements = fp.read()

    return list(filter(lambda x: x != '', requirements.split()))

def format_requires(requirement):
    '''
    Check if the specified requirements is a package or a link to github report.
    If it is a link to a github repo, it will format it according to the specification 
    Of install requirements. 
    The git hub repo url is assumed to be in the form:
    git+https://github.com/UserName/RepoName

    and will be formatted as 
    RepoName @ git+https://github.com/UserName/RepoName

    Parameters
    ----------
    requirement : str
        str with the reuirement to be analyzed
    
    Returns
    -------
    foramt_requirement: str
        requirement formatted according to install_requires specs
    '''

    if "http" not in requirement:
        return requirement

    package_name = requirement.split('/')[-1]
    return f'{package_name} @ {requirement}'

def read_description(readme_filename):
    '''
    Description package from filename

    Parameters
    ----------
    readme_filename : str
        filename with readme information (e.g README.md)

    Returns
    -------
    description : str
        str with description
    '''

    try:

        with open(readme_filename, 'r') as fp:
            description = '\n'
            description += fp.read()

        return description

    except IOError:
        return ''



here = os.path.abspath(os.path.dirname(__file__))

AUTHOR = 'Davide Bracali'
EMAIL = 'davide.bracali@studio.unibo.it'

NAME = 'SilentInfarctionSegmentationFLAIR'
DESCRIPTION = ''
REQUIRES_PYTHON = '>=3'
VERSION = None
VERSION_FILENAME = os.path.join(here, 'SilentInfarctionSegmentationFLAIR', '__version__.py')
README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
URL = 'https://github.com/DavideBracali/SilentInfarctionSegmentationFLAIR'
KEYWORDS = ['MRI','FLAIR', 'segmentation', 'thresholding', 'medical-imaging', 'brain']

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
    LONG_DESCRIPTION = read_description(README_FILENAME)
except IOError:
    LONG_DESCRIPTION = DESCRIPTION

# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
  with open(VERSION_FILENAME) as fp:
    exec(fp.read(), about)
else:
    about['__version__'] = VERSION

# parse version variables and add them to command line as definitions
Version = about['__version__'].split('.')

setup(
    name                          = NAME,
    version                       = about['__version__'],
    description                   = DESCRIPTION,
    long_description              = LONG_DESCRIPTION,
    long_description_content_type = 'text/markdown',
    author                        = AUTHOR,
    author_email                  = EMAIL,
    maintainer                    = AUTHOR,
    maintainer_email              = EMAIL,
    python_requires               = REQUIRES_PYTHON,
    install_requires              = get_requires(REQUIREMENTS_FILENAME),  #°_° fare file requirements
    url                           = URL,
    download_url                  = URL,
    keywords                      = KEYWORDS,             
    packages                      = find_packages(include=[
                                        'SilentInfarctionSegmentationFLAIR',
                                        'SilentInfarctionSegmentationFLAIR.*'],
                                        exclude=('test', 'testing')),
    include_package_data          = True, # no absolute paths are allowed
    platforms                     = 'any',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent"],
    license                       = 'MIT'
)