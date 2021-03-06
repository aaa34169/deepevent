#!/usr/bin/python
from setuptools import find_packages,setup
import os,sys


with open("README.md", "r") as fh:
    long_description = fh.read()



setup(
    name="deepevent",
    version="0.3.2",
    author="Lempereur Mathieu",
    author_email="mathieu.lempereur@univ-brest.fr",
    description="Deep Learning to identify gait events",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
    packages=find_packages(),
    entry_points={
          'console_scripts': [
              'deepevent = deepevent.__commandLine__:main'
          ]
      },

    install_requires=['tensorflow>=2.1.0',
                      'keras>=2.3.1',
                      'numpy>=1.18.1',
                      'scipy>=1.4.1',
                      'pyBTK>=0.1.1'],
    classifiers=['Programming Language :: Python',
                 'Programming Language :: Python :: 3.7',
                 'Operating System :: Microsoft :: Windows',
                 'Natural Language :: English'],

)
