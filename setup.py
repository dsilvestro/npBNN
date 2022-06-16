import os
import sys
import setuptools 

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)

if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
npBNN requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements_list = fh.read().splitlines()

setuptools.setup(
    name="npBNN",
    version="0.1.13",
    author="Daniele Silvestro and Tobias Andermann",
    description="Bayesian neural networks using Numpy and Scipy",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dsilvestro/npBNN",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU Affero General Public License v3.0",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=requirements_list
)
