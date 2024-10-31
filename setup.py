#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "numpy",
    "matplotlib",
    "scipy",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="Deltares",
    author_email="joost.denbieman@deltares.nl",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    description="A Python toolbox for wave analysis",
    install_requires=requirements,
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="deltares_wave_toolbox",
    name="deltares_wave_toolbox",
    packages=find_packages(
        include=["deltares_wave_toolbox", "deltares_wave_toolbox.*"]
    ),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/Deltares-research/wave-toolbox",
    version="1.1.0",
    zip_safe=False,
)
