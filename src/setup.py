"""
Setup script for the pampal package.
"""
from setuptools import setup, find_packages

setup(
    name="pampal",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
    ],
    author="PAMpal Development Team",
    author_email="robby.moseley@gmail.com",
    description="Python version of PAMpal for loading and processing passive acoustic data",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/Robbyswimmer/PAMPalPython",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License (GPL)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7",
)
