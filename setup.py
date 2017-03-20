#!/usr/bin/env python

from setuptools import setup

setup(
    name="SimpleSoapy",
    version="1.2.0",
    description="Simple pythonic wrapper for SoapySDR library",
    long_description=open('README.rst').read(),
    author="Michal Krenek (Mikos)",
    author_email="m.krenek@gmail.com",
    url="https://github.com/xmikos/simplesoapy",
    license="MIT",
    py_modules=["simplesoapy"],
    install_requires=[
        'numpy',
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Telecommunications Industry",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering",
        "Topic :: Communications :: Ham Radio"
    ]
)
