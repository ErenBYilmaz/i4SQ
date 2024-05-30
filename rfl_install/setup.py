#!/usr/bin/env python3

from setuptools import Extension, find_packages, setup


# To prepare a new release run clog in version 0.9.2 like this:
# clog -F -C CHANGELOG.md -r 'https://github.com/fhkiel-mlaip/rfl'


class NumpyExtension(Extension):
    def __getattribute__(self, name):
        if name == 'include_dirs':
            import numpy
            return [numpy.get_include()]

        return super().__getattribute__(name)


setup(
    name='rfl',
    version='3.0',
    packages=find_packages(),
    package_data={
        'rfl': ['*.cl']
    },
    ext_modules=[
        NumpyExtension('rfl._features', ['rfl/_features.pyx'])
    ],
    install_requires=[
        'click',
        'hiwi',
        'matplotlib',
        'numpy',
        'scikit-image',
        'scikit-learn',
        'scipy',
        'SimpleITK'
    ],
    tests_require=[
        'pep8-naming',
        'pytest',
        'pytest-flake8'
    ],
    setup_requires=[
        'pytest-runner',
        'setuptools>=18.0',
        'cython'
    ],
    entry_points='''
        [console_scripts]
        rfl=rfl.cli:main
    ''',
)
