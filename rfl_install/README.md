# rfl [![Build Status](https://travis-ci.com/fhkiel-mlaip/rfl.svg?token=WHxqMPs4z8VYjCAuFKAb&branch=master)](https://travis-ci.com/fhkiel-mlaip/rfl)

A generic object localizer using an ensemble of decision tree regressors in
combination with random masks for feature extraction.

This repository provides a Python 3.6 (or upwards) package, including a
Python API as well as a command line interface.

## Installation

To install this package you use `pip`, which also installs all the
mandatory dependencies if not available.

```shell
$ pip install git+https://github.com/fhkiel-mlaip/rfl.git
$ rfl --help
```

If you want to make use of GPU utilization, make sure to install `pyopencl` and
install appropriate GPU drivers.

```shell
$ pip install pyopencl
```

## Usage

The package provides a simple and well documented Python API to create, train
and apply a `RandomForestLocalizer`. You can [find the documentation here](https://fhkiel-mlaip.github.io/rfl).

Additionally, it also provides a command line interface (CLI) that provides easy
access to nearly all actions and parameters. For more information consult the
help output of the `rfl` command:

```shell
$ rfl --help
```
