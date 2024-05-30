"""
.. include:: ../README.md
   :start-line: 3
"""

__all__ = (
    'Potential',
    'UnaryPotential',
    'BinaryPotential',
    'TernaryPotential',
    'RteLocalizer',
    'MultiPartCNN',
    'VectorPotential',
    'KdeVectorPotential',
    'Graph',
    'Criterion',
    'MaxDistance',
    'LocResult',
    'Sample',
    'FixedGraphSample',
    'LearnableGraphSample',
    'main',
    'load',
    'working_dir'
)


import tensorflow.compat.v1 as tf

from .potentials import Potential, UnaryPotential, BinaryPotential, \
    TernaryPotential
from .potentials import RteLocalizer, MultiPartCNN
from .potentials import VectorPotential, KdeVectorPotential
from .graph import Graph
from .evaluation import Criterion, LocResult, MaxDistance
from .learning import Sample, FixedGraphSample, LearnableGraphSample
from .cli import main
from .utils import working_dir

try:
    # silence the annoying contrib warning!
    if type(tf.contrib) != type(tf):
        tf.contrib._warning = None
except AttributeError:
    pass

load = Graph.load
"""Loads an existing `Graph` model from disk (or memory)."""
