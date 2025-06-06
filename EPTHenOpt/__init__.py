# EPTHenOpt/__init__.py

"""
EPTHenOpt: A Python package for Heat Exchanger Network (HEN) Synthesis and Optimization.

This package provides tools for modeling HEN problems and solving them using
metaheuristic algorithms like Genetic Algorithm (GA) and Teaching-Learning-Based
Optimization (TLBO).
"""

# Import key classes from data model modules
from .hen_models import (
    Stream, Utility, CostParameters, HENProblem
)

# Import the main optimizer classes from their respective algorithm modules
from .ga_helpers import GeneticAlgorithmHEN
from .tlbo_helpers import TeachingLearningBasedOptimizationHEN

# Import the base optimizer class for users who might want to extend the package
from .base_optimizer import BaseOptimizer

# Import the most useful utility functions
from .utils import (
    load_data_from_csv,
    calculate_lmtd,
    find_stream_index_by_id,
    display_optimization_results,
    display_problem_summary,
    display_help,
)

# Import the core parallel execution function
from .cores import run_parallel_with_migration

# Define the public API of the package using __all__.
__all__ = [
    # Models
    'Stream', 'Utility', 'CostParameters', 'HENProblem',

    # Optimizers
    'GeneticAlgorithmHEN', 'TeachingLearningBasedOptimizationHEN',

    # Base Class
    'BaseOptimizer',

    # Utilities
    'load_data_from_csv', 'calculate_lmtd', 'find_stream_index_by_id',
    'display_optimization_results', 'display_problem_summary', 'display_help',

    # Core Execution
    'run_parallel_with_migration',
]

__version__ = "0.2.2" # Incremented version for new structure
__author__ = "Maetee Lorprajuksiri (26008353@pttgcgroup.com) E-PT-PX Department, GC Maintenance and Engineering Co. Ltd."
