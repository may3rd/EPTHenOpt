# EPTHenOpt/__init__.py

"""
EPTHenOpt: A Python package for Heat Exchanger Network (HEN) Synthesis and Optimization.

This package provides tools for modeling HEN problems and solving them using
metaheuristic algorithms like Genetic Algorithm (GA) and Teaching-Learning-Based
Optimization (TLBO).
"""
import argparse
from types import SimpleNamespace

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
    'run_parallel_with_migration', 'run'
]

__version__ = "0.3.0"
__author__ = "Maetee Lorprajuksiri (26008353@pttgcgroup.com) E-PT-PX Department, GC Maintenance and Engineering Co. Ltd."

def run(**kwargs):
    """
    High-level programmatic API to run a HEN optimization.

    This function provides a simple way to configure and run an optimization
    by passing parameters as keyword arguments.

    Args:
        **kwargs: Keyword arguments corresponding to the command-line options
                  in run_problem.py. For example:
                  model='GA', epochs=10, population_size=200, etc.
    """
    # Import inside the function to avoid circular dependencies at package import time.
    from .run_problem import main as run_problem_main

    # Create a SimpleNamespace object to mimic the args object from argparse
    # This allows us to reuse the existing main logic from run_problem.py
    # Set default values for all possible arguments
    defaults = {
        'streams_file': "streams.csv",
        'utilities_file': "utilities.csv",
        'matches_U_file': None,
        'forbidden_matches_file': None,
        'required_matches_file': None,
        'model': 'GA',
        'population_size': 200,
        'epochs': 10,
        'generations_per_epoch': 20,
        'number_of_workers': 1,
        'num_stages': 0,
        'noverbose': False,
        'EMAT_setting': 3.0,
        'default_U_overall': 0.5,
        'default_exch_fixed_cost': 0.0,
        'default_exch_area_coeff': 1000.0,
        'default_exch_area_exp': 0.6,
        'ga_crossover_prob': 0.85,
        'ga_mutation_prob_Z_setting': 0.1,
        'ga_mutation_prob_R_setting': 0.1,
        'ga_r_mutation_std_dev_factor_setting': 0.1,
        'ga_elitism_frac': 0.1,
        'ga_tournament_size': 3,
        'tlbo_teaching_factor': 0,
        'utility_cost_factor': 1.0,
        'pinch_dev_penalty_factor': 150.0,
        'sws_max_iter': 300,
        'sws_conv_tol': 1e-5,
        'initial_penalty': 1e3,
        'final_penalty': 1e7
    }

    # Update defaults with user-provided kwargs
    defaults.update(kwargs)
    args = SimpleNamespace(**defaults)

    # Call the main logic from run_problem.py
    run_problem_main(args)
